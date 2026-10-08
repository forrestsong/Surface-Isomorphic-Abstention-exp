"""S1 奇点探测（第一层：全局扫描）。

融合两条独立探针：

  A. **AWQ 激活显著性**（AWQ 成熟技术的核心量，作为奇点探针）
     - 对每个 nn.Linear 累积逐**输入通道**的 E|x_j|、sqrt(E x_j²)、峰度
     - 导出每层「显著性集中度」：top-1% 通道能量占比、有效通道数(参与比)、超通道数
     - 高集中度 ⇒ 少数方向承载几乎全部信号 = **表示折叠型奇点**
     - 对 MLP down_proj 的输入额外记录「超级激活」通道（tech plan §3.4 的超权重锚点候选）

  B. **谱/死方向签名 DDS**（tech plan §3.2）
     - 对每个权重矩阵取 top-k 奇异值谱（随机化 SVD）
     - stable rank = ‖W‖_F²/σ₁²、top-1 能量占比、有效秩、谱间隙
     - 低 stable rank / 高 top-1 占比 ⇒ 参数空间**秩亏** = 参数简并型奇点

  C. **StarNet(DeltaNet) 状态转移谱**
     - g = -exp(A_log)·softplus(a + dt_bias)，逐头衰减 exp(g)
     - 有效时间尺度数(参与比) 低 ⇒ 多头功能简并 = **状态谱简并型奇点**

输出：out/s1_probe.json + out/s1_probe.npz（逐通道明细）。
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    MODEL_DIR, CALIB_PATH, load_model, get_text_backbone, layer_types,
    find_linears, ChannelActivationAccumulator, load_calib_texts,
    effective_weight, _lora_delta,
)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# 逐层聚合工具
# ══════════════════════════════════════════════════════════════════════════

def parse_linear_name(name: str):
    """'layers.12.mlp.down_proj' → (12, 'mlp.down_proj')。"""
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "layers":
        return int(parts[1]), ".".join(parts[2:])
    return -1, name


def effective_dim(m: torch.Tensor) -> float:
    """参与比 = (Σm)² / Σm² —— 有效活跃通道数。"""
    s = m.clamp_min(0).double()
    denom = (s * s).sum()
    if denom <= 0:
        return 0.0
    return float((s.sum() ** 2) / denom)


def top_share(m: torch.Tensor, frac: float) -> float:
    """幅度最大的 frac 比例通道占 L1 总量之比。"""
    s = m.clamp_min(0).double().flatten()
    if s.sum() <= 0:
        return 0.0
    k = max(1, int(round(frac * s.numel())))
    top = torch.topk(s, k).values.sum()
    return float(top / s.sum())


def gini(m: torch.Tensor) -> float:
    s = m.clamp_min(0).double().flatten().sort().values
    n = s.numel()
    if n == 0 or s.sum() <= 0:
        return 0.0
    idx = torch.arange(1, n + 1, dtype=torch.float64, device=s.device)
    return float((2 * (idx * s).sum() / (n * s.sum())) - (n + 1) / n)


def spectral_metrics(W: torch.Tensor, k: int = 48) -> dict:
    """随机化 SVD 取 top-k 谱，算秩亏签名。"""
    Wf = W.detach().float()
    fro2 = float((Wf * Wf).sum())
    if fro2 <= 0:
        return {}
    q = min(k, min(Wf.shape) - 1)
    try:
        U, S, V = torch.svd_lowrank(Wf, q=q, niter=3)
    except Exception:
        return {}
    S = S.clamp_min(0)
    s1 = float(S[0]) if S.numel() else 0.0
    s2 = float(S[1]) if S.numel() > 1 else 0.0
    e = S.double() ** 2
    tot = e.sum()
    out = {
        "sigma_max": s1,
        "fro_norm": fro2 ** 0.5,
        "stable_rank": (fro2 / (s1 * s1)) if s1 > 0 else float("inf"),
        "top1_energy_share": float(e[0] / tot) if tot > 0 else 0.0,
        "topk_energy_share": float(e.sum() / fro2) if fro2 > 0 else 0.0,
        "cond_topk": (s1 / float(S[-1])) if float(S[-1]) > 0 else float("inf"),
        "spectral_gap_1_2": (s1 / s2) if s2 > 0 else float("inf"),
        "n_spectrum": int(S.numel()),
    }
    p = e / tot.clamp_min(1e-30)
    p = p[p > 0]
    out["effective_rank_k"] = float(torch.exp(-(p * p.log()).sum()))
    return out


# ══════════════════════════════════════════════════════════════════════════
# DeltaNet 衰减谱
# ══════════════════════════════════════════════════════════════════════════

class DecayAccumulator:
    """hook linear_attn.in_proj_a 的输出，累积逐头衰减 exp(g) 的统计。"""

    def __init__(self, backbone: nn.Module):
        self.handles = []
        self.data = {}   # layer_idx -> {"sum": (H,), "sumsq": (H,), "n": float}
        for i, layer in enumerate(backbone.layers):
            attn = getattr(layer, "linear_attn", None)
            if attn is None or not hasattr(attn, "in_proj_a"):
                continue
            self.handles.append(
                attn.in_proj_a.register_forward_hook(self._make(i, attn))
            )

    def _make(self, i, attn):
        def hook(module, inputs, output):
            import torch.nn.functional as F
            a = output.detach().float()
            A = attn.A_log.detach().float().exp()
            g = -A * F.softplus(a + attn.dt_bias.detach().float())
            decay = torch.exp(g)                       # (B, L, H)
            d = decay.reshape(-1, decay.shape[-1])
            rec = self.data.setdefault(
                i, {"sum": torch.zeros(d.shape[-1], dtype=torch.float64, device=d.device),
                    "sumsq": torch.zeros(d.shape[-1], dtype=torch.float64, device=d.device),
                    "n": 0.0, "all": []})
            rec["sum"] += d.double().sum(0)
            rec["sumsq"] += (d.double() ** 2).sum(0)
            rec["n"] += d.shape[0]
            # 采样保存原始衰减向量（限制体积），用于时间尺度熵
            if len(rec["all"]) < 20000:
                rec["all"].append(d[:: max(1, d.shape[0] // 256)].cpu())
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def result(self):
        out = {}
        for i, rec in self.data.items():
            n = rec["n"] or 1.0
            mean = rec["sum"] / n
            var = (rec["sumsq"] / n) - mean * mean
            allc = torch.cat(rec["all"], 0) if rec["all"] else torch.zeros(1, 1)
            # 有效时间尺度数：对**头平均衰减**做参与比
            pr = effective_dim(mean)
            # 逐 token 的头间变异（若某 token 上多头衰减几乎相同 ⇒ 头简并）
            head_std = allc.std(dim=-1).mean().item() if allc.numel() > 1 else 0.0
            out[i] = {
                "mean_decay_per_head": mean.cpu().tolist(),
                "decay_mean": float(mean.mean()),
                "decay_std_across_heads": float(mean.std()),
                "effective_timescales": pr,
                "n_heads": int(mean.numel()),
                "head_degeneracy": pr / max(mean.numel(), 1),
                "per_token_head_std": float(head_std),
            }
        return out


# ══════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seq", type=int, default=256, help="标定序列数")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--svd_k", type=int, default=48)
    ap.add_argument("--adapter", default=None,
                    help="可选的 LoRA adapter 目录（编辑后复跑用）")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    print(f"[S1] 加载模型 ...")
    model, tokenizer = load_model(MODEL_DIR, precision="bf16")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()
        print(f"[S1] 已加载 adapter: {args.adapter}")
    bb = get_text_backbone(model)
    lt = layer_types(model)
    print(f"[S1] 加载 {time.time()-t0:.1f}s  层={len(bb.layers)}")

    linears = find_linears(model)
    print(f"[S1] Linear 数 = {len(linears)}")

    # ── A. 激活显著性 ────────────────────────────────────────────────
    acc = ChannelActivationAccumulator(linears)
    decay_acc = DecayAccumulator(bb)

    texts = load_calib_texts(CALIB_PATH, args.n_seq)
    print(f"[S1] 标定序列 {len(texts)} 条, L={args.seq_len}")
    n_tok = 0
    t1 = time.time()
    with torch.no_grad():
        for i, txt in enumerate(texts):
            enc = tokenizer(txt, return_tensors="pt", truncation=True,
                            max_length=args.seq_len)
            if enc["input_ids"].shape[1] < 4:
                continue
            enc = {k: v.cuda() for k, v in enc.items()}
            n_tok += enc["input_ids"].shape[1]
            model(**enc)
            if (i + 1) % 32 == 0:
                print(f"       {i+1}/{len(texts)}  {time.time()-t1:.0f}s  "
                      f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB")
    acc.remove()
    decay_acc.remove()
    print(f"[S1] 采集完成 {time.time()-t1:.0f}s, 总 token={n_tok}")

    act = acc.result()
    decay = decay_acc.result()

    # ── 逐 Linear 显著性指标 ─────────────────────────────────────────
    per_linear = {}
    npz_save = {}
    for name, r in act.items():
        layer, role = parse_linear_name(name)
        m = r["mean_abs"]
        d = {
            "layer": layer,
            "role": role,
            "in_features": int(m.numel()),
            "mean_abs_mean": float(m.mean()),
            "mean_abs_max": float(m.max()),
            "rms_mean": float(r["rms"].mean()),
            "effective_dim": effective_dim(m),
            "effective_dim_frac": effective_dim(m) / m.numel(),
            "top1pct_share": top_share(m, 0.01),
            "top0p1pct_share": top_share(m, 0.001),
            "gini": gini(m),
            "kurt_like_mean": float(r["kurt_like"].mean()),
            "kurt_like_max": float(r["kurt_like"].max()),
            "n_super_channels_20x_median": int(
                (m > 20 * m.median()).sum()) if m.numel() else 0,
            "max_abs_max": float(r["max_abs"].max()),
        }
        per_linear[name] = d
        if layer >= 0:
            npz_save[f"act__{name.replace('.', '_')}"] = m.cpu().numpy()

    # ── B. 权重谱 ────────────────────────────────────────────────────
    print("[S1] 计算权重谱 (DDS) ...")
    t2 = time.time()
    per_weight = {}
    n_lora_wrapped = 0
    n_lora_delta_nonzero = 0
    lora_layers = set()
    for name, mod in linears.items():
        layer, role = parse_linear_name(name)
        # ★ 用**等效权重** W_base + ΔW_lora：未编辑层 ΔW=0（与旧读数逐位一致），
        #   被编辑层则反映编辑后的真实权重谱。
        Weff = effective_weight(mod)
        sm = spectral_metrics(Weff, k=args.svd_k)
        if not sm:
            continue
        sm["layer"] = layer
        sm["role"] = role
        sm["shape"] = list(Weff.shape)
        delta = _lora_delta(mod)
        if delta is not None:
            n_lora_wrapped += 1
            nz = bool(float(delta.abs().max()) > 0)
            sm["is_lora_wrapped"] = True
            sm["lora_delta_nonzero"] = nz
            if nz:
                n_lora_delta_nonzero += 1
                lora_layers.add(layer)
        per_weight[name] = sm
    print(f"[S1] 谱完成 {time.time()-t2:.0f}s  "
          f"LoRA 包装层={n_lora_wrapped} 其中 ΔW≠0 者={n_lora_delta_nonzero} "
          f"涉及层={sorted(x for x in lora_layers if x >= 0)}")

    meta = {
        "model_dir": MODEL_DIR,
        "adapter": args.adapter,
        "n_lora_wrapped_linears": n_lora_wrapped,
        "n_lora_delta_nonzero": n_lora_delta_nonzero,
        "lora_edited_layers": sorted(x for x in lora_layers if x >= 0),
        "n_layers": len(bb.layers),
        "layer_types": lt,
        "full_attention_layers": [i for i, t in enumerate(lt) if t == "full_attention"],
        "n_linears": len(linears),
        "n_calib_seq": len(texts),
        "calib_tokens": n_tok,
        "seq_len": args.seq_len,
        "svd_k": args.svd_k,
        "elapsed_s": time.time() - t0,
    }
    out = {"meta": meta, "per_linear": per_linear, "per_weight": per_weight,
           "deltanet_decay": decay}

    with open(os.path.join(args.out, "s1_probe.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    import numpy as np
    np.savez_compressed(os.path.join(args.out, "s1_probe.npz"), **npz_save)
    print(f"[S1] 写出 {args.out}/s1_probe.json + .npz  (总 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

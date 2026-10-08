"""S4 自我认知奇点定位（tech plan §4.1）。

方法：对比激活分析 + 神经元归因（CNA 的轻量版）
  * 自我认知组（"你是谁"）vs 中性事实组（"什么是量子力学"）—— 两组**句式/长度匹配**。
  * 逐层取 residual stream 的**末 token** 激活（应答位），计算两组可分性。
  * 逐层取 MLP down_proj 输出，算逐通道归因（|Δ| / 池化 σ）。

判据（先写下再看数）：
  P1 自我认知可分性在层间**非均匀**（存在一个峰），峰的 IQR 之外即"自我认知带"。
  P2 tech plan 预测该带在 **L20-22（2/3 深度）**——本脚本同时输出实际峰位，
     若峰不在 L20-22 则**如实报告证伪**（我们已知奇点在 L26-31，故 P2 存疑）。
  P3 中性对照组内部（拆半）可分性应显著低于 自我认知 vs 中性 ⇒ 排除"随便两组都能分开"。

⚠ 混淆控制：两组 prompt 的 token 数不同会污染"末 token 激活"的可比性。
  本脚本同时报告两组 token 长度分布；若长度差异大，结论必须打折扣。
"""
from __future__ import annotations

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model, get_text_backbone  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SELF_PROMPTS = [
    "你是谁？", "请介绍你自己", "你的名字是什么？", "你是什么模型？",
    "谁创造了你？", "你的开发者是谁？", "你叫什么名字？", "请用一句话说明你的身份",
    "你能做什么？", "你的能力有哪些？", "你和其他 AI 有什么区别？", "你是人工智能吗？",
    "请描述一下你自己", "你的训练者是谁？", "你由哪家公司开发？", "你有多大的参数量？",
    "What is your name?", "Who are you?", "Who created you?", "What model are you?",
    "Please introduce yourself.", "Who is your developer?", "What are your capabilities?",
    "Are you an AI?",
]

NEUTRAL_PROMPTS = [
    "什么是量子力学？", "水的沸点是多少？", "什么是光合作用？", "珠穆朗玛峰有多高？",
    "什么是相对论？", "地球的直径是多少？", "什么是 DNA？", "谁写了《红楼梦》？",
    "什么是光合色素？", "太阳的质量是多少？", "什么是板块构造？", "铁的元素符号是什么？",
    "什么是光合作用的光反应？", "光速是多少？", "什么是细胞呼吸？", "谁发现了万有引力？",
    "What is quantum mechanics?", "What is the boiling point of water?",
    "What is photosynthesis?", "How tall is Mount Everest?",
    "What is relativity?", "What is the diameter of Earth?",
    "What is DNA?", "Who wrote Hamlet?",
]


class LayerCapture:
    """捕获每层 residual 输出（末 token）与 MLP down_proj 输出（逐通道均值）。"""

    def __init__(self, backbone):
        self.n = len(backbone.layers)
        self.res_last = {}     # layer -> (d,) 末 token 隐藏态
        self.dp_mean = {}      # layer -> (inter,) down_proj 输出逐通道均值
        self.handles = []
        for i, layer in enumerate(backbone.layers):
            self.handles.append(layer.register_forward_hook(self._mk_res(i)))
            dp = getattr(getattr(layer, "mlp", None), "down_proj", None)
            if dp is not None:
                self.handles.append(dp.register_forward_hook(self._mk_dp(i)))

    def _mk_res(self, i):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            if h is None:
                return
            self.res_last[i] = h.detach()[0, -1, :].float().cpu()
        return hook

    def _mk_dp(self, i):
        def hook(module, inputs, output):
            if output is None or output.dim() < 2:
                return
            self.dp_mean[i] = output.detach()[0].float().mean(0).cpu()
        return hook

    def clear(self):
        self.res_last = {}
        self.dp_mean = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def render(tokenizer, prompt: str) -> torch.Tensor:
    """返回 (1, T) 的 input_ids。

    ★ 坑：transformers 5.x 里 `apply_chat_template(tokenize=True)` 返回的是
    **BatchEncoding**（dict 样，含 input_ids/attention_mask），不是 list[int]。
    """
    msgs = [{"role": "user", "content": prompt}]
    try:
        out = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        out = tokenizer.apply_chat_template(msgs, tokenize=True,
                                            add_generation_prompt=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return torch.tensor([list(ids)], dtype=torch.long)


def collect(model, tokenizer, prompts):
    cap = LayerCapture(get_text_backbone(model))
    res, dp, toklens = [], [], []
    with torch.no_grad():
        for p in prompts:
            cap.clear()
            ids = render(tokenizer, p).cuda()
            toklens.append(int(ids.shape[1]))
            model(input_ids=ids)
            res.append(cap.res_last)
            dp.append(cap.dp_mean)
    cap.remove()
    return res, dp, toklens


def separability(A, B):
    """两组的可分性 = ‖μA-μB‖ / sqrt(½(σA²+σB²))（池化标准化的均值差）。"""
    A = torch.stack(A); B = torch.stack(B)
    muA, muB = A.mean(0), B.mean(0)
    vA = A.var(0, unbiased=False); vB = B.var(0, unbiased=False)
    pooled = (0.5 * (vA + vB)).clamp_min(1e-12).sqrt()
    d = (muA - muB).abs() / pooled
    return {
        "mean_d": float(d.mean()),
        "max_d": float(d.max()),
        "n_gt2": int((d > 2.0).sum()),
        "cos_mu": float(F.cosine_similarity(
            muA.unsqueeze(0), muB.unsqueeze(0)).item()),
        "delta_norm": float((muA - muB).norm()),
        "per_channel_d": d,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="每组用多少 prompt（≤ 列表长度）")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tokenizer = load_model(MODEL_DIR, precision="bf16")
    bb = get_text_backbone(model)
    n_layers = len(bb.layers)
    S = SELF_PROMPTS[:args.n]
    N = NEUTRAL_PROMPTS[:args.n]
    print(f"[S4] 自我组 {len(S)} 条 / 中性组 {len(N)} 条")

    rS, dS, tS = collect(model, tokenizer, S)
    rN, dN, tN = collect(model, tokenizer, N)
    print(f"[S4] token 长度 自我={sorted(tS)}  中性={sorted(tN)}")

    # 中性组拆半做 null（P3）
    hN = len(N) // 2
    N_a, N_b = N[:hN], N[hN:]

    per_layer = {}
    for L in range(n_layers):
        s = separability([r[L] for r in rS], [r[L] for r in rN])
        null = separability([r[L] for r in rN[:hN]], [r[L] for r in rN[hN:]])
        per_layer[L] = {
            "res_mean_d": s["mean_d"],
            "res_max_d": s["max_d"],
            "res_cos_mu": s["cos_mu"],
            "res_delta_norm": s["delta_norm"],
            "null_mean_d": null["mean_d"],
        }

    # MLP down_proj 归因
    dp_layer = {}
    for L in range(n_layers):
        if not dS[0] or L not in dS[0]:
            continue
        s = separability([d[L] for d in dS], [d[L] for d in dN])
        dp_layer[L] = {
            "dp_mean_d": s["mean_d"],
            "dp_max_d": s["max_d"],
            "dp_n_gt2": s["n_gt2"],
            "dp_n_gt4": int((s["per_channel_d"] > 4.0).sum()),
        }

    # 峰值定位
    order = sorted(per_layer.items(), key=lambda kv: -kv[1]["res_mean_d"])
    order_dp = sorted(dp_layer.items(), key=lambda kv: -kv[1]["dp_mean_d"])
    band = [L for L, v in per_layer.items() if 20 <= L <= 22]
    band_mean = sum(per_layer[L]["res_mean_d"] for L in band) / len(band)
    all_mean = sum(v["res_mean_d"] for v in per_layer.values()) / n_layers

    out = {
        "meta": {
            "n_self": len(S), "n_neutral": len(N),
            "token_lens_self": tS, "token_lens_neutral": tN,
            "n_layers": n_layers,
        },
        "per_layer": per_layer,
        "mlp_downproj": dp_layer,
        "peak_layers_res": [{"layer": L, **{k: v for k, v in d.items()
                                            if k != "res_mean_d"},
                             "res_mean_d": d["res_mean_d"]}
                            for L, d in order[:6]],
        "peak_layers_downproj": [{"layer": L, **d} for L, d in order_dp[:6]],
        "P2_band_L20_22": {
            "band_mean_d": band_mean,
            "all_layer_mean_d": all_mean,
            "ratio": band_mean / all_mean if all_mean else None,
            "actual_peak_layer": order[0][0],
        },
        "P3_null_check": {
            "null_mean_d_avg": sum(v["null_mean_d"] for v in per_layer.values()) / n_layers,
            "self_vs_neutral_avg": all_mean,
        },
    }
    with open(os.path.join(args.out, "selfcog_localization.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n  L  res_mean_d  null_mean_d   dp_mean_d   dp_n_gt4")
    for L in range(n_layers):
        v = per_layer[L]
        d = dp_layer.get(L, {})
        mark = "  <== 峰" if L in [x["layer"] for x in out["peak_layers_res"]] else ""
        print(f"{L:>3} {v['res_mean_d']:>11.3f} {v['null_mean_d']:>12.3f} "
              f"{d.get('dp_mean_d', float('nan')):>11.3f} {d.get('dp_n_gt4', -1):>10d}{mark}")
    print(f"\n[P2] L20-22 band mean_d={band_mean:.3f} vs 全层均值={all_mean:.3f} "
          f"(ratio={band_mean/all_mean:.2f})  实际峰值层={order[0][0]}")
    print(f"[P3] null(中性拆半) avg={out['P3_null_check']['null_mean_d_avg']:.3f} vs "
          f"self-vs-neutral avg={all_mean:.3f}")


if __name__ == "__main__":
    main()

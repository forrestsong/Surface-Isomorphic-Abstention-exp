"""G3-0 目标3 前置：§5.2「更新是否向主奇异子空间漂移」可测性 + 自然尺度标定。

技术方案 §5.2 要求追踪「更新矩阵在主奇异子空间上的能量占比」，超过 15% 触发谱校准。
但**在采用任何阈值之前，必须先测出该指标的自然尺度（null）**——否则 15% 无 referent
（本项目铁律：null/ceiling 优先，阈值必须按实测分辨率设定）。

本脚本不需要前向：直接读 adapter 的 A/B 与基座 W，算
    ΔW = (alpha/r) · B @ A
然后对每个模块算两个「主方向能量占比」：

  A) **主奇异子空间占比**（§5.2 的字面指标）
     V_k = 基座 W 的 top-k 右奇异向量（输入空间）
     share = ‖ΔW·V_k‖_F² / ‖ΔW‖_F²          null = k / in_features
  B) **显著通道占比**（目标1 的 AWQ 显著性给出的「奇点坐标」）
     S = 该层 mean|x| 的 top-1% 通道
     share = ‖ΔW[:,S]‖_F² / ‖ΔW‖_F²          null = 0.01

两个 null 都是**闭式**的，无需额外实验。另外加一个**随机对照**（同形状、同 Frobenius
范数的随机 ΔW）来验证仪器本身：它必须回落到 null 附近，否则指标有偏。
"""
from __future__ import annotations

import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from safetensors import safe_open

MODEL = "/home/songsen/workspace/LivingFormer/Qwen3.5Finetuning/Qwen3.5-9B/master"
ADAPTER = "out/lora_selfcog/adapter_model.safetensors"
OUT = "out"
K_SVD = 64          # top-k 奇异方向
SALIENT_FRAC = 0.01  # top-1%
SEED = 0


def load_index():
    with open(os.path.join(MODEL, "model.safetensors.index.json"), encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def get_base_tensor(index, name):
    shard = index[name]
    with safe_open(os.path.join(MODEL, shard), framework="pt") as f:
        return f.get_tensor(name)


def main():
    wp = load_index()
    npz = np.load(f"{OUT}/s1_probe.npz")

    with safe_open(ADAPTER, framework="pt") as f:
        akeys = [k for k in f.keys() if k.endswith("lora_A.weight")]
        ad = {k: f.get_tensor(k).float() for k in f.keys()}

    # 带宽 alpha/r
    with open("out/lora_selfcog/adapter_config.json", encoding="utf-8") as f:
        acfg = json.load(f)
    scaling = acfg["lora_alpha"] / acfg["r"]
    print(f"[G3-0] 模块数={len(akeys)}  scaling=alpha/r={scaling:g}  K_SVD={K_SVD}")

    g = torch.Generator().manual_seed(SEED)
    rows = []
    for ka in sorted(akeys):
        mod = ka[len("base_model.model."):-len(".lora_A.weight")]
        kb = ka.replace("lora_A", "lora_B")
        A = ad[ka]                      # (r, in)
        B = ad[kb]                      # (out, r)
        # ★ SVD 必须放 GPU：12288×4096 的 svd_lowrank 在 CPU 上每矩阵数十秒。
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        W = get_base_tensor(wp, mod + ".weight").float().to(dev)
        dW = (scaling * (B @ A)).to(dev)   # (out, in)

        out_f, in_f = W.shape
        # ── A) 主奇异子空间（基座 W 的 top-k 右奇异向量）─────────────
        q = min(K_SVD, min(W.shape) - 1)
        _, _, V = torch.svd_lowrank(W, q=q, niter=3)
        proj = dW @ V                                  # (out, k)
        share_prin = float(proj.pow(2).sum() / dW.pow(2).sum())
        null_prin = q / in_f

        # ── B) 显著通道（目标1 的 AWQ 显著性 top-1%）─────────────────
        # 模块名带前缀 model.language_model. ⇒ 需转成主干相对名才能对上 npz 键
        rel = mod.split("model.language_model.")[-1]
        key = "act__" + rel.replace(".", "_")
        if key in npz.files:
            m = npz[key].astype(np.float64)
            k_s = max(1, int(round(SALIENT_FRAC * m.size)))
            S = torch.tensor(np.argsort(-m)[:k_s].copy(), dtype=torch.long)
            share_sal = float(dW[:, S].pow(2).sum() / dW.pow(2).sum())
            n_sal = len(S)
        else:
            share_sal, n_sal = float("nan"), 0
        null_sal = n_sal / in_f if n_sal else float("nan")

        # ── 随机对照（同形状/同 Frobenius）────────────────────────────
        rnd = torch.randn(dW.shape, generator=g).to(dW.device)
        rnd = rnd / rnd.norm() * dW.norm()
        r_prin = float((rnd @ V).pow(2).sum() / rnd.pow(2).sum())
        r_sal = (float(rnd[:, S].pow(2).sum() / rnd.pow(2).sum())
                 if n_sal else float("nan"))

        parts = rel.split(".")
        layer = int(parts[1]) if len(parts) > 1 and parts[0] == "layers" else -1
        role = ".".join(parts[2:])
        rows.append({
            "module": mod, "layer": layer, "role": role,
            "shape": [out_f, in_f], "scaling": scaling,
            "dw_fro": float(dW.norm()),
            "share_principal": share_prin, "null_principal": null_prin,
            "enrich_principal": share_prin / null_prin if null_prin else None,
            "rand_principal": r_prin,
            "share_salient": share_sal, "null_salient": null_sal,
            "enrich_salient": (share_sal / null_sal
                               if null_sal and null_sal == null_sal else None),
            "rand_salient": r_sal,
            "n_salient_channels": n_sal,
        })
        print(f"  L{layer:02d} {role:<22} 主奇异占比={share_prin:6.2%} "
              f"(null {null_prin:5.2%}, ×{share_prin/null_prin:5.2f}, "
              f"rand {r_prin:5.2%})  显著通道占比={share_sal:6.2%} "
              f"(null {null_sal:4.2%}, ×{share_sal/null_sal:5.2f}, rand {r_sal:5.2%})"
              if n_sal else f"  L{layer:02d} {role:<22} (无显著性数据)")

    # ── 聚合 ────────────────────────────────────────────────────────
    def agg(key):
        v = np.array([r[key] for r in rows if r[key] == r[key]], float)
        return (float(v.mean()), float(v.std()), float(v.min()), float(v.max()),
                int(v.size))

    res = {}
    for k in ("share_principal", "enrich_principal", "rand_principal",
              "share_salient", "enrich_salient", "rand_salient"):
        m, s, lo, hi, n = agg(k)
        res[k] = {"mean": m, "std": s, "min": lo, "max": hi, "n": n}
    res["_K_SVD"] = K_SVD
    res["_salient_frac"] = SALIENT_FRAC
    res["_plan_threshold_15pct"] = 0.15
    res["_n_modules_with_share_gt_15pct"] = int(
        sum(1 for r in rows if r["share_principal"] > 0.15))
    res["_n_modules_with_share_gt_15pct_salient"] = int(
        sum(1 for r in rows if r["share_salient"] == r["share_salient"]
            and r["share_salient"] > 0.15))

    with open(f"{OUT}/g3_subspace.json", "w", encoding="utf-8") as f:
        json.dump({"per_module": rows, "aggregate": res,
                   "plan_metric": "§5.2 主奇异子空间能量占比，方案建议阈值 15%"},
                  f, ensure_ascii=False, indent=2)

    print("\n=== 聚合 ===")
    for k in ("share_principal", "rand_principal", "share_salient", "rand_salient"):
        a = res[k]
        print(f"{k:<18} mean={a['mean']:7.3%} sd={a['std']:6.3%} "
              f"min={a['min']:6.3%} max={a['max']:6.3%} n={a['n']}")
    print(f"\n主奇异子空间占比 > 15%（方案阈值）的模块数："
          f"{res['_n_modules_with_share_gt_15pct']}/{len(rows)}")
    print(f"[G3-0] 写出 {OUT}/g3_subspace.json")


if __name__ == "__main__":
    main()

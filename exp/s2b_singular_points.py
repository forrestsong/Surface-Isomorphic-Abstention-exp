"""S2b 奇异点**坐标**提取（tech plan §3.5「超权重坐标列表」）。

从 s1 的逐通道 mean|x| 明细里取出：
  * 每层关键投影的 top-K 通道坐标（layer, role, channel_index, mean|x|, 相对倍数）
  * 「超通道」：mean|x| > 20× 该层中位数的通道（AWQ 意义上被保护的显著通道）

判据/参考：mean|x| 均匀时 top-1% 通道恰占 1% 能量 ⇒ null = 0.01。
"""
from __future__ import annotations

import os
import json
import numpy as np

OUT = "out"
KEYS_OF_INTEREST = ("mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
                    "self_attn.o_proj", "linear_attn.out_proj",
                    "self_attn.v_proj", "self_attn.q_proj",
                    "linear_attn.in_proj_qkv")


def main():
    with open(os.path.join(OUT, "s1_probe.json"), encoding="utf-8") as f:
        d = json.load(f)
    npz = np.load(os.path.join(OUT, "s1_probe.npz"))

    per_linear = d["per_linear"]
    points = []
    for name, v in per_linear.items():
        role = v["role"]
        if role not in KEYS_OF_INTEREST:
            continue
        key = "act__" + name.replace(".", "_")
        if key not in npz.files:
            continue
        m = npz[key].astype(np.float64)
        med = float(np.median(m))
        k = 8
        idx = np.argsort(-m)[:k]
        pts = [{
            "channel": int(j),
            "mean_abs": float(m[j]),
            "x_median": float(m[j] / med) if med > 0 else None,
        } for j in idx]
        super_ch = int((m > 20 * med).sum()) if med > 0 else 0
        points.append({
            "module": name,
            "layer": v["layer"],
            "role": role,
            "in_features": v["in_features"],
            "median_mean_abs": med,
            "max_mean_abs": float(m.max()),
            "max_over_median": float(m.max() / med) if med > 0 else None,
            "n_super_channels_20x": super_ch,
            "top_channels": pts,
        })

    points.sort(key=lambda p: -(p["max_over_median"] or 0))
    out = {
        "note": "每行是一个模块的显著通道坐标；x_median = 该通道 mean|x| / 模块中位数。"
                "n_super_channels_20x = mean|x| 超中位数 20 倍的通道数（超权重锚点候选）。",
        "top_modules_by_concentration": points[:40],
        "all_modules": points,
    }
    with open(os.path.join(OUT, "singular_points.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{'module':<44} {'max/med':>8} {'n_super20x':>10}  top3 channels")
    for p in points[:18]:
        t3 = ",".join(f"c{c['channel']}({c['x_median']:.1f}x)"
                      for c in p["top_channels"][:3])
        print(f"{p['module']:<44} {p['max_over_median']:>8.1f} "
              f"{p['n_super_channels_20x']:>10d}  {t3}")


if __name__ == "__main__":
    main()

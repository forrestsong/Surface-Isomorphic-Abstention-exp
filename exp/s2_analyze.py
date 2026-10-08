"""S2 奇点图谱构建（从 s1_probe.json 出发）。

★ 度量语义纪律（本项目血泪）：
  - `top1pct_share` 的 null = **0.01**（通道幅度均匀时 top-1% 恰占 1%）。
    它落到 0.139 ⇒ 14× 均匀基线，是真·集中。
  - `stable_rank = ‖W‖_F²/σ₁²` 的 null = **min(out,in)**（满秩）。远低于它 ⇒ 秩亏。
  - DeltaNet 衰减：**参与比高 ≠ 头多样**。若所有头衰减都 ≈0.9，参与比 ≈32（顶格），
    但那正是**头简并**（全体同一时间尺度）。⇒ 用**跨头变异系数 CV** 度量时间尺度多样性，
    CV 低 = 简并。⚠ 这是"度量方向反了"的经典陷阱，必须在报告里写明。
  - 复合分数只是**排序便利**，不作为证据；结论必须由**多个独立判据**同时成立支撑。
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np

OUT = "out"


def load(outdir=None):
    d0 = outdir or OUT
    with open(os.path.join(d0, "s1_probe.json"), encoding="utf-8") as f:
        d = json.load(f)
    z = np.load(os.path.join(d0, "s1_probe.npz"))
    return d, z


def agg_by_layer(d):
    pl, pw = d["per_linear"], d["per_weight"]
    layers = {}
    for name, v in pl.items():
        L = v["layer"]
        if L < 0:
            continue
        layers.setdefault(L, {"act": {}, "weight": {}})
        layers[L]["act"][v["role"]] = v
    for name, v in pw.items():
        L = v["layer"]
        if L < 0:
            continue
        layers.setdefault(L, {"act": {}, "weight": {}})
        layers[L]["weight"][v["role"]] = v
    return layers


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = (np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / den) if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=OUT, help="包含 s1_probe.json/.npz 的目录")
    args = ap.parse_args()
    outdir = args.dir
    d, npz = load(outdir)
    lt = d["meta"]["layer_types"]
    full_attn = set(d["meta"]["full_attention_layers"])
    layers = agg_by_layer(d)
    dec = d["deltanet_decay"]

    rows = []
    for L in range(d["meta"]["n_layers"]):
        e = layers[L]
        act, w = e["act"], e["weight"]

        # ── 激活侧：表示折叠 ────────────────────────────────────────
        dp = act.get("mlp.down_proj", {})
        act_conc = dp.get("top1pct_share", float("nan"))          # null=0.01
        act_effdim = dp.get("effective_dim_frac", float("nan"))   # 1.0=不塌缩
        act_max = dp.get("mean_abs_max", float("nan"))            # 通道平均幅度的最大值
        act_absmax = dp.get("max_abs_max", float("nan"))          # 单 token 峰值
        # 该层所有投影的平均集中度
        act_conc_all = float(np.nanmean([v["top1pct_share"] for v in act.values()]))

        # ── 参数侧：秩亏 ────────────────────────────────────────────
        srs = [v["stable_rank"] for v in w.values()]
        w_min_sr = float(np.nanmin(srs)) if srs else float("nan")
        w_med_sr = float(np.nanmedian(srs)) if srs else float("nan")
        # 秩亏度：与满秩基线的比值（越小越亏），取最小者最严重
        rankdef = []
        for v in w.values():
            full = min(v["shape"])
            rankdef.append(v["stable_rank"] / full)
        w_rankdef = float(np.nanmin(rankdef)) if rankdef else float("nan")
        dpw = w.get("mlp.down_proj", {})
        w_dp_sr = dpw.get("stable_rank", float("nan"))

        # ── DeltaNet 时间尺度多样性 ─────────────────────────────────
        if str(L) in dec:
            md = np.asarray(dec[str(L)]["mean_decay_per_head"], float)
            cv = float(md.std() / md.mean()) if md.mean() > 0 else 0.0
            decay_mean = float(md.mean())
        else:
            cv = float("nan")
            decay_mean = float("nan")

        rows.append({
            "layer": L,
            "type": "full_attention" if L in full_attn else "linear_attention",
            "act_conc_downproj": act_conc,
            "act_effdim_frac": act_effdim,
            "act_meanabs_max": act_max,
            "act_absmax_peak": act_absmax,
            "act_conc_all_proj": act_conc_all,
            "w_min_stable_rank_frac": w_rankdef,
            "w_downproj_stable_rank": w_dp_sr,
            "w_med_stable_rank": w_med_sr,
            "dnet_timescale_cv": cv,
            "dnet_decay_mean": decay_mean,
        })

    # ── 复合排序（仅便利，非证据）──────────────────────────────────
    def z(x):
        x = np.asarray(x, float)
        m, s = np.nanmean(x), np.nanstd(x)
        return (x - m) / (s if s > 0 else 1.0)

    conc = z([r["act_conc_downproj"] for r in rows])
    collapse = z([-r["act_effdim_frac"] for r in rows])
    dead = z([-r["w_min_stable_rank_frac"] for r in rows])
    for i, r in enumerate(rows):
        r["score_activation"] = float(conc[i] + collapse[i])
        r["score_param"] = float(dead[i])
        r["score_total"] = float(conc[i] + collapse[i] + dead[i])

    rows.sort(key=lambda r: -r["score_total"])

    # ── 假设检验（tech plan §2/§3.5）────────────────────────────────
    fa = [r for r in rows if r["type"] == "full_attention"]
    la = [r for r in rows if r["type"] == "linear_attention"]
    def mean(key, group):
        return float(np.nanmean([r[key] for r in group]))
    hyp = {}
    hyp["H_A_activation_collapse_FA_vs_LA"] = {
        "act_conc_FA": mean("act_conc_downproj", fa),
        "act_conc_LA": mean("act_conc_downproj", la),
        "effdim_FA": mean("act_effdim_frac", fa),
        "effdim_LA": mean("act_effdim_frac", la),
        "param_rankdef_FA": mean("w_min_stable_rank_frac", fa),
        "param_rankdef_LA": mean("w_min_stable_rank_frac", la),
    }
    band = [r for r in rows if 20 <= r["layer"] <= 22]
    hyp["H_B_two_thirds_depth_band_L20_22"] = {
        "act_conc_band": mean("act_conc_downproj", band),
        "act_conc_all_layers": mean("act_conc_downproj", rows),
        "score_total_band_mean": mean("score_total", band),
    }
    early = [r for r in rows if r["layer"] <= 8]
    late = [r for r in rows if r["layer"] >= 26]
    hyp["H_C_superweight_early_MLP"] = {
        "act_absmax_peak_early_max": float(np.nanmax([r["act_absmax_peak"] for r in early])),
        "act_absmax_peak_late_max": float(np.nanmax([r["act_absmax_peak"] for r in late])),
        "act_conc_early_mean": mean("act_conc_downproj", early),
        "act_conc_late_mean": mean("act_conc_downproj", late),
        "verdict": ("在早期层未见超激活" if
                    np.nanmax([r["act_absmax_peak"] for r in late]) >
                    2 * np.nanmax([r["act_absmax_peak"] for r in early]) else "早期层亦存在"),
    }
    # DeltaNet 简并
    dnet = [r for r in rows if r["type"] == "linear_attention"]
    hyp["H_D_deltanet_timescale_degeneracy"] = {
        "cv_mean": mean("dnet_timescale_cv", dnet),
        "cv_min": float(np.nanmin([r["dnet_timescale_cv"] for r in dnet])),
        "most_degenerate_layers": sorted(
            [(r["layer"], round(r["dnet_timescale_cv"], 4)) for r in dnet
             if not np.isnan(r["dnet_timescale_cv"])],
            key=lambda x: x[1])[:5],
    }

    # ── 奇异点（逐通道）─────────────────────────────────────────────
    points = []
    for key in npz.files:
        arr = npz[key]
        if not key.startswith("act__"):
            continue
        points.append((key[len("act__"):], arr))
    # 逐通道明细留给 s2b（需要与准确模块名对应）；此处只存幅度指纹
    _ = points

    out = {
        "meta": d["meta"],
        "hypotheses": hyp,
        "ranked_layers": rows,
        "activation_concentration_null": 0.01,
        "deltanet_metric_note": (
            "dnet_timescale_cv 是跨头衰减的变异系数；**低 = 各头时间尺度几乎相同 = 简并**。"
            "不要用参与比（它把所有头都近似相等时误报为'32 个有效时间尺度'）。"),
    }
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "singularity_map.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ── 打印 ────────────────────────────────────────────────────────
    print(f"{'L':>3} {'type':>16} {'conc1%':>8} {'effdim':>7} {'absmax':>8} "
          f"{'rankdef':>8} {'dnetCV':>7} {'score':>7}")
    for r in sorted(rows, key=lambda r: r["layer"]):
        print(f"{r['layer']:>3} {r['type']:>16} {r['act_conc_downproj']:>8.4f} "
              f"{r['act_effdim_frac']:>7.3f} {r['act_absmax_peak']:>8.2f} "
              f"{r['w_min_stable_rank_frac']:>8.4f} "
              f"{r['dnet_timescale_cv']:>7.3f} {r['score_total']:>7.2f}")
    print("\n=== 假设检验 ===")
    print(json.dumps(hyp, ensure_ascii=False, indent=2))
    print("\n=== Top-8 奇点层（复合排序）===")
    for r in rows[:8]:
        print(f"  L{r['layer']:02d} {r['type']:>16} score={r['score_total']:+.2f} "
              f"conc={r['act_conc_downproj']:.4f} effdim={r['act_effdim_frac']:.3f} "
              f"peak={r['act_absmax_peak']:.2f}")


if __name__ == "__main__":
    main()

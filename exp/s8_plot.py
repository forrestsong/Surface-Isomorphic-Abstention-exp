"""S8 奇点图谱可视化（彩色图表）。

输出到 `../figures/`：
  fig1_singularity_map.png   主图谱：逐层指标热力图 + 逐指标剖面（含 null 基线）
  fig2_global_channels.png   「全局坐标结构」证据：in_proj_qkv 逐层×逐通道显著性热力图
  fig3_selfcog_edit.png      自我认知定位 + 编辑前后漂移 + 编辑三项判据
  fig4_llc_vs_dds.png        LLC × 奇点复合分（显示二者不一致）

设计纪律：
  * 每个标量图都画 **null 基线**（top1% 的 null=0.01、effdim 的 null=1.0）——
    没有基线的「高」不可解释。
  * 层类型用**颜色区分**（全注意力 vs DeltaNet），因为本实验的核心结论之一就是二者异质。
  * 热力图用**逐列 z 分数**（单位不可比），故必须标 diverging colormap 且注明。
"""
from __future__ import annotations

import os
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ── 中文字体（系统有 Noto Sans CJK）──────────────────────────────────
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC",
    "WenQuanYi Zen Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"

OUT = "out"
POST = "out_post"
FIG = "../figures"

C_FA = "#d62728"    # 全注意力
C_LA = "#1f77b4"    # GatedDeltaNet
C_REF = "#555555"   # 参考基线


def jload(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_all():
    sm = jload(f"{OUT}/singularity_map.json")
    post = jload(f"{POST}/singularity_map.json")
    sc = jload(f"{OUT}/selfcog_localization.json")
    ed = jload(f"{OUT}/edit_result.json")
    llc = []
    for p in sorted(glob.glob(f"{OUT}/llc_L*_lr*.json")):
        llc.append(jload(p))
    llc.sort(key=lambda r: int(r["tag"].split("_")[0].lstrip("L")))
    return sm, post, sc, ed, llc


# ══════════════════════════════════════════════════════════════════════
# fig1 主图谱
# ══════════════════════════════════════════════════════════════════════

def fig1(sm):
    rows = sorted(sm["ranked_layers"], key=lambda r: r["layer"])
    L = np.array([r["layer"] for r in rows])
    is_fa = np.array([r["type"] == "full_attention" for r in rows])

    cols = [
        ("激活集中度\ntop-1% 占比", [r["act_conc_downproj"] for r in rows], False),
        ("表示塌缩\n1 − 有效维占比", [1 - r["act_effdim_frac"] for r in rows], False),
        ("超激活峰值\nlog10(max|x|)", [np.log10(max(r["act_absmax_peak"], 1e-3)) for r in rows], False),
        ("权重秩亏\n1 − stable_rank/满秩", [1 - r["w_min_stable_rank_frac"] for r in rows], False),
        ("DeltaNet\n时间尺度 CV", [r["dnet_timescale_cv"] for r in rows], True),
        ("复合奇点分\n(z 和，仅排序用)", [r["score_total"] for r in rows], False),
    ]

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=0.55, wspace=0.28)

    # ── (a) 热力图：逐列 z 分数 ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    M = np.full((len(cols), len(L)), np.nan)
    for j, (_, vals, _) in enumerate(cols):
        v = np.array(vals, float)
        m, s = np.nanmean(v), np.nanstd(v)
        M[j] = (v - m) / (s if s > 0 else 1.0)
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r",
                   norm=TwoSlopeNorm(vmin=-2.2, vcenter=0, vmax=2.2))
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([c[0] for c in cols], fontsize=8.5)
    ax.set_xticks(range(0, len(L), 2))
    ax.set_xticklabels([f"{int(x)}" for x in L[::2]], fontsize=8)
    ax.set_xlabel("层号 (depth)", fontsize=9)
    for i, fa in enumerate(is_fa):
        if fa:
            ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, len(cols),
                                       fill=False, ec="k", lw=1.6, zorder=3))
    for j in range(len(cols) + 1):
        ax.axhline(j - 0.5, color="w", lw=1.2)
    ax.set_title("(a) 逐层奇点指标热力图（每行独立 z 分数；黑框 = 全注意力层；红=高/蓝=低）",
                 fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cb.set_label("列内 z 分数", fontsize=8)
    cb.ax.tick_params(labelsize=8)

    def profile(idx, key, title, null=None, null_label="", ylog=False,
                color_by_type=True, annotate="max"):
        ax = fig.add_subplot(gs[idx])
        v = np.array([r[key] for r in rows], float)
        if ylog:
            ax.set_yscale("log")
        if color_by_type:
            ax.plot(np.where(is_fa, L, np.nan), np.where(is_fa, v, np.nan),
                    "o-", color=C_FA, ms=4.5, lw=1.4, label="全注意力")
            ax.plot(np.where(~is_fa, L, np.nan), np.where(~is_fa, v, np.nan),
                    "o-", color=C_LA, ms=4.5, lw=1.4, label="GatedDeltaNet")
        else:
            ax.plot(L, v, "o-", color=C_LA, ms=4)
        if null is not None:
            ax.axhline(null, color=C_REF, ls="--", lw=1.2,
                       label=null_label or f"null = {null}")
        k = int(np.nanargmax(v) if annotate == "max" else np.nanargmin(v))
        ax.annotate(f"L{int(L[k])}", (L[k], v[k]), textcoords="offset points",
                    xytext=(4, 6), fontsize=8.5, color="crimson", weight="bold")
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("层号", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.6, which="both")
        ax.legend(fontsize=7, loc="best", framealpha=0.9)
        return ax

    profile((1, 0), "act_conc_downproj",
            "(b) 激活集中度（AWQ 显著性）  ↑=少数通道承载一切",
            null=0.01, null_label="null = 0.01（均匀）")
    profile((1, 1), "act_effdim_frac",
            "(c) down_proj 输入有效维占比  ↓=表示塌缩",
            null=1.0, null_label="null = 1.0（不塌缩）", color_by_type=False)
    profile((1, 2), "w_min_stable_rank_frac",
            "(d) 权重秩亏度（对数轴）  ↓=死方向多",
            null=1.0, null_label="null = 1.0（满秩）",
            color_by_type=False, ylog=True, annotate="min")

    ax = fig.add_subplot(gs[2, 0])
    v = np.array([r["dnet_timescale_cv"] for r in rows], float)
    m = ~is_fa
    ax.plot(L[m], v[m], "o-", color=C_LA, ms=4.5, lw=1.4)
    k = int(np.nanargmin(v))
    ax.annotate(f"最简并 L{int(L[k])}", (L[k], v[k]), textcoords="offset points",
                xytext=(4, -12), fontsize=8.5, color="crimson", weight="bold")
    ax.set_title("(e) DeltaNet 跨头时间尺度 CV\n↓ = 各头时间尺度趋同 = 简并", fontsize=9.5)
    ax.set_xlabel("层号", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.6)

    ax = fig.add_subplot(gs[2, 1])
    ax.bar(L, [r["score_total"] for r in rows],
           color=[C_FA if f else C_LA for f in is_fa])
    ax.set_title("(f) 复合奇点分（仅排序便利，非证据）\n红=全注意力  蓝=DeltaNet", fontsize=9.5)
    ax.set_xlabel("层号", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    ax = fig.add_subplot(gs[2, 2])
    ax.scatter([r["act_effdim_frac"] for r in rows],
               [r["act_conc_downproj"] for r in rows],
               c=[C_FA if f else C_LA for f in is_fa], s=42, zorder=3)
    for r in rows:
        if r["layer"] in (0, 2, 18, 26, 29, 30, 31):
            ax.annotate(f"L{r['layer']}", (r["act_effdim_frac"], r["act_conc_downproj"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(0.01, color=C_REF, ls="--", lw=1.1)
    ax.set_xlabel("有效维占比（→1 越健康）", fontsize=8)
    ax.set_ylabel("激活集中度", fontsize=8)
    ax.set_title("(g) 塌缩 × 集中：右上角 = 双重奇点", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.6)

    fig.suptitle("Qwen3.5-9B 奇点图谱（AWQ 激活显著性 + 谱签名/DDS + DeltaNet 状态转移谱）",
                 fontsize=13, y=0.985)
    p = f"{FIG}/fig1_singularity_map.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[S8] {p}")


# ══════════════════════════════════════════════════════════════════════
# fig2 全局坐标结构
# ══════════════════════════════════════════════════════════════════════

def fig2():
    """全局坐标结构：把 4096 维压到**跨层最持久的 K 个通道**，否则每列不足 1 像素看不见。

    ★ 第一版把 4096 列全画进 ~1800px ⇒ 竖向亮条完全不可见（自己的教训）。
    """
    npz = np.load(f"{OUT}/s1_probe.npz")
    probe = jload(f"{OUT}/s1_probe.json")
    if probe is None:
        print("[S8] fig2 跳过：缺 s1_probe.json")
        return
    role = "linear_attn.in_proj_qkv"
    pairs = []
    for name, v in probe["per_linear"].items():
        if v["role"] != role:
            continue
        key = "act__" + name.replace(".", "_")
        if key in npz.files:
            pairs.append((v["layer"], npz[key].astype(np.float64)))
    pairs.sort()
    if not pairs:
        print("[S8] fig2 跳过：找不到 in_proj_qkv 数据")
        return
    Ls = [p[0] for p in pairs]
    M = np.stack([p[1] for p in pairs])                 # (n_layers, 4096)
    ratio = M / np.median(M, axis=1, keepdims=True)     # 逐层按中位数归一

    K = 36
    peak = ratio.max(axis=0)                            # 跨层最大倍数
    top = np.argsort(-peak)[:K]                         # 最持久的离群通道
    Z = np.log10(np.clip(ratio[:, top], 1e-3, None))    # (n_layers, K)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 1.0, 1.0],
                          width_ratios=[2.6, 1.0], hspace=0.55, wspace=0.22)

    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(Z, aspect="auto", cmap="inferno", vmin=0.0,
                   vmax=float(np.percentile(Z, 99.5)))
    ax.set_yticks(range(len(Ls)))
    ax.set_yticklabels([str(x) for x in Ls], fontsize=8)
    ax.set_ylabel("层号（自上而下递增）", fontsize=10)
    ax.set_xticks(range(K))
    ax.set_xticklabels([str(c) for c in top], rotation=90, fontsize=8)
    ax.set_xlabel("隐藏维通道索引（按跨层峰值排序，左强右弱）", fontsize=10)
    ax.set_title("(a) 全局坐标结构：跨层最持久的 36 个离群通道 × 逐层显著性\n"
                 "整列持续发亮 = 同一通道在几乎每一层都是离群值（不是逐层局部的）",
                 fontsize=12)
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.022)
    cb.set_label("log10( mean|x_j| / 该层中位数 )", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    ax = fig.add_subplot(gs[1, 0])
    for c in top[:5]:
        ax.plot(Ls, np.log10(np.clip(ratio[:, c], 1e-3, None)),
                "o-", ms=4, lw=1.4, label=f"通道 {int(c)}")
    ax.axhline(np.log10(20), color="crimson", ls="--", lw=1.2, label="20× 中位数")
    ax.set_ylabel("log10(倍数)", fontsize=9)
    ax.set_xlabel("层号", fontsize=9)
    ax.set_title("(b) 前 5 个通道的逐层剖面：几乎每层都远超 20× 阈值", fontsize=10.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25, lw=0.6)

    ax = fig.add_subplot(gs[1, 1])
    srt = np.sort(M[-1])[::-1]
    tot = srt.sum()
    cum = np.cumsum(srt) / tot
    ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum * 100, color=C_LA, lw=2)
    ax.set_xlabel("通道百分比（按幅度降序）", fontsize=9)
    ax.set_ylabel("累积占比 %", fontsize=9)
    ax.set_title(f"(c) 末层能量集中\ntop-1% 通道占 {cum[int(0.01*len(cum))]*100:.1f}%",
                 fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.6)

    ax = fig.add_subplot(gs[2, 0])
    ax.bar(range(len(Ls)), np.log10(ratio.max(1)), color=C_LA)
    ax.axhline(np.log10(20), color="crimson", ls="--", lw=1.2, label="20×")
    ax.set_xticks(range(len(Ls)))
    ax.set_xticklabels([str(x) for x in Ls], fontsize=8)
    ax.set_xlabel("层号", fontsize=9)
    ax.set_ylabel("log10(峰值/中位)", fontsize=9)
    ax.set_title("(d) 每层最强通道的相对幅度", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    ax = fig.add_subplot(gs[2, 1])
    pers = np.array([(ratio[:, c] > 20).sum() for c in range(ratio.shape[1])])
    pk = peak
    n_glob = int(((pers > 12) & (pk > 20)).sum())
    ax.scatter(pk, pers, s=7, alpha=0.35, color="#8c564b")
    ax.set_xscale("log")
    ax.axvline(20, color="crimson", ls="--", lw=1.1, label="20× 阈值")
    ax.axhline(12, color="crimson", ls=":", lw=1.1, label="12 层阈值")
    ax.set_xlabel("跨层峰值倍数（log）", fontsize=9)
    ax.set_ylabel("超阈的层数", fontsize=9)
    ax.set_title(f"(e) 峰值 × 持久性\n仅 **{n_glob}** 个通道同时高且持久", fontsize=10.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.6, which="both")

    fig.suptitle("全局坐标结构证据：少数隐藏通道在几乎每一层都是极端离群值"
                 "（Qwen3.5-9B，in_proj_qkv）", fontsize=13, y=0.975)
    p = f"{FIG}/fig2_global_channels.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[S8] {p}  (top 通道 = {[int(c) for c in top[:8]]})")


def sm_flat(sm):
    return sm.get("per_linear", {})


# ══════════════════════════════════════════════════════════════════════
# fig3 自我认知定位 + 编辑
# ══════════════════════════════════════════════════════════════════════

def fig3(sc, ed, sm, post):
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    # (a) 可分性剖面 + null
    ax = fig.add_subplot(gs[0, :2])
    Ls = sorted(sc["per_layer"], key=int)
    x = [int(k) for k in Ls]
    d = [sc["per_layer"][k]["res_mean_d"] for k in Ls]
    nl = [sc["per_layer"][k]["null_mean_d"] for k in Ls]
    ax.plot(x, d, "o-", color="#2ca02c", ms=5, lw=1.8, label="自我认知 vs 中性（可分性）")
    ax.plot(x, nl, "s--", color=C_REF, ms=4, lw=1.3, label="null：中性组内部拆半")
    k = int(np.argmax(d))
    ax.axvline(x[k], color="crimson", ls=":", lw=1.6,
               label=f"实测峰 = L{x[k]}（≈{100*x[k]/31:.0f}% 深度）")
    ax.axvspan(20, 22, color="orange", alpha=0.18,
               label="tech plan 预测带 L20–22（2/3 深度）")
    ax.set_xlabel("层号", fontsize=9)
    ax.set_ylabel("可分性 mean_d", fontsize=9)
    ax.set_title("(a) 自我认知奇点定位：实测峰在 L13–14，方案预测的 L20–22 处无明显峰",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.6)

    # (b) MLP 归因神经元数
    ax = fig.add_subplot(gs[0, 2])
    dp = sc["mlp_downproj"]
    y = [dp.get(k, {}).get("dp_n_gt4", 0) for k in Ls]
    ax.bar(x, y, color=["#ff7f0e" if 12 <= t <= 17 else "#8c564b" for t in x])
    ax.set_title("(b) MLP 高归因神经元数\n(>4σ)", fontsize=10)
    ax.set_xlabel("层号", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    # (c) 编辑前后漂移
    ax = fig.add_subplot(gs[1, :2])
    before = {r["layer"]: r["act_conc_downproj"] for r in sm["ranked_layers"]}
    after = {r["layer"]: r["act_conc_downproj"] for r in post["ranked_layers"]}
    edited = set(post.get("meta", {}).get("lora_edited_layers") or [])
    Lk = sorted(before)
    dv = []
    for Lk_ in Lk:
        a = after.get(Lk_, np.nan)
        dv.append((a - before[Lk_]) / before[Lk_] * 100 if a == a and before[Lk_] else 0.0)
    ax.bar(Lk, dv, color=["crimson" if t in edited else "#bbbbbb" for t in Lk])
    ax.axhline(0, color="k", lw=1)
    me = np.mean([d_ for d_, t in zip(dv, Lk) if t in edited])
    ax.axhline(me, color="crimson", ls="--", lw=1.4,
               label=f"被编辑层均值 {me:+.2f}%")
    ax.set_xlabel("层号", fontsize=9)
    ax.set_ylabel("激活集中度变化 %", fontsize=9)
    ax.set_title("(c) LoRA 编辑（L12–17）后的奇点漂移：只有被编辑层系统性下移，其余≈0",
                 fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    # (d) 编辑判据
    ax = fig.add_subplot(gs[1, 2])
    b, e = ed["base"], ed["edited"]
    labels = ["身份命中率", "概念溢出率"]
    xs = np.arange(2)
    ax.bar(xs - 0.19, [b["identity_hit_rate"], b["overflow_rate"]], 0.36,
           label="编辑前", color="#999999")
    ax.bar(xs + 0.19, [e["identity_hit_rate"], e["overflow_rate"]], 0.36,
           label="编辑后", color="#2ca02c")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.axhline(0.8, color="crimson", ls="--", lw=1.2, label="判据 ≥0.8")
    ax.set_title(f"(d) 编辑效果\n回归 CE Δ = {e['delta_ce_pct']:+.2f}%（判据 <5%）",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    fig.suptitle("目标2：自我认知奇点的定位与编辑（LoRA L12–17，7.50 M 参数，34 s）",
                 fontsize=13, y=0.985)
    p = f"{FIG}/fig3_selfcog_edit.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[S8] {p}")


# ══════════════════════════════════════════════════════════════════════
# fig4 LLC × DDS
# ══════════════════════════════════════════════════════════════════════

def fig4(llc, sm):
    if not llc:
        print("[S8] fig4 跳过：无 LLC 结果")
        return
    score = {r["layer"]: r["score_total"] for r in sm["ranked_layers"]}
    pts = [(int(r["tag"].split("_")[0].lstrip("L")), r["llc_mean"]) for r in llc]
    pts = [(L, v) for L, v in pts if L in score]
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    xs = [score[L] for L, _ in pts]
    ys = [v for _, v in pts]
    ax.scatter(xs, ys, s=130, c=xs, cmap="RdBu_r", edgecolor="k", zorder=3)
    for (L, v), x in zip(pts, xs):
        ax.annotate(f"L{L}", (x, v), textcoords="offset points", xytext=(9, 5), fontsize=10)
    ax.axhline(np.mean(ys), color=C_REF, ls="--", lw=1.2,
               label=f"LLC 均值 {np.mean(ys):.2f}")
    ax.set_xlabel("奇点复合分（AWQ 显著性 + DDS 谱，←小  大→）", fontsize=10)
    ax.set_ylabel("localized LLC", fontsize=10)
    ax.set_title("LLC × 奇点指标 —— 二者方向相反（Spearman ρ = −0.80）\n"
                 "最强奇点 L31 的 LLC 最低 ⇒ tech plan §3.3 的交叉验证不成立", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.text(0.02, 0.03, "⚠️ localized LLC（仅目标层块采样）、单链、lr 未系统标定",
            transform=ax.transAxes, fontsize=8, color="#666666")
    p = f"{FIG}/fig4_llc_vs_dds.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"[S8] {p}")


def fig5(sm):
    """跨种子重复：效应量 vs 种子间噪声。"""
    ms = jload(f"{OUT}/multiseed.json")
    if ms is None:
        print("[S8] fig5 跳过：缺 multiseed.json")
        return
    per, a = ms["per_seed"], ms["aggregate"]
    seeds = [r["seed"] for r in per]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    fig.subplots_adjust(top=0.76, wspace=0.22)

    ax = axes[0]
    x = np.arange(len(seeds))
    ax.bar(x - 0.19, [r["drift_edited_mean_pct"] for r in per], 0.36,
           color="crimson", label="被编辑层（L12–17）")
    ax.bar(x + 0.19, [r["drift_other_mean_pct"] for r in per], 0.36,
           color="#bbbbbb", label="其余 26 层")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("激活集中度变化 %", fontsize=10)
    ax.set_title("(a) 逐种子：编辑层系统性下移，其余层≈0\n"
                 f"（{a['n_edited_all_negative']}/{a['n_seeds']} 种子「6 层全负」，"
                 f"{a['n_gap_negative']}/{a['n_seeds']} 种子 gap<0）", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    ax = axes[1]
    gaps = [r["drift_gap_pp"] for r in per]
    ax.bar(x, gaps, 0.55, color="#2ca02c")
    m, sd = a["drift_gap_pp"]["mean"], a["drift_gap_pp"]["sd"]
    ax.axhline(m, color="crimson", ls="--", lw=1.8, label=f"均值 {m:+.2f} pp")
    ax.fill_between([-0.5, len(seeds) - 0.5], m - sd, m + sd,
                    color="crimson", alpha=0.15, label=f"±1 sd ({sd:.2f})")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("配对差（编辑层 − 其余）pp", fontsize=10)
    ax.set_title(f"(b) 配对差 t = {a['drift_gap_t']:.2f}（n={a['n_seeds']}）\n"
                 "全部同号 ⇒ 效应大于种子噪声", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    # 附：编辑三项判据的跨种子散布
    txt = (f"身份命中 {a['identity_hit_rate']['mean']:.2f}±"
           f"{a['identity_hit_rate']['sd']:.2f}   "
           f"ΔCE {a['delta_ce_pct']['mean']:+.2f}±{a['delta_ce_pct']['sd']:.2f}%   "
           f"溢出 {a['overflow_rate']['mean']:.2f}")
    fig.suptitle("跨种子重复（5 个独立种子）：编辑效果与奇点漂移都稳健\n" + txt,
                 fontsize=12.5, y=0.97)
    p = f"{FIG}/fig5_multiseed.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[S8] {p}")


def fig6():
    """目标3 前提检验：习得 vs 遗忘（含噪声底与阳性对照）。"""
    g = jload(f"{OUT}/g3_compare.json")
    if g is None:
        print("[S8] fig6 跳过：缺 g3_compare.json")
        return
    arms = g["arms"]
    nf = g.get("instrument_noise_floor") or {}
    stages = ["base", "after_d1", "after_d2", "destructive_ctrl"]
    labels = ["基线", "注入域1", "续注域2", "破坏性对照"]
    colors = {"band12_17": "#1f77b4", "band26_31": "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    fig.subplots_adjust(top=0.78, wspace=0.28)

    x = np.arange(len(stages))
    w = 0.36

    def series(a, key):
        """取各阶段的某个量；None/缺失 → nan（matplotlib 能画 nan 但画不了 None）。"""
        return [np.nan if a["rows"].get(s, {}).get(key) is None
                else a["rows"].get(s, {}).get(key, np.nan) for s in stages]

    ax = axes[0]
    for i, (tag, a) in enumerate(arms.items()):
        ax.bar(x + (i - 0.5) * w, series(a, "acquire_acc"), w,
               label=f"{tag} {a['layers']}", color=colors.get(tag, "gray"))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("留出改写问法的命中率", fontsize=10)
    ax.set_title("(a) 习得：0.04 → 0.88\n（新知识确实学进去了）", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    ax = axes[1]
    for i, (tag, a) in enumerate(arms.items()):
        ax.bar(x + (i - 0.5) * w, series(a, "verifiable"), w, label=tag,
               color=colors.get(tag, "gray"))
    if nf:
        ax.axhspan(nf["mean"] - nf["sd"], nf["mean"] + nf["sd"], color="gray",
                   alpha=0.25, label=f"基座噪声底 ±1sd ({nf['sd']:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("可验证能力（math/format/code）", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) 遗忘：真实注入后完全没动；\n破坏性对照直接归零 ⇒ 仪器是灵的",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    ax = axes[2]
    for i, (tag, a) in enumerate(arms.items()):
        ax.bar(x + (i - 0.5) * w, series(a, "enrich_principal"), w, label=tag,
               color=colors.get(tag, "gray"))
    ax.axhline(15.0 / 1.39, color="crimson", ls="--", lw=1.6,
               label="方案阈值 15% ≈ ×10.8 富集")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("主奇异子空间富集（×null）", fontsize=10)
    ax.set_title("(c) §5.2 指标：真实注入 ×1.2–1.9，\n破坏性对照也只 ×2.1，够不到阈值",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    fig.suptitle("目标3 前提检验：注入新知识（48 条 QA ×2 轮）⇒ 习得成功，"
                 "而可验证能力遗忘 = 0.000（阳性对照证仪器有效）", fontsize=12.5, y=0.97)
    p = f"{FIG}/fig6_goal3_premise.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[S8] {p}")


def fig7():
    """2×2 因子分离：深度 vs 奇点。"""
    d = jload(f"{OUT}/g3_depth_factor.json")
    if d is None:
        print("[S8] fig7 跳过：缺 g3_depth_factor.json")
        return
    per = sorted(d["per_arm"], key=lambda r: r["mean_depth"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    fig.subplots_adjust(top=0.80, wspace=0.24)

    ax = axes[0]
    for pair, (hi, lo) in (("early", ("early_high", "early_low")),
                           ("late", ("late_high", "late_low"))):
        R = {r["arm"]: r for r in per}
        ax.annotate("", xy=(R[hi]["mean_depth"], R[hi]["acquire_acc"]),
                    xytext=(R[lo]["mean_depth"], R[lo]["acquire_acc"]),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#888888"))
    for r in per:
        ax.scatter(r["mean_depth"], r["acquire_acc"], s=170,
                   c=[r["mean_score"]], cmap="RdBu_r", vmin=-2, vmax=3,
                   edgecolor="k", zorder=4)
        ax.annotate(f"{r['arm']}\nscore {r['mean_score']:+.2f}",
                    (r["mean_depth"], r["acquire_acc"]),
                    textcoords="offset points", xytext=(10, -6), fontsize=8.5)
    ax.set_xlabel("平均深度", fontsize=10)
    ax.set_ylabel("习得（留出改写问法命中率）", fontsize=10)
    ax.set_title("(a) 习得几乎只随深度变化\n"
                 "同深度内高/低奇点（红/蓝）差异很小、且早期对符号为反", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6)

    ax = axes[1]
    pairs = list(d["pairs"].items())
    x = np.arange(len(pairs))
    hi = [v["high"] for _, v in pairs]
    lo = [v["low"] for _, v in pairs]
    ax.bar(x - 0.19, hi, 0.36, color="#d62728", label="高奇点臂")
    ax.bar(x + 0.19, lo, 0.36, color="#1f77b4", label="低奇点臂")
    for i, (pn, v) in enumerate(pairs):
        ax.annotate(f"高−低 = {v['high_minus_low']:+.3f}",
                    (i, max(hi[i], lo[i]) + 0.045), ha="center", fontsize=10,
                    color="crimson", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pn}\n(深度差 {v['high_depth']-v['low_depth']:+.1f} 层, "
                        f"奇点分差 {v['high_score']-v['low_score']:+.2f})"
                        for pn, v in pairs], fontsize=9)
    ax.set_ylabel("习得", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("(b) 两个 pair 都 high>low ? "
                 f"{d['both_pairs_high_gt_low']}\n⇒ 奇点不是驱动因素", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    fig.suptitle("2×2 因子分离（4 臂容量严格配平：各 18 模块 / 3.74 M，同一进程共享基线）\n"
                 "驱动注入效果的是深度，不是奇点强度", fontsize=12.5, y=0.97)
    p = f"{FIG}/fig7_depth_vs_singularity.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[S8] {p}")


def fig8():
    """热更新：域间干扰 + 剂量-反应安全包线。"""
    d = jload(f"{OUT}/g3_multidomain.json")
    if d is None:
        print("[S8] fig8 跳过：缺 g3_multidomain.json")
        return
    doms = d["config"]["seq_domains"]
    snaps = d["protocol_S"]
    tags = [s["tag"] for s in snaps]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    fig.subplots_adjust(top=0.78, wspace=0.30)

    # (a) 干扰矩阵
    ax = axes[0]
    M = np.array([[s["acquire"][dm] for dm in doms] for s in snaps], float)
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(doms)))
    ax.set_xticklabels(doms, fontsize=9)
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels([t.replace("S_after_", "注入后: ") for t in tags], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="w" if M[i, j] < 0.5 else "k", fontsize=10, weight="bold")
    ax.set_title("(a) 域间干扰矩阵（习得率）\n对角线=刚注入，之后的任务把它压下去",
                 fontsize=11)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.tick_params(labelsize=8)

    # (b) 关键对照：注入知识 vs 预训练能力
    ax = axes[1]
    def traj(dm):
        return [s["acquire"][dm] for s in snaps]
    for dm in doms:
        ax.plot(range(len(tags)), traj(dm), "o-", ms=6, lw=1.8, label=dm)
    ver = [s.get("verifiable") for s in snaps]
    ax.plot(range(len(tags)), ver, "s--", color="k", ms=7, lw=2,
            label="预训练能力(math/fmt/code)")
    nf = d["baseline"]["acquire"]
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels(["base", "注入A", "注入B", "注入C"], fontsize=9)
    ax.set_ylabel("习得率 / 可验证能力", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) 核心对照：注入的域互相覆盖（掉到 0.20/0.40），\n"
                 "而预训练能力逐位不变（0.771→0.771）", fontsize=11)
    ax.legend(fontsize=8.5, loc="center left")
    ax.grid(alpha=0.25, lw=0.6)

    # (c) 剂量-反应
    ax = axes[2]
    pts = [(d["protocol_S"][1]["verifiable"], 1e-4, 0.0)]
    base_ce = d["baseline"]["lm_ce"]
    pts += [(r["verifiable"], r["lr"],
             (r["lm_ce"] - base_ce) / base_ce * 100)
            for r in d["protocol_D"] + [d["destructive_control"]]]
    pts.sort(key=lambda t: t[1])
    lrs = [p[1] for p in pts]
    vers = [p[0] for p in pts]
    ax.plot(lrs, vers, "o-", color="#2ca02c", ms=8, lw=2, label="可验证能力")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.axvline(3e-4, color="orange", ls="--", lw=1.6, label="安全上界 ≈3e-4")
    ax.axvspan(3e-4, 1e-3, color="orange", alpha=0.15)
    ax.set_xlabel("学习率 lr（对数轴）", fontsize=10)
    ax.set_ylabel("可验证能力", fontsize=10)
    ax.set_title("(c) 剂量-反应：安全包线\nlr≤3e-4 安全；1e-3 处崩塌（能力归零）",
                 fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.25, lw=0.6, which="both")

    fig.suptitle("热更新（顺序多域注入）：真正被覆盖的是「此前注入的知识」，"
                 "不是预训练能力；且存在可操作的安全学习率区间", fontsize=12.5, y=0.97)
    p = f"{FIG}/fig8_hotupdate_interference.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[S8] {p}")


def fig9():
    """fig9：§5.2 解法检验 —— 参数隔离有效，但朴素组合失败。"""
    iso = jload(f"{OUT}/g3_isolation.json")
    if not iso:
        print("[S8] fig9 跳过：缺少 out/g3_isolation.json")
        return
    cfg = iso["config"]
    doms = cfg["domains"]
    C_ISO = "#2ca02c"
    C_ORA = "#ff7f0e"

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.2))
    x = np.arange(len(doms))
    w = 0.36

    # ── (a) 保留率：隔离 vs 容量配平的共享 ──────────────────────────
    ax = axes[0]
    rI, rS = [], []
    for i, d in enumerate(doms):
        fI = iso["arm_isolated"][i]["acquire"][d]
        lI = iso["isolated_alone"][d][d]
        fS = iso["arm_shared48"][i]["acquire"][d]
        lS = iso["arm_shared48"][-1]["acquire"][d]
        rI.append(lI / fI if fI else 0.0)
        rS.append(lS / fS if fS else 0.0)
    ax.bar(x - w / 2, rI, w, color=C_ISO, label="隔离（3×r16）")
    ax.bar(x + w / 2, rS, w, color=C_FA, label="共享（1×r48，容量相同）")
    for xi, v in zip(x - w / 2, rI):
        ax.text(xi, v + 0.03, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    for xi, v in zip(x + w / 2, rS):
        ax.text(xi, v + 0.03, f"{v:.3f}", ha="center", fontsize=10)
    ax.axhline(1.0, color=C_REF, ls="--", lw=1.2, label="保留率 = 1.0（不遗忘）")
    ax.set_xticks(x)
    ax.set_xticklabels(doms, fontsize=10)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("习得保留率", fontsize=10)
    ax.set_title("(a) 参数隔离有效，且不是容量假象\n"
                 f"参数量恰好相等：{cfg['n_params_3xr16']/1e6:.3f}M = "
                 f"{cfg['n_params_1xr48']/1e6:.3f}M", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ── (b) 朴素组合的破坏 ──────────────────────────────────────────
    ax = axes[1]
    alone = [iso["isolated_alone"][d][d] for d in doms]
    comp = [iso["composed"]["acquire"][d] for d in doms]
    ax.bar(x - w / 2, alone, w, color=C_LA, label="单独激活该子模块")
    ax.bar(x + w / 2, comp, w, color=C_ORA, label="三子模块 ΔW 直接相加")
    for xi, v in zip(x + w / 2, comp):
        ax.text(xi, v + 0.03, f"{v:.3f}", ha="center", fontsize=10)
    for xi, a_, c_ in zip(x, alone, comp):
        ax.annotate(f"{c_-a_:+.3f}", (xi + w / 2, max(a_, c_) + 0.13),
                    ha="center", fontsize=9.5, color="#b03000", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(doms, fontsize=10)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("该域习得得分", fontsize=10)
    ax.set_title("(b) 朴素组合失败：三个域全被拖坏\n"
                 "⇒ 门控是承重件，不是可选项", fontsize=10.5)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ── (c) 组合下预训练能力未受损 ─────────────────────────────────
    ax = axes[2]
    bv, cv = iso["baseline"]["verifiable"], iso["composed"]["verifiable"]
    ax.bar([0, 1], [bv, cv], 0.5, color=[C_REF, C_ORA])
    for xi, v in zip([0, 1], [bv, cv]):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    dce = (iso["composed"]["lm_ce"] - iso["baseline"]["lm_ce"]) \
        / iso["baseline"]["lm_ce"] * 100
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["原始模型", "组合后"], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("可验证能力通过率", fontsize=10)
    ax.set_title("(c) 组合不伤预训练能力\n"
                 f"Δ = {cv-bv:+.3f}（组内噪声级别），但 LM CE {dce:+.2f}%",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("§5.2 解法检验：参数隔离【有效】（保留 1.000 vs 容量配平共享 "
                 f"{min(rS):.3f}）；朴素组合【失败】—— 必须靠门控", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    p = f"{FIG}/fig9_isolation_composition.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[S8] {p}")


def fig10():
    """fig10：§5.2 第 2 条主张（LoRA-Null 零空间初始化）的检验。

    ★ 关键口径：标准 LoRA 的 B=0 已使初始 ΔW≡0 ⇒「初始更新不干扰已有知识」
      **本来就成立**；该主张只能指**训练轨迹**。故本图看的是「干扰随训练步数的增长」。
    """
    d = jload(f"{OUT}/g3_nullinit.json")
    if not d:
        print("[S8] fig10 跳过：缺少 out/g3_nullinit.json")
        return
    C_STD, C_NULL, C_CTL, C_REP = "#d62728", "#2ca02c", "#7f7f7f", "#9467bd"
    KB = int(d["config"]["k_basis"])
    conds = ["std_chunk", "null_chunk", "std_chunk_rep", "std_single"]
    lbl = {"std_chunk": "std（baseline）", "null_chunk": "null（零空间）",
           "std_chunk_rep": "std 重复（噪声量尺）",
           "std_single": "std（单次 200 步，分段对照）"}
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0))

    # ── (a) 干扰曲线：A 的习得 vs domB 训练步数 ──────────────────────
    ax = axes[0]
    for c, col in (("std_chunk", C_STD), ("null_chunk", C_NULL),
                   ("std_chunk_rep", C_REP)):
        cv = d["conditions"][c]["curve"]
        xs = [p["step"] for p in cv]
        ys = [p["A"] for p in cv]
        ax.plot(xs, ys, "-o", color=col, lw=2, ms=6, label=lbl[c],
                alpha=0.9 if c != "std_chunk_rep" else 0.65)
    for x, y in zip([p["step"] for p in d["conditions"]["std_chunk"]["curve"]],
                    [p["A"] for p in d["conditions"]["std_chunk"]["curve"]]):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(4, 7), fontsize=8.5, color=C_STD)
    cs = d["conditions"]["std_single"]["curve"][-1]
    ax.plot([cs["step"]], [cs["A"]], "*", color=C_CTL, ms=17, zorder=5,
            label=lbl["std_single"])
    ax.set_xlabel("domB 的训练步数", fontsize=10)
    ax.set_ylabel("domA 的习得（A、B 同时激活）", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_title("(a) 新子模块对已有知识的干扰曲线\n"
                 "干扰**自始即存在**（25 步就到 0.53–0.80）", fontsize=10.5)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    # ── (b) 保留率 vs 新域习得，含同处置重复（噪声量尺）──────────────
    ax = axes[1]
    x = np.arange(len(conds))
    w = 0.36
    ret = [d["conditions"][c]["retention_A"] for c in conds]
    bacc = [d["conditions"][c]["B_alone"] for c in conds]
    cols = [C_STD, C_NULL, C_REP, C_CTL]
    ax.bar(x - w / 2, ret, w, color=cols, label="domA 保留率 (A|B)")
    ax.bar(x + w / 2, bacc, w, color=cols, alpha=0.45, hatch="//",
           label="domB 自身习得")
    for xi, v in zip(x - w / 2, ret):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=9.5,
                fontweight="bold")
    for xi, v in zip(x + w / 2, bacc):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl[c].replace("（", "\n（") for c in conds], fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("得分", fontsize=10)
    _rep = d["conditions"].get("std_chunk_rep")
    _b = d["conditions"]["std_chunk"]
    _same = (abs(_rep["retention_A"] - _b["retention_A"]) < 1e-9 if _rep
             else False)
    ax.set_title("(b) 保留率 vs 新域习得\n"
                 + ("同处置重复完全复现（0.23 = 0.23）⇒ init 已受控"
                    if _same else "★ 两个 std 条件本身就差 ⇒ 无分辨力"),
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # ── (c) 代价-收益：以「探针条数」为单位（离散、可解释）──────────
    ax = axes[2]
    NP = 15
    cs = d["conditions"]
    if "std_chunk_rep" in cs:                      # 新口径：受控 init
        d_ret = (cs["null_chunk"]["A_with_B"]
                 - cs["std_chunk"]["A_with_B"]) * NP
        d_acq = (cs["null_chunk"]["B_alone"]
                 - cs["std_chunk"]["B_alone"]) * NP
    else:                                          # 旧口径（未控 init）
        d_ret = (cs["null_chunk"]["A_with_B"]
                 - cs["std_chunk"]["A_with_B"]) * NP
        d_acq = (cs["null_chunk"]["B_alone"]
                 - cs["std_chunk"]["B_alone"]) * NP
    ax.bar([0, 1], [d_ret, d_acq], 0.5, color=["#2ca02c", "#d62728"])
    for xi, v in zip([0, 1], [d_ret, d_acq]):
        ax.text(xi, v + (0.22 if v >= 0 else -0.62), f"{v:+.0f} 条",
                ha="center", fontsize=13, fontweight="bold")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["旧知识保留\n(A、B 同时激活)", "新域自身习得\n(B 单独激活)"],
                       fontsize=9.5)
    ax.set_ylabel(f"相对 std 的变化（探针条数 / {NP}）", fontsize=10)
    lo = min(min(d_ret, d_acq), 0) - 2.0
    hi = max(max(d_ret, d_acq), 0) + 2.2
    ax.set_ylim(lo, hi)
    ax.set_title("(c) 代价-收益：以探针条数计\n"
                 f"用 {d_acq:+.0f} 条习得换 {d_ret:+.0f} 条保留 ⇒ 亏", fontsize=10.5)
    ax.text(0.02, 0.97, "初始化的结构性性质确实成立：\n"
                        "A 主方向能量 0.165 → 0.00000",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            style="italic", color=C_REF,
            bbox=dict(fc="white", ec=C_REF, alpha=0.75, lw=0.6))
    ax.grid(axis="y", alpha=0.3)

    noise = d.get("same_treatment_noise", float("nan"))
    eff = d.get("null_net_effect", float("nan"))
    controlled = "std_chunk_rep" in d.get("conditions", {})
    fig.suptitle("§5.2 第 2 条（LoRA-Null 零空间初始化）：性质成立，但收益微、代价大"
                 f"（保留 {eff:+.3f}，同处置重复噪声 {noise:+.3f}"
                 f"{'，init 已受控' if controlled else '，⚠️ init 未受控'}）",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p = f"{FIG}/fig10_nullinit.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[S8] {p}")


def fig11():
    """fig11：§5.2 第 3 条（门控路由）——朴素组合失败后，路由能否救回来。"""
    d = jload(f"{OUT}/g3_router.json")
    if not d:
        print("[S8] fig11 跳过：缺少 out/g3_router.json")
        return
    doms = d["config"]["domains"]
    dh, sm = d["domain_holdout"], d["summary"]
    C = {"base": "#7f7f7f", "random": "#8c564b", "sum": "#d62728",
         "router": "#2ca02c", "oracle": "#1f77b4"}
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0))

    # ── (a) 逐域习得：五臂对照 ──────────────────────────────────────
    ax = axes[0]
    arms = ["base", "random", "sum", "router", "oracle"]
    x = np.arange(len(doms))
    w = 0.16
    for i, a in enumerate(arms):
        ys = [dh[a][dd] for dd in doms]
        ax.bar(x + (i - 2) * w, ys, w, color=C[a], label=a)
    ax.set_xticks(x)
    ax.set_xticklabels(doms, fontsize=10)
    ax.set_ylabel("留出改写问法的习得率", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("(a) 路由把「相加」的崩塌救回来了\n"
                 f"均值 {sm['mean_sum']:.2f} → {sm['mean_router']:.2f}"
                 f"（oracle {sm['mean_oracle']:.2f}）", fontsize=10.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    # ── (b) 门控混淆矩阵 ────────────────────────────────────────────
    ax = axes[1]
    conf = np.array(d["gate"]["confusion"], dtype=float)
    cls = d["gate"]["classes"]
    row = conf / np.clip(conf.sum(1, keepdims=True), 1, None)
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cls)))
    ax.set_xticklabels(cls, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(cls)))
    ax.set_yticklabels(cls, fontsize=9)
    for i in range(len(cls)):
        for j in range(len(cls)):
            ax.text(j, i, f"{int(conf[i,j])}", ha="center", va="center",
                    fontsize=10,
                    color="white" if row[i, j] > 0.55 else "black")
    ax.set_xlabel("预测", fontsize=10)
    ax.set_ylabel("真值", fontsize=10)
    ax.set_title(f"(b) 门控留出混淆（训练准确率 "
                 f"{d['gate']['train_acc']:.3f}）\n"
                 f"none 误路由率 = {d['gate']['none_misroute_rate']:.3f}",
                 fontsize=10.5)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # ── (c) 无关查询：能力是否守住 ──────────────────────────────────
    ax = axes[2]
    tags = ["base", "router", "sum"]
    cf = d["capability"]
    vs = [cf[t]["verifiable"]["acc"] for t in tags]
    ces = [cf[t]["lm"]["ce"] for t in tags]
    ax.bar([0, 1, 2], vs, 0.5, color=[C["base"], C["router"], C["sum"]])
    for xi, v in zip([0, 1, 2], vs):
        ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=10.5,
                fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["base", "router", "sum"], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("可验证能力通过率", fontsize=10)
    dce = [(c_ - ces[0]) / ces[0] * 100 for c_ in ces]
    ax.set_title("(c) 无关查询未被劫持\n"
                 f"ΔCE：router {dce[1]:+.2f}% / sum {dce[2]:+.2f}%", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("§5.2 第 3 条（门控路由）：router 追平 oracle 的 "
                 f"{sm['router_over_oracle']:.1%}，而朴素相加只有 "
                 f"{sm['mean_sum']/sm['mean_oracle']:.1%}、随机路由 "
                 f"{sm['mean_random']/sm['mean_oracle']:.1%}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    p = f"{FIG}/fig11_router.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[S8] {p}")


def main():
    os.makedirs(FIG, exist_ok=True)
    sm, post, sc, ed, llc = load_all()
    if sm is None:
        raise SystemExit("缺少 out/singularity_map.json，先跑 s2_analyze.py")
    fig1(sm)
    fig2()
    if sc and ed and post:
        fig3(sc, ed, sm, post)
    else:
        print("[S8] fig3 跳过：缺少 selfcog/edit/post 数据")
    fig4(llc, sm)
    fig5(sm)
    fig6()
    fig7()
    fig8()
    fig9()
    fig10()
    fig11()
    print(f"[S8] 全部图表写出到 {os.path.abspath(FIG)}")


if __name__ == "__main__":
    main()

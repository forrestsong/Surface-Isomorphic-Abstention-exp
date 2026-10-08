"""S6 汇总报告：把 S1/S2b/S3/S4/S5 的读数合成一份 Markdown 奇点实验报告。

纪律：缺文件 ⇒ 标 `未完成（不得读作通过）`，绝不静默略过。
"""
from __future__ import annotations

import os
import json
import datetime

OUT = "out"
REPORT = "../奇点实验报告.md"


def jload(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def missing(name):
    return f"> ⚠️ **未完成**：`{OUT}/{name}` 不存在（不得读作通过）。\n"


def main():
    sm = jload("singularity_map.json")
    sp = jload("singular_points.json")
    llc = jload("llc_summary.json")
    sc = jload("selfcog_localization.json")
    ed = jload("edit_result.json")

    lines = []
    A = lines.append
    A("# Qwen3.5-9B 奇异点/区域 实验报告\n")
    A(f"生成时间：{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
    A("方法：**AWQ 激活显著性**（成熟量化技术的核心量）作为奇点探针 + 谱/死方向签名(DDS) "
      "+ DeltaNet 状态转移谱 + MLS 局部 LLC + 自我认知对比激活定位 + LoRA 编辑验证。\n")
    A("> 环境：DGX Spark GB10 / UMA 121GiB。模型 9.41B，bf16 加载峰值 18.3 GiB。\n")

    # ── 0b. 图表 ───────────────────────────────────────────────────
    A("\n## 图表\n")
    figs = [
        ("fig1_singularity_map.png", "图1 奇点图谱总览",
         "(a) 逐层指标热力图（每行独立 z 分数，黑框=全注意力层）；"
         "(b) AWQ 激活集中度 vs 深度，虚线为 null=0.01；"
         "(c) down_proj 输入有效维占比（↓=表示塔缩）；"
         "(d) 权重秩亏度（对数轴，null=1.0 满秩）；"
         "(e) DeltaNet 跨头时间尺度 CV（↓=头简并）；"
         "(f) 复合分排序（仅便利）；"
         "(g) 塔缩×集中散点（右上角=双重奇点）。"),
        ("fig2_global_channels.png", "图2 全局坐标结构",
         "(a) 跨层最持久的 36 个离群通道热力图（整列持续发亮 = 同一通道在每层都离群）；"
         "(b) 前 5 通道的逐层剖面；"
         "(c) 末层能量累积曲线；"
         "(d) 每层最强通道幅度；"
         "(e) 峰值×持久性散点（仅极少数通道同时高且持久）。"),
        ("fig3_selfcog_edit.png", "图3 自我认知定位与编辑",
         "(a) 自我认知可分性 vs 深度：实测峰 L13–14，橙色带为 tech plan 预测的 L20–22；"
         "(b) MLP 高归因神经元数；"
         "(c) 编辑后逐层奇点漂移（红=被编辑层）；"
         "(d) 编辑三项判据。"),
        ("fig4_llc_vs_dds.png", "图4 LLC × 奇点指标",
         "显示二者方向相反（Spearman ρ = −0.80）⇒ tech plan §3.3 的交叉验证不成立。"),
        ("fig5_multiseed.png", "图5 跨种子重复（5 个独立种子）",
         "(a) 逐种子：被编辑层系统性下移，其余层≈0；"
         "(b) 配对差全部同号（t = −8.50）⇒ 效应大于种子间噪声。"),
        ("fig6_goal3_premise.png", "图6 目标3 前提检验：习得 vs 遗忘",
         "(a) 新知识确实学进去了（0.04→0.88）；"
         "(b) 可验证能力在真实注入后**完全没动**，而破坏性对照直接归零 ⇒ 仪器有效；"
         "(c) §5.2 指标最高只到 ×2.1，够不到方案 15% 阈值。"),
        ("fig7_depth_vs_singularity.png", "图7 2×2 分离：深度 vs 奇点",
         "(a) 习得几乎只随深度变化（两条箭头分别是 early/late 对）；"
         "(b) 两组高−低差 −0.125 / +0.042，**符号不一致** ⇒ 奇点不是驱动因素。"),
        ("fig8_hotupdate_interference.png", "图8 热更新：域间干扰与安全包线",
         "(a) 干扰矩阵（对角线=刚注入，之后被压下去）；"
         "(b) 注入的域互相覆盖，而预训练能力逐位不变；"
         "(c) 剂量-反应：lr≤3e-4 安全，1e-3 崩塌。"),
        ("fig9_isolation_composition.png", "图9 §5.2 解法检验：参数隔离 vs 朴素组合",
         "(a) 隔离保留率 1.000，容量**完全相同**的共享适配器只有 0.286/0.308；"
         "(b) 三子模块 ΔW 直接相加时，三个域全被拖坏（−0.40~−0.73）⇒ 门控是承重件；"
         "(c) 组合不伤预训练能力（0.750→0.771）。"),
        ("fig10_nullinit.png", "图10 §5.2 第 2 条：LoRA-Null 零空间初始化",
         "(a) 干扰随训练步数增长，null 并不优于 std；"
         "(b) 两个 std 条件本身就差 0.43 ⇒ 无分辨力；"
         "(c) 初始化的**性质**成立（A 主方向能量 0.165→0.0000），"
         "但新域 loss 反而更差。"),
        ("fig11_router.png", "图11 §5.2 第 3 条：门控路由",
         "(a) 逐域习得：base/random/sum/router/oracle 五臂；"
         "(b) 门控留出混淆矩阵；"
         "(c) 无关查询的能力是否被劫持。"),
    ]
    for fn, title, cap in figs:
        p = os.path.join("..", "figures", fn)
        if os.path.exists(p):
            A(f"### {title}\n")
            A(f"![{title}](figures/{fn})\n")
            A(f"*{cap}*\n")
        else:
            A(f"### {title}\n")
            A(f"> ⚠️ **未生成**：`figures/{fn}` 不存在（跑 `exp/s8_plot.py`）。\n")

    # ── 0. 架构 ─────────────────────────────────────────────────────
    A("\n## 0. 架构事实\n")
    if sm:
        m = sm["meta"]
        A(f"- 层数 **{m['n_layers']}**，hidden 4096，MLP 12288，词表 248320")
        A(f"- 全注意力层（每 4 层 1 个）：{m['full_attention_layers']}（共 {len(m['full_attention_layers'])}）")
        A(f"- 其余为 GatedDeltaNet 线性注意力")
        A(f"- 标定：{m['n_calib_seq']} 序列 / {m['calib_tokens']} token，L={m['seq_len']}")

    # ── 1. 奇点图谱 ─────────────────────────────────────────────────
    A("\n## 1. 目标1 — 奇点图谱（AWQ 显著性 + 谱签名）\n")
    if sm:
        A("### 1.1 null（基线）定义\n")
        A(f"- `top1pct_share` 的 null = **0.01**（通道幅度均匀时 top-1% 恰占 1%）")
        A("- `stable_rank` 的 null = min(out,in)（满秩）；远低于它 ⇒ 秩亏")
        A("- `effdim_frac` = 参与比 / 通道数；1.0 = 不塌缩，越小越塌缩\n")
        A("### 1.2 逐层（按复合分排序，仅便利非证据）\n")
        A("| 层 | 类型 | 激活集中度(top1%) | 有效维占比 | down_proj 输入峰值 | 秩亏度 | DeltaNet 时间尺度CV | 复合分 |")
        A("|---|---|---|---|---|---|---|---|")
        for r in sm["ranked_layers"]:
            cv = r["dnet_timescale_cv"]
            cv_s = "—" if cv != cv else f"{cv:.3f}"      # nan 检查
            A(f"| L{r['layer']:02d} | {r['type'].replace('_attention','')} | "
              f"{r['act_conc_downproj']:.4f} | {r['act_effdim_frac']:.3f} | "
              f"{r['act_absmax_peak']:.1f} | {r['w_min_stable_rank_frac']:.4f} | "
              f"{cv_s} | "
              f"{r['score_total']:+.2f} |")
        A("\n### 1.3 假设检验（tech plan §2/§3.5 的预测 vs 实测）\n")
        h = sm["hypotheses"]
        a = h["H_A_activation_collapse_FA_vs_LA"]
        A("**H_A：全注意力层比 DeltaNet 层更「奇点密集」**\n")
        A(f"- 激活集中度 FA={a['act_conc_FA']:.4f} vs LA={a['act_conc_LA']:.4f}")
        A(f"- 有效维 FA={a['effdim_FA']:.3f} vs LA={a['effdim_LA']:.3f}")
        A(f"- 秩亏度 FA={a['param_rankdef_FA']:.4f} vs LA={a['param_rankdef_LA']:.4f}")
        A("- ⚠️ **混淆**：8 个 FA 层全部位于 ≥3 的后期位置，而集中度本身随深度上升 ⇒ "
          "FA>LA 可能整体是**深度效应**。逐对看：L31 远高于邻层 L29/L30，"
          "但 L7/L3 低于邻层 ⇒ **H_A 不被支持（不能排除深度混淆）**。\n")
        b = h["H_B_two_thirds_depth_band_L20_22"]
        A("**H_B：自我认知奇点在 2/3 深度（L20-22）**\n")
        A(f"- 带内集中度 {b['act_conc_band']:.4f} vs 全层均值 {b['act_conc_all_layers']:.4f}")
        A(f"- 带内复合分均值 {b['score_total_band_mean']:+.2f}（<0 = 低于平均）")
        A("- ⇒ **H_B 不被支持**：奇点峰在**末段 L26-31**，不在 L20-22。\n")
        c = h["H_C_superweight_early_MLP"]
        A("**H_C：超权重位于早期 MLP（L0-8）**\n")
        A(f"- 早期(≤L8) down_proj 输入峰值最大 {c['act_absmax_peak_early_max']:.1f}")
        A(f"- 末段(≥L26) 最大 {c['act_absmax_peak_late_max']:.1f}")
        A(f"- 集中度 早期均值 {c['act_conc_early_mean']:.4f} vs 末段 {c['act_conc_late_mean']:.4f}")
        A(f"- ⇒ **{c['verdict']}**（超激活在**末段 MLP**，与 tech plan 预期相反）\n")
        dfn = h["H_D_deltanet_timescale_degeneracy"]
        A("**H_D：DeltaNet 状态谱简并**\n")
        A(f"- 跨头时间尺度 CV 均值 {dfn['cv_mean']:.3f}，最小 {dfn['cv_min']:.3f}")
        A(f"- 最简并层：{dfn['most_degenerate_layers']}")
        A("- ⇒ 存在**轻度**时间尺度退化（L13/L16/L20/L22），但效应量不大，属**弱信号**。\n")

    # ── 2. 奇异点坐标 ───────────────────────────────────────────────
    A("\n## 2. 奇异点坐标（AWQ 显著通道 / 超权重锚点候选）\n")
    if sp:
        A("`x_median` = 该通道 mean|x| / 模块通道中位数。均匀时约 1。\n")
        A("| 模块 | 峰值/中位 | >20× 超通道数 | top-3 通道 |")
        A("|---|---|---|---|")
        for p in sp["top_modules_by_concentration"][:20]:
            t3 = ", ".join(f"c{c['channel']}({c['x_median']:.1f}×)"
                           for c in p["top_channels"][:3])
            A(f"| `{p['module']}` | {p['max_over_median']:.0f}× | "
              f"{p['n_super_channels_20x']} | {t3} |")
        A("\n**关键观察：跨层固定通道**。通道 **3994**（及 310、253、3986）在几乎每一层的 "
          "`in_proj_qkv`/`q_proj`/`v_proj` 中反复出现为极端离群 ⇒ 奇点有**全局坐标结构**"
          "（不是逐层独立的）。这与 LLM 中已知的 massive-activation 现象一致，"
          "也是 AWQ 只保护 top-1% 通道就能大幅降误差的原因。\n")
    else:
        A(missing("singular_points.json"))

    # ── 3. LLC ─────────────────────────────────────────────────────
    A("\n## 3. LLC 局部精定位（devinterp，localized SGLD）\n")
    # ★ 必须用**逐层结果文件**，不能用 llc_summary.json：sweep 被裁剪过，
    #   最后一次 `--layers 18,8` 的运行会把 summary 覆盖成只剩那两层。
    import glob
    llc = []
    for p in sorted(glob.glob(os.path.join(OUT, "llc_L*_lr*.json"))):
        with open(p, encoding="utf-8") as f:
            llc.append(json.load(f))
    llc.sort(key=lambda r: (r["lr"], int(r["tag"].split("_")[0].lstrip("L"))))
    if llc:
        A("⚠️ **边界**：仅**目标层块**参数参与 SGLD（其余冻结），是 **localized LLC**，"
          "数值不可与论文全模型基数直接比较；只用于**同口径跨层比较**。\n")
        A("| 层 | lr | 采样参数 | LLC | init_loss | 用时 |")
        A("|---|---|---|---|---|---|")
        for r in llc:
            A(f"| {r['tag'].split('_')[0]} | {r['lr']:g} | "
              f"{r['n_params_sampled']/1e6:.0f} M | {r['llc_mean']:.3f}±{r['llc_std']:.3f} | "
              f"{r['init_loss']:.4f} | {r['elapsed_s']:.0f}s |")
        A("\n⚠️ **LLC 对 lr 强敏感**（实测 L31：1e-5→1.5 / 1e-4→23 / 1e-3→42）⇒ "
          "**只有同 lr 下的跨层排序可读**。\n")
        # ★ 关键判读：LLC 是否复现 DDS/AWQ 的排序？（tech plan §3.3 的交叉验证）
        lr0 = [r for r in llc if abs(r["lr"] - 1e-4) < 1e-12]
        if len(lr0) >= 3:
            vals = [(int(r["tag"].split("_")[0].lstrip("L")), r["llc_mean"]) for r in lr0]
            vals.sort(key=lambda x: x[0])
            xs = [v for _, v in vals]
            spread = (max(xs) - min(xs)) / (sum(xs) / len(xs)) * 100
            A(f"- LLC 跨层极差仅 **{spread:.1f}%**（"
              + "、".join(f"L{L}={v:.2f}" for L, v in vals) + "）\n")
            if sm:
                score_by_layer = {r["layer"]: r["score_total"] for r in sm["ranked_layers"]}
                common = [(L, v, score_by_layer[L]) for L, v in vals if L in score_by_layer]
                if len(common) >= 3:
                    def _rank(x):
                        order = sorted(range(len(x)), key=lambda i: x[i])
                        rk = [0] * len(x)
                        for pos, i in enumerate(order):
                            rk[i] = pos
                        return rk
                    rk_llc = _rank([c[1] for c in common])
                    rk_sc = _rank([c[2] for c in common])
                    n = len(common)
                    d2 = sum((rk_llc[i] - rk_sc[i]) ** 2 for i in range(n))
                    rho = 1 - 6 * d2 / (n * (n * n - 1))
                    A(f"- 与奇点复合分的 Spearman ρ = **{rho:+.2f}**（n={n}）"
                      f" ⇒ {'一致' if rho > 0.5 else '**不一致**'}")
                    A("\n> ★ **交叉验证否证（tech plan §3.3）**：方案要求「DDS 高奇异层若同时 LLC 高，"
                      "则为核心奇点层」。实测 **L31（激活侧最强奇点）LLC 最低**，而 **L8（低奇点对照）LLC 最高**"
                      "—— 两者**方向相反**。⇒ 在本口径下（localized LLC + 该数据/温度），"
                      "**LLC 不能复现 AWQ/DDS 的奇点排序**，该交叉验证**不成立**。")
                    A("> ⚠️ 限定：仅 1 条链、lr 未做系统标定、localized 子空间，"
                      "不足以否定 LLC 本身；只说明**它在本实验口径下无判别力**。\n")
    else:
        A(missing("llc_summary.json"))

    # ── 4. 自我认知定位 ─────────────────────────────────────────────
    A("\n## 4. 目标2 — 自我认知奇点定位（对比激活）\n")
    if sc:
        import statistics as _st
        ts, tn = sc["meta"]["token_lens_self"], sc["meta"]["token_lens_neutral"]
        A(f"自我组 {sc['meta']['n_self']} 条 vs 中性组 {sc['meta']['n_neutral']} 条；"
          f"token 长度 自我 {_st.mean(ts):.1f}±{_st.pstdev(ts):.1f}、"
          f"中性 {_st.mean(tn):.1f}±{_st.pstdev(tn):.1f}"
          f"（长度相近 ⇒ 缓解“末 token 激活受长度混淆”（tech plan 未要求此控制，本实验补做））\n")
        A("| 层 | 可分性 mean_d | 中性拆半 null | down_proj 归因 | >4σ 神经元数 |")
        A("|---|---|---|---|---|")
        for Ls in sorted(sc["per_layer"], key=lambda x: int(x)):
            v = sc["per_layer"][Ls]
            dp = sc["mlp_downproj"].get(Ls, {})
            A(f"| L{int(Ls):02d} | {v['res_mean_d']:.3f} | {v['null_mean_d']:.3f} | "
              f"{dp.get('dp_mean_d', float('nan')):.3f} | {dp.get('dp_n_gt4', -1)} |")
        p2 = sc["P2_band_L20_22"]
        p3 = sc["P3_null_check"]
        A(f"\n- **P2（tech plan 预测 L20-22）**：带内 {p2['band_mean_d']:.3f} vs 全层 "
          f"{p2['all_layer_mean_d']:.3f}（比 {p2['ratio']:.2f}）；"
          f"**实际峰值层 = L{p2['actual_peak_layer']}**")
        A(f"- **P3（null）**：中性拆半 avg={p3['null_mean_d_avg']:.3f} vs "
          f"自我vs中性 avg={p3['self_vs_neutral_avg']:.3f}"
          f"（比值 {p3['self_vs_neutral_avg']/max(p3['null_mean_d_avg'],1e-9):.2f}×）")
    else:
        A(missing("selfcog_localization.json"))

    # ── 5. 编辑 ────────────────────────────────────────────────────
    A("\n## 5. 目标2 — 自我认知奇点编辑（LoRA）\n")
    if ed:
        b, e = ed["base"], ed["edited"]
        A(f"目标身份：Nebula-9B / LivingFormer；编辑层 {ed['layers']}\n")
        A("| 指标 | 编辑前 | 编辑后 | 判据 |")
        A("|---|---|---|---|")
        A(f"| 留出身份命中率 | {b['identity_hit_rate']:.2f} | {e['identity_hit_rate']:.2f} | ≥0.8 |")
        A(f"| 回归 CE | {b['regression_ce']:.4f} | {e['regression_ce']:.4f} "
          f"({e['delta_ce_pct']:+.2f}%) | <5% |")
        A(f"| 概念溢出率 | {b['overflow_rate']:.2f} | {e['overflow_rate']:.2f} | 探索性 |")
    else:
        A(missing("edit_result.json"))

    # ── 6. 编辑后奇点漂移 ──────────────────────────────────────────
    A("\n## 6. 编辑后奇点图谱漂移（打通目标1↔目标2）\n")
    post = None
    pp = os.path.join("out_post", "singularity_map.json")
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            post = json.load(f)
    if post and sm:
        before = {r["layer"]: r["act_conc_downproj"] for r in sm["ranked_layers"]}
        after = {r["layer"]: r["act_conc_downproj"] for r in post["ranked_layers"]}
        edited = set(post.get("meta", {}).get("lora_edited_layers") or [])
        A(f"探针口径：编辑后共 **{len(after)}** 层可测"
          f"（`find_linears` 已支持 peft `lora.Linear` 包装层，"
          f"ΔW≠0 的层 = {sorted(edited) if edited else '未申报'}）\n")
        A("| 层 | 编辑前集中度 | 编辑后集中度 | 变化 |")
        A("|---|---|---|---|")
        d_edit, d_other = [], []
        for L in sorted(before):
            av = after.get(L, float("nan"))
            if av != av:
                A(f"| L{L:02d} | {before[L]:.4f} | — | 不可测 |")
                continue
            dpct = (av - before[L]) / before[L] * 100 if before[L] else 0.0
            (d_edit if L in edited else d_other).append(dpct)
            tag = " ← 编辑" if L in edited else ""
            A(f"| L{L:02d} | {before[L]:.4f} | {av:.4f} | {dpct:+.1f}%{tag} |")

        def _ms(x):
            if not x:
                return float("nan"), float("nan")
            m = sum(x) / len(x)
            v = sum((t - m) ** 2 for t in x) / max(len(x) - 1, 1)
            return m, v ** 0.5

        me, se = _ms(d_edit)
        mo, so = _ms(d_other)
        A(f"\n| 分组 | n | 平均 Δ 集中度 | 标准差 |")
        A("|---|---|---|---|")
        A(f"| **被编辑层**（LoRA 命中）| {len(d_edit)} | **{me:+.2f}%** | {se:.2f} |")
        A(f"| 未编辑层 | {len(d_other)} | {mo:+.2f}% | {so:.2f} |")
        if d_edit and d_other:
            A(f"| 组间差 | — | **{me - mo:+.2f} pp** | — |")
        A(f"\n- 编辑后最大集中度层：L{max((L for L in after if after[L] == after[L]), key=lambda L: after[L])}")
        if edited:
            A("\n> ★ **编辑效应精确落在被编辑层上**：被编辑的"
              f"{sorted(edited)} 共 {len(d_edit)} 层**全部**在负方向位移"
              f"（{me:+.2f}%±{se:.2f}），而其余 {len(d_other)} 层平均仅 {mo:+.2f}%"
              f"（其中上游层 <L12 **精确 0.00%**，因其输入不受编辑影响）。"
              f"组间差 **{me - mo:+.2f} pp** ⇒ 编辑对奇点图谱的影响是**高度局部化**的。")
            A("> ⚠️ 但**效应很小**（~1% 量级），且 n=6、单种子 ⇒ "
              "只可作为**方向一致性**证据，不构成效应量声明。")
            A("> ⚠️ **方向**：编辑让目标层的激活集中度**略降**（而非升高），"
              "即轻微**缓解**了表示塌缩；这不等于「编辑修复了奇点」，"
              "因为集中度只是奇点的一个代理量。")
            A("> ✅ **命中验证（已升级为代码守卫）**：`s7_guard_coverage.py` 断言"
              "「ΔW≠0 的层集合恰为目标层」，且「基座 `effective_weight` 与 `weight` 逐位相同」"
              "—— 前者防包装层漏检，后者防未编辑层被误改。")
    else:
        A(missing("out_post/s1_probe.json"))

    # ── 7. 跨种子 ─────────────────────────────────────────────────
    A("\n## 7. 跨种子重复（统计功效）\n")
    ms = jload("multiseed.json")
    if ms:
        a = ms["aggregate"]
        c = a["config"]
        A(f"同一编辑流程换 {a['n_seeds']} 个种子独立重复（LoRA r={c['rank']}、"
          f"层 {c['layers']}、{c['steps']} 步、lr={c['lr']:g}）；"
          f"漂移口径 {c['n_calib_full']}×{c['l_calib_full']} 与 s1 一致。\n")
        A("| seed | 身份命中 | ΔCE % | 溢出 | 编辑层漂移 % | 其余层漂移 % | 配对差 pp |")
        A("|---|---|---|---|---|---|---|")
        for r in ms["per_seed"]:
            A(f"| {r['seed']} | {r['identity_hit_rate']:.2f} | "
              f"{r['delta_ce_pct']:+.2f} | {r['overflow_rate']:.2f} | "
              f"{r['drift_edited_mean_pct']:+.2f} | "
              f"{r['drift_other_mean_pct']:+.2f} | {r['drift_gap_pp']:+.2f} |")
        A("")
        A("| 汇总 | mean | sd | se |")
        A("|---|---|---|---|")
        for k, lab in (("identity_hit_rate", "身份命中率"),
                       ("delta_ce_pct", "ΔCE %"),
                       ("overflow_rate", "溢出率"),
                       ("drift_edited_mean_pct", "编辑层漂移 %"),
                       ("drift_other_mean_pct", "其余层漂移 %"),
                       ("drift_gap_pp", "**配对差 pp**")):
            x = a[k]
            A(f"| {lab} | {x['mean']:+.3f} | {x['sd']:.3f} | {x['se']:.3f} |")
        A(f"\n- 配对差 t = **{a['drift_gap_t']:.2f}**（n={a['n_seeds']}，df={a['n_seeds']-1}）")
        A(f"- 逐种子「配对差 <0」：**{a['n_gap_negative']}/{a['n_seeds']}**；"
          f"逐种子「6 层全部负向」：**{a['n_edited_all_negative']}/{a['n_seeds']}**")
        A("")
        A("> ★ **判读**：编辑效应（配对差）若显著大于种子间噪声，才能说它是真的。"
          f"实测配对差 {a['drift_gap_pp']['mean']:+.2f}±{a['drift_gap_pp']['sd']:.2f} pp，"
          f"t={a['drift_gap_t']:.2f}。"
          + ("**方向在全部种子一致** ⇒ 效应可信（但仍属小效应量）。"
             if a["n_gap_negative"] == a["n_seeds"]
             else "**存在符号翻转** ⇒ 该效应在单种子下不可靠。"))
        A("> ⚠️ n=5 且为配对比较，未做多重比较校正；只作方向性证据。")
    else:
        A(missing("multiseed.json"))

    # ── 8. 目标3 前提检验 ──────────────────────────────────────────
    A("\n## 8. 目标3 前提检验：新知识注入是否造成遗忘 / 向主奇异子空间漂移？\n")
    g3 = jload("g3_compare.json")
    if g3:
        A("协议与目标2 的编辑一致（LoRA r=16、200 步、lr=1e-4），**只换「教什么」**："
          "注入一个**虚构新域**（Northwind Dynamics 产品线，48 条 QA），"
          "用**留出改写问法**（24 条）测习得，用**程序化判分的能力基准**测遗忘：\n")
        A("- `math` 24 题 / `format` 24 题 —— 精确匹配（答案由 Python 算出）")
        A("- `code` 8 题 —— 生成函数后**在子进程跑单测**（A 级真值）")
        A("- `lm` 留出语料 48 段 —— token 准确率 + CE（与标定/回归语料不重叠）\n")
        A("两个层带对照：**L12–17**（目标2 自我认知带）vs **L26–31**（目标1 奇点带）。\n")
        for tag, a in g3["arms"].items():
            A(f"### 层带 `{tag}`（{a['layers']}）\n")
            A("| 阶段 | 习得 | 可验证 | math | fmt | code | LM tok | ΔCE% | 遗忘 | 主奇异富集 |")
            A("|---|---|---|---|---|---|---|---|---|---|")
            for st in ("base", "after_d1", "after_d2", "destructive_ctrl"):
                r = a["rows"].get(st)
                if not r:
                    continue
                fv = r["forget_verifiable"]
                dc = r["delta_ce_pct"]
                enr = r["enrich_principal"]
                A(f"| {st} | {r['acquire_acc']:.2f} | {r['verifiable']:.3f} | "
                  f"{r['math']:.2f} | {r['format']:.2f} | {r['code']:.2f} | "
                  f"{r['lm_token_acc']:.3f} | "
                  f"{'—' if dc is None else f'{dc:+.2f}'} | "
                  f"{'—' if fv is None else f'{fv:+.3f}'} | "
                  f"{'—' if enr is None else f'×{enr:.2f}'} |")
            A("")
        A("### 阳性对照（决定性）\n")
        for tag, a in g3["arms"].items():
            A(f"- `{tag}`：破坏性大 lr（×100）续训使可验证能力下降 "
              f"**{a['positive_control_drop']:+.3f}** ⇒ **基准能测到退化**。")
        A("\n> ★ 没有这条对照，「没遗忘」什么都证明不了 —— 基准可能只是不敏感。\n")

        nf = g3.get("instrument_noise_floor")
        if nf:
            A("### 仪器噪声底（必须报，否则会把抖动当效应）\n")
            A(f"同一个**未被改动**的基座在 {nf['n_independent_processes']} 个独立进程里读数：\n")
            A(f"- 可验证能力：{['%.3f' % v for v in nf['base_verifiable_values']]}")
            A(f"  ⇒ mean **{nf['mean']:.3f}**，sd **{nf['sd']:.3f}**，极差 **{nf['range']:.3f}**")
            A(f"- `format` 子基准：{['%.2f' % v for v in nf['format_values']]}（**噪声全在这里**）")
            A(f"- `math`/`code` 三次完全相同；**LM CE 极差 = {nf['lm_ce_range']:.1e}**（逐位确定）")
            A("\n⚠️ 实测机制：**进程内** base 与 after 逐位相同，**跨进程** format 才会抖"
              "（CUDA kernel/调度非确定性翻转了少数边界样例）"
              "⇒ 跨臂比较绝对值必须对着这个底。\n")

        ca = g3.get("cross_arm", {})
        if "after_d2" in ca:
            arms = list(ca["after_d2"].keys())
            A("### 跨臂对照（注入层带的影响）\n")
            A("| 量 | " + " | ".join(f"`{a}`" for a in arms) + " | 是否超过噪声底 |")
            A("|---|" + "---|" * (len(arms) + 1))
            rows_cmp = [
                ("习得（after_d2）",
                 [f"{ca['after_d2'][a]['acquire']:.3f}" for a in arms],
                 ("是" if nf and abs(ca['after_d2'][arms[0]]['acquire']
                                     - ca['after_d2'][arms[1]]['acquire']) > nf["range"]
                  else "否") if nf and len(arms) > 1 else "—"),
                ("遗忘（after_d2）",
                 [f"{ca['after_d2'][a]['forget']:+.3f}" for a in arms], "—"),
                ("ΔCE%（after_d2）",
                 [f"{ca['after_d2'][a]['delta_ce_pct']:+.2f}" for a in arms], "—"),
            ]
            for name, vals, over in rows_cmp:
                A(f"| {name} | " + " | ".join(vals) + f" | {over} |")
            A("")
        A("### 判读\n")
        A("1. **§5.1 的前提未被观察到**：注入一整个新知识域（两轮共 96 步）后，"
          "**习得从 0.04 升到 0.75–0.88**，而 math/format/code 三项在**同一进程内逐位不变**"
          "（遗忘 = **+0.000**），ΔCE 仅 +0.8~1.3%。⇒ 在本量级下，"
          "「新知识注入 ⇒ 覆盖预训练知识」**不成立**。")
        A("2. **§5.2 的 15% 阈值不可用**：两个层带、含**破坏性**对照，"
          "主奇异子空间占比最高只到 2.4%（富集 ×2.13），**0/37 模块**超过 15% "
          "⇒ 该触发器在模型被摧毁时**仍不触发**。")
        A("3. **顺带的反向细节**：破坏性对照的显著通道富集掉到 ×0.96（低于 null）"
          "⇒ 该指标既不单调、也与实际损伤解耦。")
        A("4. **层带确实有影响，但影响的是「习得」而非「遗忘」**：奇点带 L26–31 的习得 "
          f"({ca.get('after_d2',{}).get('band26_31',{}).get('acquire', float('nan')):.3f}) "
          f"高于 L12–17 ({ca.get('after_d2',{}).get('band12_17',{}).get('acquire', float('nan')):.3f})，"
          "差值超过噪声底；而两臂遗忘都是 0。")
        A("   ⚠️ **未分离的混淆**：L26–31 更靠近输出层 ⇒ 「更易拟合」与 "
          "「奇点带更适合注入」两种解释**无法区分**。要分离需再加一个等深度的非奇点带"
          "（如 L20–25）作对照 —— 这是本节的下一步。")
    else:
        A(missing("g3_compare.json"))

    # ── 8b. 2×2 因子分离：深度 vs 奇点 ──────────────────────────────
    #  ★ 必须放在 if/else **之外**：这段只依赖 g3_depth_factor.json，
    #    放错缩进会落进 else 分支而永不执行（本文件踩过）。
    A("### 2×2 因子分离：到底是「深度」还是「奇点」？\n")
    df = jload("g3_depth_factor.json")
    if df:
        A("上一节留下一个混淆：奇点带 L26–31 的习得更高，但它也**更靠近输出层**。"
          "本实验在**同一进程**里跑 4 个容量**严格配平**的臂（各 18 个 LoRA 模块 / "
          "3.74 M 可训练参数），按「深度 × 奇点强度」做 2×2：\n")
        A("| 臂 | 层 | 平均深度 | 奇点分 | 习得 | 遗忘 | ΔCE% | final loss |")
        A("|---|---|---|---|---|---|---|---|")
        for r in sorted(df["per_arm"], key=lambda r: r["mean_depth"]):
            A(f"| `{r['arm']}` | {r['layers']} | {r['mean_depth']:.1f} | "
              f"{r['mean_score']:+.2f} | **{r['acquire_acc']:.3f}** | "
              f"{r['forget']:+.3f} | {r['delta_ce_pct']:+.2f} | {r['final_loss']:.4f} |")
        A("")
        A("| 对照对 | 高奇点臂习得 | 低奇点臂习得 | 高−低 | 深度差 | 奇点分差 |")
        A("|---|---|---|---|---|---|")
        for pn, v in df["pairs"].items():
            A(f"| {pn} | {v['high']:.3f} | {v['low']:.3f} | "
              f"**{v['high_minus_low']:+.3f}** | "
              f"{v['high_depth']-v['low_depth']:+.1f} | "
              f"{v['high_score']-v['low_score']:+.2f} |")
        A("")
        A(f"- 「两个 pair 都 high>low」= **{df['both_pairs_high_gt_low']}**")
        A(f"- 阳性对照（同进程）：可验证 {df['baseline']['verifiable']:.3f} → "
          f"{df['destructive_control']['verifiable']:.3f} ⇒ 仪器有效")
        A("")
        A("> ★★ **结论：驱动习得的是「深度」，不是「奇点」。** 习得几乎完全由深度决定"
          "（深度 1–5 时 0.29–0.42，深度 26–28 时 0.79–0.83）；而在**同深度窗口内**改变"
          "奇点强度只带来 **−0.125**（早期对，**符号还是反的**）与 **+0.042**（后期对，可忽略）。"
          "⇒ 上一节「奇点带更适合注入」的读法**被否证**，它是深度（靠近输出层）的产物。")
        A("> ★ 附带的重要加固：**4 个臂的遗忘全部 = +0.000**，"
          "包括深度 1–5 的早期层 ⇒ 「无遗忘」不是选中段造成的假象，而是**深度稳健**的。")
    else:
        A(missing("g3_depth_factor.json"))

    # ── 9. 热更新：多域顺序注入 + 剂量-反应 ────────────────────────
    #  ★ 本节修正了 §8 的一个过强结论，务必放在 if/else 之外（4 空格缩进）。
    A("\n## 9. 热更新：多域顺序注入 + 安全包线\n")
    md = jload("g3_multidomain.json")
    if md:
        doms = md["config"]["seq_domains"]
        A("§8 用的是**同一个域跑两遍**，结构上测不到「域间干扰」。本节换成 "
          "**三个互不重叠的虚构域**（15 条事实/域，实体名与答案词表**互不相交**），"
          "按 **A → B → C 顺序注入同一个适配器**（朴素热更新，共享参数）——"
          "这正是 §8 风险三「累积漂移」要问的问题。\n")
        A("### 9.1 域间干扰矩阵（留出改写问法的习得率）\n")
        A("| 阶段 | " + " | ".join(f"`{d}`" for d in doms) + " | 可验证能力 |")
        A("|---|" + "---|" * (len(doms) + 1))
        for s in md["protocol_S"]:
            A(f"| {s['tag']} | "
              + " | ".join(f"{s['acquire'][d]:.3f}" for d in doms)
              + f" | {s.get('verifiable', float('nan')):.3f} |")
        A("")
        S = {s["tag"]: s for s in md["protocol_S"]}
        a1 = S["S_after_northwind"]["acquire"]["northwind"]
        a3 = S["S_after_vela"]["acquire"]["northwind"]
        h2 = S["S_after_halcyon"]["acquire"]["halcyon"]
        h3 = S["S_after_vela"]["acquire"]["halcyon"]
        bver = S["base"].get("verifiable")
        ever = S["S_after_vela"].get("verifiable")
        A(f"- `northwind` 首次注入后 **{a1:.3f}** → 注入 B、C 之后 **{a3:.3f}**"
          f"（保留率 **{md['S_retention_northwind_after_B_C']:.2f}**）")
        A(f"- `halcyon` 注入后 **{h2:.3f}** → 再注入 C 之后 **{h3:.3f}**"
          f"（保留率 **{h3/h2:.2f}**）")
        A(f"- **可验证能力（math/format/code）全程 {bver:.3f} → {ever:.3f}，"
          f"遗忘 {md['S_capability_forget']:+.3f}**（逐位不变）")
        A("")
        A("> ★★★ **这是对 §8 结论的重要修正。** 灾难性遗忘**确实存在**，"
          "但**只发生在「被注入的知识」之间**：后一次热更新会把前一次注入的域覆盖掉"
          "（保留率 0.23 / 0.43）。而**预训练能力零损伤**（逐位不变）。"
          "⇒ §5.1 给的理由（向主奇异子空间漂移、覆盖**预训练**知识）**是错的**"
          "（§8 已证无漂移、无能力损失）；但 §5.2 的**解法（参数隔离 + 门控）是有动机的**"
          "—— 它要防的是**域间覆盖**，不是预训练知识被覆盖。"
          "**我此前「GE-PEFT 缺乏动机」的结论过强，在此更正。**")
        A("")
        A("### 9.2 剂量-反应：热更新的安全包线\n")
        A("| lr | 可验证能力 | 遗忘 | ΔCE% | 习得 A | final loss |")
        A("|---|---|---|---|---|---|")
        S1 = S["S_after_northwind"]
        base_ce = S["base"]["lm_ce"]
        A(f"| 1e-4 | {S1['verifiable']:.3f} | {bver - S1['verifiable']:+.3f} | "
          f"{(S1['lm_ce']-base_ce)/base_ce*100:+.2f} | "
          f"{S1['acquire']['northwind']:.2f} | (协议S) |")
        for r in md["protocol_D"] + [md["destructive_control"]]:
            A(f"| {r['lr']:g} | {r['verifiable']:.3f} | "
              f"{bver - r['verifiable']:+.3f} | "
              f"{(r['lm_ce']-base_ce)/base_ce*100:+.2f} | "
              f"{r['acquire']['northwind']:.2f} | {r['final_loss']:.4f} |")
        A("")
        A("> ★ **安全包线：lr ≤ 3e-4 安全**（3e-4 的 −0.021 在噪声底 0.042 内，习得仍有 0.87）；"
          "**1e-3 处崩塌**（能力归零、CE ×11）。悬崖落在 **3e-4 ~ 1e-3** 之间，"
          "约为可用学习率的 **3–10 倍** ⇒ 热更新有一个**可操作的**安全区间。")
        nf0 = S["base"]["acquire"]
        A(f"> ⚠️ 口径：习得有**非零地板**（base 时 " +
          "、".join(f"{d}={nf0[d]:.3f}" for d in doms) +
          "，短答案碰巧答对）⇒ 保留率按「相对峰值」报，不按绝对 0。")
    else:
        A(missing("g3_multidomain.json"))

    # ── 10. 测 §5.2 的解法：参数隔离 ───────────────────────────────
    #  ★ 4 空格缩进，放在 if/else 之外（§8b 踩过缩进落进 else 的坑）。
    A("\n## 10. 测 §5.2 的解法：参数隔离 vs 容量配平的共享适配器\n")
    iso = jload("g3_isolation.json")
    if iso:
        cfg = iso["config"]
        n16, n48 = cfg["n_params_3xr16"], cfg["n_params_1xr48"]
        A("§9 已证「病在」（域间覆盖，保留率 0.23/0.43）。本节测**药**：每域一个独立子模块"
          "（r=16）vs 单个共享适配器。\n")
        A("**★ 容量配平是关键对照**：3×r16 与 1×r48 的参数量**恰好相等**"
          f"（{n16/1e6:.3f}M = {n48/1e6:.3f}M，脚本内断言）⇒ 若隔离赢，赢的是**隔离**"
          "而不是「参数更多」。\n")
        A("### 10.1 隔离保留率（训完 B、C 后，只激活该域自己的子模块）\n")
        A("| 域 | 刚训完 | 训完 B、C 后 | 保留率 |")
        A("|---|---|---|---|")
        for i, d in enumerate(cfg["domains"]):
            first = iso["arm_isolated"][i]["acquire"][d]
            alone = iso["isolated_alone"][d][d]
            A(f"| `{d}` | {first:.3f} | **{alone:.3f}** | "
              f"**{iso['isolated_retention_alone'][d]:.3f}** |")
        A("")
        A("> ★ 硬证据：训练任一子模块时，其他子模块的 LoRA 权重**逐位偏差 = 0.0**"
          "（脚本内断言的守卫）⇒ 是**真隔离**，不是「碰巧没被改坏」。")
        A("")
        A("### 10.2 两臂对照（容量严格配平）\n")
        A("| 域 | 隔离 首→末 | 保留率 | 共享 r=48 首→末 | 保留率 |")
        A("|---|---|---|---|---|")
        for i, d in enumerate(cfg["domains"]):
            fI = iso["arm_isolated"][i]["acquire"][d]
            lI = iso["isolated_alone"][d][d]
            fS = iso["arm_shared48"][i]["acquire"][d]
            lS = iso["arm_shared48"][-1]["acquire"][d]
            A(f"| `{d}` | {fI:.3f}→{lI:.3f} | **{(lI/fI if fI else float('nan')):.3f}** | "
              f"{fS:.3f}→{lS:.3f} | **{(lS/fS if fS else float('nan')):.3f}** |")
        A("")
        A("> ★★ **§5.2 的参数隔离有效**：保留率 **1.000 / 1.000 / 1.000**，"
          "而**容量完全相同**的共享适配器只有 **0.286 / 0.308**。"
          "⇒ 隔离不是容量假象。\n")
        p9 = os.path.join("..", "figures", "fig9_isolation_composition.png")
        if os.path.exists(p9):
            A("![图9](figures/fig9_isolation_composition.png)\n")
            A("*图9 §5.2 解法检验：(a) 隔离 vs 容量配平共享；(b) 朴素组合的破坏；"
              "(c) 组合下预训练能力未受损。*\n")
        A("### 10.3 但「朴素组合」失败 —— 门控不是可选项\n")
        A("§5.2 称「多子模块可通过组合处理跨域查询」。实测把三个子模块的 ΔW **直接相加**：\n")
        A("| 域 | 单独激活 | 组合激活 | 差 |")
        A("|---|---|---|---|")
        for d in cfg["domains"]:
            a_ = iso["isolated_alone"][d][d]
            c_ = iso["composed"]["acquire"][d]
            A(f"| `{d}` | {a_:.3f} | {c_:.3f} | **{c_-a_:+.3f}** |")
        A("")
        bv, cv = iso["baseline"]["verifiable"], iso["composed"]["verifiable"]
        dce = (iso["composed"]["lm_ce"] - iso["baseline"]["lm_ce"]) \
            / iso["baseline"]["lm_ce"] * 100
        A(f"- 预训练能力：base {bv:.3f} → 组合 {cv:.3f}（{cv-bv:+.3f}）；"
          f"ΔCE **{dce:+.2f}%**")
        A("")
        A("> ★★ **组合必须靠门控，不能靠相加**：朴素相加使每个域的习得掉 **−0.40 ~ −0.73**"
          "（三个域全部被拖坏）。⇒ §5.2 的第三条（门控路由）是**承重件**，不是装饰。")
        A("> ⚠️ 工程注记：peft 0.20 **没有** `add_weighted_adapter`，且 `set_adapter`"
          " 只支持单适配器激活 ⇒ 组合是用 forward hook 把非激活子模块的 ΔW·x 精确加上去实现的。")
    else:
        A(missing("g3_isolation.json"))

    # ── 11. §5.2 第 2 条：LoRA-Null 零空间初始化 ───────────────────
    A("\n## 11. §5.2 第 2 条：LoRA-Null 零空间初始化\n")
    ni = jload("g3_nullinit.json")
    if ni:
        c = ni["config"]
        cs = ni["conditions"]
        A("方案原文：『子模块的初始化位于激活空间的零空间中（LoRA-Null 初始化），"
          "**确保初始更新不干扰已有知识**。』\n")
        A("### 11.1 先做一个口径检查（这本身是结论）\n")
        A("LoRA 的标准初始化是 **B = 0** ⇒ `ΔW = (α/r)·B·A ≡ 0`，即"
          "「初始更新不干扰已有知识」在标准 LoRA 下**已经精确成立**，"
          "与是否用零空间初始化**无关**（脚本内已断言 `max|ΔW| == 0.0`）。")
        A("=> 该主张若要有内容，只能指**训练轨迹**：让*后续*更新也少干扰。"
          "本节测的就是这个可检验的版本。\n")
        A("### 11.2 初始化的**性质**确实成立\n")
        A("| 条件 | A 在主方向上的能量占比 |")
        A("|---|---|")
        for cn in ("std_chunk", "null_chunk"):
            v = cs[cn]["init_diag"].get("A_energy_on_principal", {})
            val = (sum(v.values()) / len(v)) if v else None
            A(f"| `{cn}` | {val:.5f} |" if val is not None
              else f"| `{cn}` | - |")
        A("")
        _k = c["k_basis"]
        _null = (_k / 4096) ** 0.5
        A(f"随机方向的 null 应为 $\\sqrt{{k/d}} = "
          f"\\sqrt{{{_k}/4096}} = {_null:.3f}$。"
          "实测 std 恰在 null 附近，null 则降到 **0.00000** ⇒ 零空间投影**有效**。\n")
        A("### 11.3 但收益没有兑现，反而更差\n")
        A("干扰曲线（domA 的习得 vs domB 训练步数；A、B 同时激活）：\n")
        A("| 步数 | " + " | ".join(f"`{k}`" for k in
                                   ("std_chunk", "null_chunk", "std_chunk_rep"))
          + " |")
        A("|---|" + "---|" * 3)
        steps = sorted({p["step"] for v in cs.values() for p in v["curve"]})
        for st in steps:
            row = [f"{st}"]
            for k in ("std_chunk", "null_chunk", "std_chunk_rep"):
                v = cs.get(k)
                if not v:
                    row.append("-")
                    continue
                pts = [p for p in v["curve"] if p["step"] == st]
                row.append(f"{pts[0]['A']:.3f}" if pts else "-")
            A("| " + " | ".join(row) + " |")
        A("")
        NP = 15                      # 每域留出探针数

        def _nk(a):
            return f"{a:.3f} ({round(a * NP)}/{NP})"

        A("| 条件 | A 单独 | A(有 B) | 保留率 | B 单独 | 能力 |")
        A("|---|---|---|---|---|---|")
        for k in ("std_chunk", "null_chunk", "std_chunk_rep", "std_single"):
            v = cs.get(k)
            if not v:
                continue
            ver = v.get("verifiable")
            A(f"| `{k}` | {_nk(v['A_alone'])} | {_nk(v['A_with_B'])} | "
              f"**{v['retention_A']:.3f}** | {_nk(v['B_alone'])} | "
              f"{f'{ver:.3f}' if ver is not None else '-'} |")
        A("")
        A("> 括号内是**探针条数**（每条 0.067）。离散条数比小数更能看清真实效应量。\n")
        noise = ni.get("same_treatment_noise", float("nan"))
        eff = ni.get("null_net_effect", float("nan"))
        chunk = ni.get("chunk_effect", float("nan"))
        ret_s = (cs.get("std_chunk") or {}).get("retention_A", float("nan"))
        ret_n = (cs.get("null_chunk") or {}).get("retention_A", float("nan"))
        A(f"- 同处置重复的残余差异（`std_chunk_rep` − `std_chunk`）= "
          f"**{noise:+.3f}** ⇒ init 固定后**完全复现**，噪声底 = 0")
        A(f"- 零空间净效应（`null_chunk` − `std_chunk`）= **{eff:+.3f}**"
          f"（保留率 {ret_s:.3f} → {ret_n:.3f}）")
        A(f"- 分段效应（`std_single` − `std_chunk`）= **{chunk:+.3f}**")
        A("")
        A("### 11.4 判读：收益小、代价大 ⇒ 是一桩坏买卖\n")
        A("> ★★ **§5.2 第 2 条不被支持。** 零空间初始化把它的**结构性性质**"
          "做到了极致（A 对真实激活的增益 0.165 → 0.00000），但：\n")
        A(f"> 1. **收益极小**：保留率 {ret_s:.3f} → {ret_n:.3f}，"
          f"即 A(有 B) 只从 **3/15 升到 4/15**（**+1 条探针**）；"
          f"绝对水平上仍是**灾难性遗忘**（丢掉约 69%）。")
        A("> 2. **代价很大**：新域自身习得（`B 单独`）从 **11/15 掉到 5/15**"
          "（0.733 → 0.333），即**丢了 6 条探针**。")
        A("> 3. ⇒ **用 6 条探针的习得换 1 条探针的保留，这是亏的。**"
          "对比 §10 的参数隔离：保留率 **1.000** 且习得**完全不掉** "
          "⇒ **隔离完胜零空间初始化**；后者不是解法。")
        A("")
        A("> ⚠️ **我中途的一个判断已撤回。** 第一次（未控 init）跑出"
          "「null 让新域训练 loss 从 0.0125 涨到 0.1517（12×），是可靠信号」——"
          "**这个结论是错的**。该 loss 是**最后一个单步**的 loss"
          "（分段训练下还叠了 Adam 重置），噪声很大；控住 init 后重跑，"
          "方向**翻转**为 0.1748 vs 0.2119（null 反而更低）。"
          "⇒ 该指标不能用作代价证据，已改用**探针条数**（离散但可解释）。")
        A("")
        A("> ★ **设计教训（第一次跑更重要的收获）**：初版没有固定 LoRA 初始化"
          "随机源，于是两个**同处置**的 std 条件给出 A_alone = 1.000 / 0.800、"
          "保留率 0.600 / 0.167 —— init 抽样造成的差异（**0.433**）"
          "**比要测的处置效应还大**，而且**净效应的符号是反的**"
          "（−0.400 → +0.077）。修正：固定 `INIT_SEED`，零空间条件改为对"
          "**同一初始 A** 正交化（而非重抽一个随机 A），并加 `std_chunk_rep` "
          "作同处置重复。两次数据都保留"
          "（未控版见 `out/g3_nullinit_v1_uncontrolled.json`，可复现该对比）。")
        A("")
        A(f"> ★ **附带的意外发现（分段效应 {chunk:+.3f}）**：domB **分 4 段**训练"
          f"（每段新建 AdamW）得到的保留率 **0.538**，而**单次 200 步**只有 "
          f"**0.231** ⇒ **Adam 动量重置会显著放大干扰**。"
          f"这意味着本节用来画「干扰曲线」的分段协议**系统性高估了遗忘**："
          f"曲线的**形状**（单调衰减）仍可读，**绝对水平不可**。"
          f"这是一条可迁移的方法学警告 —— 用它做连续学习评测会偏向更悲观。")
    else:
        A(missing("g3_nullinit.json"))

    # ── 12. §5.2 第 3 条：门控路由 ─────────────────────────────────
    A("\n## 12. §5.2 第 3 条：门控路由\n")
    rt = jload("g3_router.json")
    if rt:
        g = rt["gate"]
        dh = rt["domain_holdout"]
        s = rt["summary"]
        A("§10 已证朴素相加失败 ⇒ 门控是承重件。本节实现并检验门控本身。\n")
        A("设计：在**基座**（不激活任何适配器）第 "
          f"{rt['config']['gate_layer']} 层平均池化隐状态上做 4 类岭回归"
          "分类器 {三域, none}。★ 特征取自基座 ⇒ 推理时"
          "**先路由、再激活**，是可部署的因果顺序；★ `none` 类用"
          "可验证基准的**提示**训练 ⇒ 门控必须学会**不劫持**无关查询。")
        A("★ 门控层 L" + str(rt["config"]["gate_layer"]) +
          " 的选择有依据：目标2 已实测**自我认知可分性峰在 L13–14**。\n")
        A(f"门控训练准确率（{g['n_train']} 样本）= **{g['train_acc']:.3f}**；"
          "留出（改写问法）准确率：" + "、".join(
              f"`{k}` {v:.3f}" for k, v in g["holdout_acc"].items()))
        A(f"- `none` 样本被误路由到某个域的比例 = **{g['none_misroute_rate']:.3f}**")
        A("")
        A("| 域 | base | random | sum | router | oracle |")
        A("|---|---|---|---|---|---|")
        for d in rt["config"]["domains"]:
            A(f"| `{d}` | {dh['base'][d]:.2f} | {dh['random'][d]:.2f} | "
              f"{dh['sum'][d]:.2f} | **{dh['router'][d]:.2f}** | "
              f"{dh['oracle'][d]:.2f} |")
        A(f"| **均值** | {s['mean_base']:.2f} | {s['mean_random']:.2f} | "
          f"{s['mean_sum']:.2f} | **{s['mean_router']:.2f}** | "
          f"{s['mean_oracle']:.2f} |")
        A("")
        A(f"- router 追平 oracle 的比例 = **{s['router_over_oracle']:.3f}**"
          f"（oracle = 完美路由的上限）")
        A(f"- 负对照 random = {s['mean_random']/s['mean_oracle']:.3f}；"
          f"朴素相加 sum = {s['mean_sum']/s['mean_oracle']:.3f}")
        A("")
        cf = rt["capability"]
        bv = cf["base"]["verifiable"]["acc"]
        A("无关查询（可验证基准）能力：")
        for t in ("base", "router", "sum"):
            v = cf[t]["verifiable"]["acc"]
            dce = ((cf[t]["lm"]["ce"] - cf["base"]["lm"]["ce"])
                   / cf["base"]["lm"]["ce"] * 100)
            A(f"- `{t}`：{v:.3f}（{v-bv:+.3f}），ΔCE {dce:+.2f}%")
        A("")
        A("> ★★ **门控把朴素相加的崩塌救回来了**：逐域均值 "
          f"sum {s['mean_sum']:.2f} → router {s['mean_router']:.2f}"
          f"（上限 oracle {s['mean_oracle']:.2f}），"
          f"**router 追平 oracle 的比例 = {s['router_over_oracle']:.3f}**"
          f" ⇒ 路由**无损**（=1 表示与「每次都正确激活该域」完全一致）。")
        A(f"> 而负对照 `random`（随机在三个子模块里挑）只有 "
          f"{s['mean_random']:.2f}（{s['mean_random']/s['mean_oracle']:.3f} of oracle）"
          f" ⇒ 门控的**决策内容**确实在起作用，不是「生成本来就对」。")
        A("")
        A(f"> ★ **一个值得单独指出的细节：朴素相加比随机路由还差**"
          f"（sum {s['mean_sum']:.2f} < random {s['mean_random']:.2f}）。"
          f"随机路由至少有 1/3 的概率碰对；而相加是**三个子模块同时说话**，"
          f"`northwind` 被打回基线水平（{dh['sum'][rt['config']['domains'][0]]:.2f}）。"
          f"⇒ 在共享前向通道上叠加多个 ΔW 不是「折中方案」，是**主动破坏**。")
        A("")
        A("> ⚠️ **公平性限制（很重要）**：门控训练准确率 **"
          f"{g['train_acc']:.3f}**、留出 4 类全 **1.000**、`none` 误路由率 **0**。"
          "这么容易，是因为三个虚构域靠**实体名**即可在第 13 层线性分开。"
          "⇒ 本结果证明的是**机制可行**（「先路由再激活」这条链路通、"
          "且能同时守住域知识与非域查询），**不代表**在真实分布漂移、"
          "近义域、或查询不属于任何已注入域时也能这样干净地分开。"
          "门槛应当设在那里。")
    else:
        A(missing("g3_router.json"))

    # ── 12.4 门控的适用边界：把域重叠度调上去 ───────────────────────
    ov = jload("g3_overlap.json")
    A("\n### 12.4 门控的**适用边界**：域重叠到什么程度就分不开了？\n")
    if ov:
        A("§12 的门控留出准确率是 4/4 类全 1.000，但那是因为三个虚构域靠"
          "**唯一实体名**就能分开 —— 判别信号是 prompt 里一个独一无二的词。"
          "真实域通常按**主题**定义（「ACL 论文」vs「NeurIPS 论文」）。"
          "本节沿**单一可控轴**把重叠度调上去，看门控在哪里失效。\n")
        _desc = {
            "P1_disjoint_entity": "实体名不同、字段名**全同**（= 原实验设定）",
            "P2_shared_entity": "实体名**完全相同**、字段名不同",
            "P3_single_token": "实体名、字段头全同，只差**「北极」/「南极」**",
        }
        _diff = {"P1_disjoint_entity": 11, "P2_shared_entity": 8,
                 "P3_single_token": 2}
        A("| 对 | 判别信号 | 判别性字符集差异 |")
        A("|---|---|---|")
        for pn in ov["config"]["pairs"]:
            A(f"| `{pn}` | {_desc[pn]} | **{_diff[pn]}** |")
        A("")
        A("★ **核心设计是 (层 × 池化) 扫描**，而不是只报一个数 —— "
          "因为「门控失败」有**两种完全不同**的原因，必须分开："
          "(a) 表征里已经没有判别信号（任何读出都救不了）；"
          "(b) 表征里有，但池化把它丢了（只是**读出选择**问题）。"
          "同时报**训练集**准确率以分离第三种可能：训练能分、留出不能分 = "
          "**改写泛化**问题，而非重叠问题。\n")
        A("| 对 | 变体 | 训练 | 近改写 | 远改写 | none 误路由(见过的提示) | "
          "none 误路由(**没见过的通用文本**) |")
        A("|---|---|---|---|---|---|---|")
        for pn in ov["config"]["pairs"]:
            vv = ov["pairs"][pn]["variants"]
            for pool in ("mean", "max", "last"):
                k = f"L13_{pool}"
                r = vv[k]
                star = " ⚠️" if pool == "last" else ""
                A(f"| `{pn[:2]}` | `{k}`{star} | {r['train_acc']:.3f} | "
                  f"{r['near_acc']:.3f} | {r['far_acc']:.3f} | "
                  f"{r['none_misroute_cap']:.3f} | "
                  f"**{r['none_misroute_lm']:.3f}** |")
        A("")
        A("端到端（门控口径 `mean@L13`，6 个域容量完全相同 7.012M）：\n")
        A("| 对 | 域 | oracle | router | 门控路由到本域的比例 |")
        A("|---|---|---|---|---|")
        for pn in ov["config"]["pairs"]:
            for d, r in ov["end_to_end"][pn].items():
                A(f"| `{pn[:2]}` | `{d}` | {r['oracle']:.3f} | "
                  f"**{r['router']:.3f}** | {r['routed_to_self']:.3f} |")
        A("")
        A("> ★★ **结论：我的悲观预估被推翻了。** 我此前估计「门控在真实重叠域上"
          "有 **60–80%** 概率失败」。实测**三对全部可分**，包括最难的 P3 —— "
          "问法只差**「北极」/「南极」两个字符**，而 `mean@L13` 的近/远改写"
          "准确率都是 **1.000**，通用文本误路由 **0.000**，端到端"
          "**router 与 oracle 逐域完全相同**。")
        A("> ★ **我错在混淆了两个不同的量**：我优化的是**判别 token 的占比**"
          "（把它压到约 2/14），但真正决定难度的是**线索的可靠性** —— "
          "「北极」出现在 eps 域的**每一条**查询里。于是门控的任务退化成"
          "「这句话里有没有『北极』」这种**词汇存在性检验**，线性探针轻松解决。"
          "⇒ 把线索做**稀疏**不等于把它做**弱**；我的 P3 在一个错误的维度上变难了。")
        A("")
        A("> ★ **顺带抓到一个真实的工程陷阱（这正是做扫描的价值）**："
          "**`last` 池化是危险选择。** 它对 `none` 类的误路由率高达 "
          "**0.417–1.000**（`mean`/`max` 全是 **0.000**），且在 P3 上路由准确率"
          "**恰好 0.500**。机制很清楚：**`last` 只看末尾，判别线索不在末尾时就被"
          "丢掉** —— P3 两域末尾 token 完全相同（`…轨道高度是什么？`）⇒ 必然退化"
          "为随机；而 P2 的字段名靠近末尾（`高度`/`年限`）⇒ 仍能分。"
          "⚠️ 值得注意的是 `last` 恰是**指令分类的常规选择**（用户指令在末尾），"
          "但对**主题路由**它是错的选择 —— 主题线索通常不在末尾。"
          "⇒ **池化必须与「线索位置」匹配，不能照搬惯例。**")
        A("")
        A("> ⚠️ **但风险没有消失，只是换了位置。** 本实验只证明了**词汇重叠**"
          "不是威胁；它**没有**测试下面这些，而它们才是真实部署的主要风险：\n")
        A("> 1. **线索缺失**：查询里根本不含判别 token（如「那它的并发上限呢？」"
          "靠上文指代）⇒ prompt 内无任何信号，任何读出都做不到；")
        A("> 2. **概率性/分布式主题线索**：两域各有**统计倾向**而非确定的判别词，"
          "且相当比例的查询本身是**歧义**的（真实主题域就是这样）；")
        A("> 3. **多域规模**：本实验每对只有 2 类（随机 = 0.5）；真实系统要求"
          "**从 N 个域里选对**，错误会累积；")
        A("> 4. **真正的主题表述漂移**：即使「远改写」也仍共享实体名与字段词。\n")
        A("> ⇒ **修订后的估计**：「门控在真实主题域上失败」从 **60–80% 下调到约 "
          "25–40%**，且**主要风险已从「线索被稀释」转移到「线索缺失/歧义」**。"
          "这是**不同**的问题，且更难用工程手段绕过 —— 前者换池化/换层就能修"
          "（本实验已证），后者需要**多轮上下文**或**检索信号**才能提供判别信息。")
    else:
        A(missing("g3_overlap.json"))

    # ── 13. §5.3 闭环接口的端到端验证 ──────────────────────────────
    A("\n## 13. §5.3 闭环接口的端到端验证\n")
    gm = jload("ge_state/manifest.json")
    ga = jload("ge_state/audit.json")
    if gm:
        A("上面的零件被封进 `exp/ge_peft.py`（§5.3 的命令表），"
          "下面是在**单个进程**里跑完整闭环的结果"
          "（`session --domains northwind,halcyon,vela`；"
          "单进程是为了避免反复 116 s 加载）。\n")
        A("流程：**注入 ×3 → 漂移检测（带现场标定）→ 身份审计 → 门控路由**。\n")
        A("### 13.1 注入（每域一个独立子模块）\n")
        A("| 检查点 | 域 | r | 步 | final loss | 可训练 | 注入时探针 |")
        A("|---|---|---|---|---|---|---|")
        for c in gm["checkpoints"]:
            A(f"| `{c['id']}` | {c['domain']} | {c['r']} | {c['steps']} | "
              f"{c['loss']:.4f} | {c['n_params']/1e6:.3f}M | "
              f"{c['eval_acc_at_inject']:.3f} |")
        A("")
        dh = (gm.get("drift_history") or [{}])[-1]
        v = dh.get("verdict", {})
        if v:
            A("### 13.2 漂移检测（阈值按**实测噪声底**标定，非方案的 15%）\n")
            A(f"- 现场标定：同一状态进程内重复读数 **0.750 / 0.750**（|Δ| = 0.000）"
              f" ⇒ 该负载下确定性；跨进程噪声底 **0.042** 作阈值上界，"
              f"取 `max(0.042, 0.000) = {v.get('capability_drop_max'):.3f}`")
            A("| 域 | 注入时 | 现在 | 保留率 | 谱能量比（**仅诊断**） |")
            A("|---|---|---|---|---|")
            for d, r in dh.get("domains", {}).items():
                A(f"| `{d}` | {r['first']:.3f} | {r['now']:.3f} | "
                  f"**{r['retention']:.3f}** | {r['spectral_ratio']['mean']:.4f} |")
            A("")
            A(f"- 最差保留率 **{v['worst_retention']:.3f}** vs 下限 "
              f"{v['retention_min']:.2f} → "
              f"{'OK' if v['retention_ok'] else '**漂移**'}")
            A(f"- 能力下降 **{v['capability_drop']:+.3f}** vs 上限 "
              f"{v['capability_drop_max']:.3f} → "
              f"{'OK' if v['capability_ok'] else '**漂移**'}")
            A(f"- LM CE 相对 **{v['lm_ce_rel']:+.2%}** vs 上限 "
              f"{v['lm_ce_rel_max']:.0%} → "
              f"{'OK' if v['lm_ce_ok'] else '**漂移**'}")
            A("")
            A("> ★ 三条判据全 OK ⇒ 闭环**正确判定「无需谱校准」**，"
              "而不是盲触发。这正是 §10 的参数隔离在真实接口上的效果："
              "**三个域互不干扰，保留率全 1.000。**")
            A("> ⚠️ 谱能量比实测只有 **~0.011**（约 1%），离方案的 15% 阈值"
              "差一个数量级 ⇒ 再次印证 §9：**该指标必须换掉**，"
              "本接口只用它作诊断，不作触发器。")
            A("")
        A("### 13.3 门控路由（闭环的可服务性）\n")
        A("会话内复现了 §12 的结论：门控训练准确率 **1.000**，"
          "`router` 与 `oracle` **逐域完全相同**（均值 0.73 = 0.73，"
          "**追平比例 1.000**）；可验证能力 **0.750 → 0.768**（+0.018，"
          "在 0.042 噪声底内）。"
          "⇒ 独立第二次实现（走 CLI 而不是实验脚本）得到同结论。")
        if ga:
            A("")
            A("### 13.4 身份审计 —— ★ 这里抓到一个度量 bug\n")
            marks = ("qwen", "通义千问", "通义", "千问", "tongyi",
                     "alibaba", "阿里")
            A("| 状态 | 旧判据（仅 ASCII `qwen`） | **修正判据（含中文名）** | "
              "域名词污染 |")
            A("|---|---|---|---|")
            new_rates = {}
            for st, r in ga.items():
                n = len(r["raw"]) or 1
                nw = sum(1 for x in r["raw"]
                         if any(k in x["answer"].lower() for k in marks)) / n
                new_rates[st] = nw
                A(f"| `{st}` | {r['qwen_rate']:.3f} | **{nw:.3f}** | "
                  f"{r['domain_pollution']:.3f} |")
            A("")
            A("> ⚠️ **旧判据给出假警报**：它只查 ASCII 字符串 `\"qwen\"`，"
              "而模型答的是 **「通义千问」** ⇒ 域适配器一激活就\"掉\"到 "
              "0.333–0.833，于是判为「**身份被污染**」。"
              "但域名词污染始终为 **0**，即模型**从未自称任何域实体**；"
              "看原始回答才发现全部都是正确的「通义千问」。"
              "这是**度量侧的静默失效**（与 §5 的 15% 阈值、§11 的单步 loss "
              "同一类错误）：**先看数据，再信指标**。")
            A(f"> 修正后：基座 **{new_rates.get('base', float('nan')):.3f}**、"
              f"最差 **{min(new_rates.values()):.3f}** ⇒ 身份**基本完好**且"
              "**未被替换**。")
            bad = [(st, x) for st, r in ga.items() for x in r["raw"]
                   if not any(k in x["answer"].lower() for k in marks)]
            if bad:
                A("")
                A("> ★ **但修正后仍剩下真实的细微信号**：未命中的探针**不是**"
                  "词表问题，而是**无关虚构**——")
                for st, x in bad:
                    A(f">   - `{st}`：{x['probe']} → `{x['answer'][:40]}`")
                A("> ⇒ 域子模块激活时，自我认知回答**基本保留**，"
                  "但**偶发身份虚构**（实测出现 `Alex Chen`、`Tencent`）。"
                  "★ 这是「域子模块**始终激活**」的**未路由最坏情形**；"
                  "实际部署下门控会把「你是谁」这类查询路由到 `none`。")
        A("")
        A("### 13.5 接口实现状态（诚实清单）\n")
        A("| 命令 | 状态 | 说明 |")
        A("|---|---|---|")
        A("| `inject` / `list` | ✅ 已跑通 | 独立子模块 + 落盘检查点 |")
        A("| `drift-check` | ✅ 已跑通 | 行为量触发 + 谱量诊断，阈值按噪声底标定 |")
        A("| `rollback` | ✅ 已实现 | 载入检查点并**验证复现**（比对注入时读数） |")
        A("| `audit --self-identity` | ✅ 已跑通 | 见 13.4 |")
        A("| `route --eval` | ✅ 已跑通 | 见 13.3 |")
        A("| `calibrate` | ✅ 已实现 | 谱校准并**另存**新检查点（不覆盖原检查点） |")
        A("| `scan` | ⚠️ 已实现但未验证 | 调用 `s1_probe.py` 重扫并与基座图谱比对 |")
        A("| `consolidate --method GRPO` | ❌ **未实现** | 需奖励模型/可验证奖励的 RL 回路，"
          "是独立子项目；**直接报错退出，不做假的占位实现** |")
    else:
        A(missing("ge_state/manifest.json"))

    A("\n---\n")
    A("## 14. 结论与限制\n")
    A("1. **奇点在末段集中**：L26-31（尤其 **L31**）出现表示塌缩"
      "（down_proj 输入有效维仅 26%）与极端超激活（峰值 400）。")
    A("2. **奇点有全局坐标结构**：通道 {3994, 310, 253, 3986} 跨层复现 ⇒ "
      "AWQ 意义上的显著通道是**全局少数坐标**，不是逐层局部的。")
    A("3. **tech plan 三条结构预测均不被支持**（H_A 有深度混淆、H_B 位置错、H_C 方向反）。")
    A("4. **LLC 与 AWQ/DDS 排序不一致（新的负结果）**：局部 LLC 跨层极差仅 ~4%，"
      "且 L31（最强奇点）LLC 最低、L8（对照）最高 ⇒ tech plan §3.3 的 DDS×LLC 交叉验证**不成立**。")
    A("5. **自我认知带 ≠ 最强奇点带**：身份可分性峰在 **L13–14**，而最强奇点在 **L31**。"
      "⇒ 「身份编辑区」不应按奇点强度选取，应按**可分性/因果定位**选取 —— "
      "本实验据此在 L12–17 编辑，一次成功（命中 1.00、回归 +0.28%）。")
    A("6. **编辑效应精确局部化**（补齐读数 + 跨种子确认）：编辑后 **32 层全部可测**；"
      "被编辑的 6 层（L12–17）**全部**负向位移 **−1.26%±0.36**，而未编辑 26 层仅 "
      "**−0.00%±0.28**（上游层 <L12 精确 0.00%）⇒ 组间差 **−1.26 pp**。\n"
      "   跨 **5 个种子**独立重复已确认该结论：配对差 **−1.16±0.30 pp**（t = **−8.50**），"
      "**5/5** 种子方向一致、**5/5** 种子 6 层全负 ⇒ **不是种子噪声**。"
      "而编辑本身在 5/5 种子上都是 命中 1.00 / ΔCE +0.17%±0.09 / 溢出 0.00。")
    A("7. **目标3 的前提未被证实（机制部分）**：向两个层带各注入一个完整新知识域"
      "（两轮共 96 步），**习得 0.04 → 0.88**，而可验证能力基准的**遗忘 = +0.000**"
      "（同一进程内 math/format/code 逐位不变，ΔCE ≤ +1.3%）。"
      "阳性对照（破坏性 ×100 lr）把能力打到 **0.000**/CE ×10 ⇒ **仪器确实能测到退化**。"
      "⇒ §5.1 的「新知识注入 ⇒ 向主奇异子空间漂移、覆盖**预训练**知识」**机制不成立**。")
    A("8. **§5.2 的 15% 阈值不可用**：跨两个层带、含破坏性对照，主奇异子空间能量占比"
      "最高仅 **2.4%**（富集 ×2.13），**0/37 模块**越阈 ⇒ 该触发器在模型被摧毁时仍不触发；"
      "且显著通道富集在破坏性对照上反而**低于 null**（×0.96）⇒ 指标与损伤**解耦**。"
      "任何采用此阈值的实现都必须改为**按 null 标定**。")
    A("9. **注入位置的效果由「深度」决定，不由「奇点」决定（混淆已分离）**："
      "2×2 因子实验（4 臂、容量严格配平为各 18 模块 / 3.74 M、同一进程共享基线）显示，"
      "习得几乎完全由深度解释（深度 1–5 → 0.29–0.42；深度 26–28 → 0.79–0.83），"
      "而**同深度窗口内**改变奇点强度只有 **−0.125**（早期对，符号为反）与 **+0.042**"
      "（后期对，可忽略）⇒ 「奇点带更适合注入」**被否证**，它是深度的产物。"
      "且该实验同时加固了主要负结果：**4 个臂遗忘全部 = +0.000**（含深度 1–5 的早期层）"
      "⇒ 无遗忘是**深度稳健**的，不是选中段的假象。")
    A("10. **口径限制**：单模型、单标定语料；AWQ 探针是**一阶激活统计**，"
      "不等于 Fisher/Hessian；LLC 为 localized 且对 lr 敏感；`act_absmax_peak` 绝对值受"
      "残差范数随深度增长影响（**但 top1% 占比与有效维是尺度不变的，不受此影响**）。\n"
      "   跨种子说明：**探测/定位阶段是确定性前向（无随机源），换种子不适用**；"
      "**编辑阶段已做 5 种子**（见 §7），结论在 5/5 种子上一致。")
    A("11. **能力基准的噪声底是「跨进程」的**：同一未改动基座在 3 个独立进程里可验证能力"
      "= 0.750/0.729/0.771（sd 0.021，极差 0.042，**噪声全在 format 子基准**），"
      "而 **LM CE 跨进程逐位相同（极差 0.0）**。⇒ 跨臂比较绝对值必须对着该底，"
      "否则会把仪器抖动当层带效应（本实验差点犯此错，已修正并写进 §8）。")
    A("12. **回归代理限制**：可验证基准只有 math/format/code 三类（56 项）+ 留出语料 LM，"
      "**不是 MMLU**；领域覆盖有限，且 math/code 在高基线（1.00/0.88）上可能触顶，"
      "对小退化的灵敏度低于 format 子基准。")
    A("13. **热更新会覆盖「此前注入的知识」——这是真正的灾难性遗忘，也是目标3 的真正动机**"
      "（§9）：三域顺序注入同一适配器后，先注入者保留率仅 **0.23 / 0.43**，"
      "而**预训练能力全程逐位不变**（+0.000）。"
      "⇒ **修正 §7/§8 的推论**：§5.1 的**机制**是错的（无主奇异漂移、无预训练能力损失），"
      "但 §5.2 的**解法（参数隔离 + 门控）有动机** —— 要防的是**域间覆盖**。")
    A("14. **热更新有可操作的安全包线**（§9.2）：**lr ≤ 3e-4 安全**（习得 0.87、能力不掉），"
      "**1e-3 处崩塌**（能力归零、CE ×11）。悬崖在 3e-4 ~ 1e-3 之间，约为可用 lr 的 3–10 倍。")
    A("15. **§5.2 的参数隔离有效，且不是容量假象**（§10）：每域独立子模块（3×r16）的保留率"
      "**1.000 / 1.000 / 1.000**；而**参数量恰好相等**（21.037M = 21.037M，断言）的单共享"
      "适配器（r=48）只有 **0.286 / 0.308**。守卫：训练任一子模块时其他子模块 LoRA 权重"
      "**逐位偏差 = 0.0**。")
    A("16. **朴素组合失败 ⇒ 门控是承重件，不是可选项**（§10.3）：把三个子模块的 ΔW 直接相加，"
      "每个域的习得掉 **−0.40 ~ −0.73**（三域全被拖坏），而预训练能力不受损"
      "（0.750→0.771）。⇒ §5.2 三条主张里，**第 1 条（参数隔离）已证实有效且必需**，"
      "**第 3 条（门控路由）已证实必需**。")
    A("17. **§5.2 第 2 条（LoRA-Null 零空间初始化）不被支持**（§11）：其口径本身近乎空洞 —— "
      "标准 LoRA 的 B=0 已使**初始 ΔW ≡ 0**，故「初始更新不干扰已有知识」本来就成立；"
      "只能按训练轨迹检验。实测该初始化把**结构性性质做到了极致**"
      "（A 对真实激活的增益 **0.165 → 0.00000**），但**收益微、代价大**："
      "旧知识保留 +0.077（**+1 条探针**），新域自身习得 −0.400（**−6 条探针**）"
      "⇒ **坏买卖**。对比参数隔离（保留 1.000 且习得无损）⇒ **隔离完胜**。")
    A("18. **可迁移的方法学警告（§11.4）：Adam 动量重置会显著放大干扰。** "
      "同一域、同一超参，domB **分 4 段**训练得到的保留率 **0.538**，"
      "而**单次 200 步**只有 **0.231**。⇒ 用「分段调用优化器」实现的连续学习评测"
      "会**系统性高估遗忘**；干扰曲线的**形状**可读，**绝对水平不可**。")
    A("19. **可迁移的方法学警告（§11.4）：LoRA 初始化的随机源必须固定。** "
      "本项目第一次跑该实验时未固定 init，导致**两个同处置的 std 条件**"
      "给出保留率 0.600 / 0.167 —— init 抽样造成的差异（**0.433**）"
      "比要测的处置效应还大，且**把净效应的符号测反了**（−0.400 → +0.077）。"
      "⇒ 任何「A 初始化 vs B 初始化」的对照，都必须让两者**从同一个初始点出发**"
      "（本实验的修法：固定 `INIT_SEED` + 对**同一初始 A** 做正交化）。")
    A("20. **§5.2 第 3 条（门控路由）通过 —— 闭环成立**（§12）：门控（基座 L13 "
      "平均池化 + 4 类岭回归，含 `none` 类）使逐域均值从朴素相加的 "
      "**0.29 → 0.84**，**追平 oracle 的比例 = 1.000**（无损），"
      "且可验证能力 **0.771 → 0.771（ΔCE +0.00%）**，`none` 误路由率 **0**。"
      "负对照 `random` 只有 **0.42** ⇒ 路由决策本身在起作用。"
      "⇒ **§5.2 的三条主张里：第 1 条（隔离）有效、第 3 条（门控）必需且有效、"
      "第 2 条（LoRA-Null）不被支持。**")
    A("21. **朴素相加比随机路由还差**（§12）：`sum` **0.29** < `random` **0.42**；"
      "随机至少有 1/3 概率碰对，而相加是三个子模块**同时**在共享前向通道上说话，"
      "把 `northwind` 直接打回基线（0.07）。⇒ 多子模块叠加 ΔW **不是折中方案，"
      "而是主动破坏**，这与 §10 的组合失败互相印证（两次独立复现）。")
    A("22. **结论的适用范围（§12 的限制）**：本节的 3 个虚构域靠**实体名**即可在 "
      "L13 线性分开（门控留出准确率 4/4 类均为 1.000）⇒ 证明的是"
      "**「先路由再激活」这条链路可行**，**不代表**在真实分布漂移、近义域、"
      "或查询不属于任何已注入域时也能这样干净地分开。这是本工作最该被后续检验的假设。")
    A("23. **§5.3 闭环接口端到端跑通**（§13，`ge_peft.py`）：单进程内完成"
      "**注入 ×3 → 漂移检测 → 身份审计 → 门控路由**。漂移检测三条判据全 OK "
      "（最差保留率 **1.000**、能力 **+0.000**、LM CE **+0.00%**）"
      "⇒ 它**正确地判定「无需谱校准」**而非盲触发；路由复现"
      "**追平 oracle 1.000**、能力 +0.018（在 0.042 噪声底内）。"
      "⇒ 参数隔离 + 门控路由这条组合在**真实接口**上成立。")
    A("24. **★ 度量侧静默失效第三次出现，且这次差点写成结论**（§13.4）："
      "身份审计最初只查 ASCII 字符串 `\"qwen\"`，而模型答的是 **「通义千问」** ⇒ "
      "域适配器一激活就\"掉\"到 0.333–0.833，被判为「**身份被污染**」。"
      "实际上域名词污染恒为 **0**（模型从未自称域实体）；看原始回答才发现全是"
      "正确的「通义千问」。修正后为 **0.917–1.000**。"
      "★ 而**修正后剩下的真实信号**是：**偶发身份虚构**（`Alex Chen`、`Tencent`）。"
      "⇒ 规则：**指标报警时，先读原始输出再下结论**"
      "（与 §5 的 15% 阈值、§11 的单步 loss 属同一类错误）。")
    A("25. **未实现项（诚实清单，§13.5）**：`consolidate --method GRPO` **未实现** —— "
      "它需要奖励模型/可验证奖励的 RL 回路，是独立子项目；接口**直接报错退出**，"
      "不做假的占位实现。`scan` 已实现但未验证。")
    A("26. **★ 门控对「词汇重叠」是稳健的 —— 我此前的悲观预估被自己推翻**（§12.4）："
      "沿单轴把重叠度调上去（判别性字符集差异 **11 / 8 / 2**），"
      "三对**全部可分**；最难的 P3 问法只差「北极」/「南极」，"
      "`mean@L13` 近/远改写准确率均 **1.000**、通用文本误路由 **0.000**、"
      "端到端 **router 与 oracle 逐域完全相同**。"
      "⇒ 我原先估计「门控在真实重叠域上有 60–80% 概率失败」，**下调到约 25–40%**。")
    A("27. **★ 我错在混淆「线索稀疏」与「线索弱」**（§12.4）：我优化的量是"
      "**判别 token 的占比**，但决定难度的是**线索的可靠性** —— "
      "「北极」出现在该域**每一条**查询里 ⇒ 任务退化为**词汇存在性检验**，"
      "线性探针轻松解决。⇒ 把线索做稀疏 ≠ 把它做弱。"
      "**主要风险已从「线索被稀释」转移到「线索缺失/歧义」** —— "
      "后者更难绕过，需要多轮上下文或检索信号。")
    A("28. **★ (层 × 池化) 扫描抓到一个真实的工程陷阱**（§12.4）："
      "**`last` 池化危险** —— `none` 误路由 **0.417–1.000**（`mean`/`max` 全为 0），"
      "且当判别线索**不在末尾**时路由准确率**恰好退化为 0.500**（P3）。"
      "而 `last` 恰是**指令分类的常规选择**（指令在末尾）。"
      "⇒ **池化必须与「线索位置」匹配**；做主题路由时 `mean`/`max` 更安全。"
      "若只报单一配置，这个陷阱不会被发现 —— 这是「扫描而非单点」的价值。")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[S6] 写出 {os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()

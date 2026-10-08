"""G3-9 门控的**真正**边界：线索缺失 / 歧义 / 多域规模（`奇点实验报告.md` §12.4 的待办）。

背景（§12.4 的自评，逐字）：
    「主要风险已从『线索被稀释』转移到『**线索缺失/歧义**』 …… 本实验**未测**：
      1. 线索缺失（指代）②概率性/分布式主题线索 ③多域规模 ④真正的主题表述漂移。」

本脚本测 ① 与 ③（②的极端形式 = 线索可靠性 0，一并覆盖）。

★ 核心概念区分（本实验的**主要论点**）：
  `partial` / `generic` 两档在**单轮**里**信息论上不可识别** —— 每个域的问法**逐字相同**。
  因此这两档**不是**「门控失败」，它们问的是**另一个问题**：
      **门控能不能知道自己不知道？**
  于是评价量不是 accuracy，而是三分量：
      `route_acc`（路由到正确域） + `abstain`（弃权） + `misroute`（**自信地答错**）
  一个可部署的系统应当把 misroute 换成 abstain —— 这与 accuracy 无关，是**安全**性质。
  `coref` 档则问：多轮上下文是否**必要且充分**地补回判别信息（full-ctx vs 仅当前轮）。

单轴控制（`topic_domains.py` 内断言，全部通过）：
  域**同构**（同字段、同实体后缀、同问法模板），唯一差别是 series 名（= 唯一线索）。
  ⇒ cue（完整）→ partial（仅后缀，N 域共享）→ generic（无线索）单调递减。
  ⇒ 且**答案全局唯一**（120 条）⇒ 错路由一定产出可检出的错误答案。

阶段
  S1 **多域规模**：N ∈ {2,4,6,8}，门控 retrain，报 acc / 随机基线 1/(N+1) / none 误路由。
  S2 **线索阶梯**（N=6）：cue(阳性对照) / partial / generic / coref×2（full-ctx vs 仅当前轮）。
  S3 **弃权**：margin 阈值（在**训练集**上标定到 5% 弃权）在缺失线索档上的弃权率。
  S4 **端到端**：oracle / router / router+abstain，逐档报 route_acc / abstain / misroute
      + 可验证能力（是否被劫持）。

守卫（沿用本项目纪律）
  * zero 适配器 ΔW 逐位=0，且「激活 zero」与「不加适配器」logits **逐位相同**；
  * 训任一子模块时其他子模块权重**逐位**不变（真隔离）；
  * 容量严格配平（N 个子模块参数量断言相等）；
  * 门控特征取自**基座**（先路由再激活，可部署的因果顺序）；
  * 阳性对照（cue 档必须高）+ 负对照（random 路由）。
"""
from __future__ import annotations

import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model, get_text_backbone  # noqa: E402
from s5_edit import train_edit, render, generate  # noqa: E402
import capability as cap  # noqa: E402
from g3_5_isolation import (lora_mods, set_trainable, snapshot_adapter,  # noqa: E402
                            snapshot_diff, LAYERS, TARGETS, R16, A16,
                            STEPS, BATCH, SEQ, SEED)
from g3_7_router import fit_gate, gate_logits, GATE_LAYER  # noqa: E402
import topic_domains as T  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ZERO = "zero"
NONE = "none"
N_LIST = [2, 4, 6, 8]
N_MAIN = 6
ABSTAIN_Q = 0.05          # 训练集上允许的弃权率（用于标定 margin 阈值）
OUT_JSON = "out/g3_9_missing_cue.json"


# ══════════════════════════════════════════════════════════════════
# 特征：基座第 L 层平均池化
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def layer_means(model, tok, texts, layer=GATE_LAYER, bs=4, max_len=SEQ):
    """单轮文本 → 平均池化隐状态 (n, d)（CPU float32）。"""
    bb = get_text_backbone(model)
    holder = {}

    def hook(m, i, o):
        holder["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    hd = bb.layers[layer].register_forward_hook(hook)
    out = []
    model.eval()
    try:
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            ids, am = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
            model(input_ids=ids, attention_mask=am)
            h = holder["h"].float()
            m = am.unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    finally:
        hd.remove()
    return torch.cat(out, 0)


@torch.no_grad()
def layer_means_multiturn(model, tok, pairs, layer=GATE_LAYER, max_len=SEQ):
    """(prefix, query) 列表 → (full_mean, query_mean)。

    ★ 只保留「当前轮」的特征是**对照组**：用来分离
      「表征收到了上一轮线索（full 能分）」与「模型根本没把上下文带过来」。
    批大小固定 1：需要按前缀 token 数切分，逐条做最不容易出错（仅 ~90 条）。
    """
    bb = get_text_backbone(model)
    holder = {}

    def hook(m, i, o):
        holder["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    hd = bb.layers[layer].register_forward_hook(hook)
    fulls, qs, miss = [], [], 0
    model.eval()
    try:
        for pre, qry in pairs:
            sep = "\n"
            text = pre + sep + qry
            n_pre = len(tok(pre + sep, add_special_tokens=False)["input_ids"])
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=max_len)
            if n_pre >= enc["input_ids"].shape[1]:
                miss += 1                      # 前缀被截断 ⇒ 「当前轮」掩码为空
            model(input_ids=enc["input_ids"].cuda())
            h = holder["h"][0].float()
            n = h.shape[0]
            fulls.append(h.mean(0).cpu())
            qpart = h[min(n_pre, n):]
            if qpart.shape[0] == 0:
                qpart = h[-1:]
            qs.append(qpart.mean(0).cpu())
    finally:
        hd.remove()
    if miss:
        print(f"      [warn] {miss} 条 coref 前缀被截断（query 掩码退化）")
    return torch.stack(fulls, 0), torch.stack(qs, 0)


# ══════════════════════════════════════════════════════════════════
def build_gate(doms, X_of, X_none):
    """在给定域集合上重训一个门控（特征与 none 样本**预先算好**传入，避免重算）。

    ★ 类别配平：把 none 复制到与域样本总数相当，否则 N 增大时 none 类占比会被稀释，
      混淆「N 变大」与「none 类被稀释」两个效应。
    """
    cls = list(doms) + [NONE]
    Xp, y = [], []
    for di, d in enumerate(doms):
        Xp.append(X_of[d])
        y += [di] * X_of[d].shape[0]
    n_dom = sum(X_of[d].shape[0] for d in doms)
    rep = max(1, round(n_dom / X_none.shape[0]))
    Xn = torch.cat([X_none] * rep, 0)
    Xp.append(Xn)
    y += [len(doms)] * Xn.shape[0]
    X = torch.cat(Xp, 0)
    gate = fit_gate(X, y, len(cls))
    acc = float((gate_logits(gate, X).argmax(1) == torch.tensor(y)).float().mean())
    return gate, cls, acc, int(X.shape[0]), rep


def margin_of(logits, n_dom):
    """(n, N+1) → 域内 top1−top2（不含 none 类）。越大越自信。"""
    lg = logits[:, :n_dom]
    top2 = lg.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def tally(pred, true_idx, none_idx):
    """pred: list[int] 类别索引（<none_idx = 域，==none_idx = 弃权）。→ 三分量。

    ★ 弃权哨兵必须是 `none_idx = N`（N+1 个类的末位），**不能**用 -1：
      负数索引会与`最后一个域`混为一谈 ⇒ 把「弃权」统计成「路由正确」。
    """
    n = len(pred)
    a = sum(p == t for p, t in zip(pred, true_idx))
    ab = sum(p == none_idx for p in pred)
    return {"n": n, "route_acc": a / n, "abstain": ab / n,
            "misroute": (n - a - ab) / n}


# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    T.selfcheck(verbose=False)
    model, tok = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-9] 加载 {time.time()-t0:.0f}s")

    doms8 = T.DOM_NAMES
    lm_texts = cap.lm_texts_heldout(48, start=500)
    cap_items = (cap.build_math_items() + cap.build_format_items()
                 + cap.build_code_items())
    cap_prompts = [it["prompt"] for it in cap_items]

    # ── 训练数据（LoRA 用）────────────────────────────────────────
    samples_of = {}
    tr_texts = {}
    for d in doms8:
        s = []
        for q, a in T.train_pairs(d):
            ids = render(tok, q)
            ans = tok(a + tok.eos_token, add_special_tokens=False)["input_ids"]
            s.append((torch.tensor((ids + ans)[:SEQ]),
                      torch.tensor(([-100] * len(ids) + ans)[:SEQ])))
        samples_of[d] = s
        tr_texts[d] = [q for q, _ in T.train_pairs(d)]

    # ── 适配器：8 个子模块 + 一个永不训练的 zero ──────────────────
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=R16, lora_alpha=A16, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=TARGETS,
                     layers_pattern="layers", layers_to_transform=LAYERS)
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    ad_of = {d: f"ad_{d}" for d in doms8}
    model = get_peft_model(model, cfg, adapter_name=ad_of[doms8[0]])
    for d in doms8[1:]:
        model.add_adapter(ad_of[d], cfg)
    model.add_adapter(ZERO, cfg)
    mods = lora_mods(model)
    n_per = sum(mod.lora_A[ZERO].weight.numel() + mod.lora_B[ZERO].weight.numel()
                for mod in mods.values())
    print(f"[G3-9] 模块={len(mods)}  适配器={len(doms8)+1}  每个 {n_per/1e6:.3f}M",
          flush=True)
    # ★ 容量严格配平
    for d in doms8:
        n_d = sum(mod.lora_A[ad_of[d]].weight.numel()
                  + mod.lora_B[ad_of[d]].weight.numel() for mod in mods.values())
        assert n_d == n_per, f"容量不配平：{d} {n_d} ≠ {n_per}"
    for mod in mods.values():
        dm = mod.lora_B[ZERO].weight.float() @ mod.lora_A[ZERO].weight.float()
        assert float(dm.abs().max()) == 0.0, "zero 适配器 ΔW≠0"

    def activate(name):
        model.set_adapter(name, inference_mode=True)

    # ★ 等价性守卫：「激活 zero」== 「禁用适配器」
    with torch.no_grad():
        pr = tok(tr_texts[doms8[0]][0], return_tensors="pt",
                 truncation=True, max_length=64)
        pr = {k: v.cuda() for k, v in pr.items()}
        activate(ZERO)
        lz = model(**pr).logits.float()
        with model.disable_adapter():
            lb = model(**pr).logits.float()
        dev = float((lz - lb).abs().max())
    print(f"[G3-9] 等价性守卫 zero vs disable_adapter: max|Δlogits|={dev:.2e}")
    assert dev == 0.0, "「none」路由语义被破坏"

    # ── 训练 8 个子模块（严格隔离）───────────────────────────────
    for d in doms8:
        a = ad_of[d]
        snaps = {o: snapshot_adapter(model, ad_of[o]) for o in doms8 if o != d}
        n_tr = set_trainable(model, a)
        model.set_adapter(a)
        model.train()
        losses, ts = train_edit(model, samples_of[d], STEPS, BATCH, 1e-4,
                                seed=SEED, log_every=0, tag=f"[{a}] ")
        model.eval()
        diffs = {o: snapshot_diff(snaps[o], model, ad_of[o]) for o in snaps}
        assert all(v == 0.0 for v in diffs.values()), f"隔离失效：{diffs}"
        print(f"[G3-9] 训 {a:<12} {ts:5.0f}s loss={losses[-1]:.4f} "
              f"可训练={n_tr/1e6:.2f}M 其他子模块逐位偏差=0.0", flush=True)
    set_trainable(model, ZERO, False)

    # ── 探针集 ────────────────────────────────────────────────────
    probes = {}
    for style in ("cue", "partial", "generic"):
        probes[style] = {d: T.eval_items(d, style) for d in doms8}
    probes["coref"] = {d: T.coref_items(d) for d in doms8}

    # 特征缓存（单轮）
    activate(ZERO)      # ★ 特征一律取自基座
    print("\n[G3-9] 抽取特征（基座 L%d 平均池化）…" % GATE_LAYER)
    F = {}
    for style in ("cue", "partial", "generic"):
        for d in doms8:
            F[(style, d)] = layer_means(model, tok,
                                        [it["prompt"] for it in probes[style][d]])
    F[("none", "cap")] = layer_means(model, tok, cap_prompts)
    F[("none", "lm")] = layer_means(model, tok, lm_texts[:24])
    F[("train", "none")] = F[("none", "cap")]      # 同一批（none 类的训练样本）
    for d in doms8:
        F[("train", d)] = layer_means(model, tok, tr_texts[d])
    # coref：full-ctx 与 仅当前轮
    CF = {"full": {}, "query": {}}
    for d in doms8:
        pairs = [(it["prefix"], it["prompt"]) for it in probes["coref"][d]]
        fu, qu = layer_means_multiturn(model, tok, pairs)
        CF["full"][d], CF["query"][d] = fu, qu
    print(f"[G3-9] 特征就绪 {time.time()-t0:.0f}s")

    report = {"domains": doms8, "n_facts": T.N_FACTS, "stages": {}}

    # ══ S1 多域规模 ═══════════════════════════════════════════════
    print("\n[G3-9] ══ S1 多域规模（门控 retrain）══")
    s1 = []
    gates = {}
    for N in N_LIST:
        doms = doms8[:N]
        g, cls, acc_tr, nX, rep = build_gate(
            doms, {d: F[("train", d)] for d in doms}, F[("train", "none")])
        gates[N] = (g, cls, doms)
        pred, true = [], []
        for di, d in enumerate(doms):
            lg = gate_logits(g, F[("cue", d)])
            pred += lg.argmax(1).tolist()
            true += [di] * lg.shape[0]
        # none 误路由：通用文本（没见过的段落）
        p_lm = gate_logits(g, F[("none", "lm")]).argmax(1).tolist()
        fpr_lm = sum(p != len(doms) for p in p_lm) / len(p_lm)
        # none 误路由：能力基准提示（none 类的**训练**分布）
        p_cap = gate_logits(g, F[("none", "cap")]).argmax(1).tolist()
        fpr_cap = sum(p != len(doms) for p in p_cap) / len(p_cap)
        t = tally(pred, true, len(doms))
        rec = {"N": N, "n_train": nX, "gate_train_acc": acc_tr,
               "chance": 1.0 / (N + 1), "cue_acc": t["route_acc"],
               "abstain": t["abstain"], "misroute": t["misroute"],
               "none_fpr_lm": fpr_lm, "none_fpr_cap": fpr_cap}
        s1.append(rec)
        print(f"   N={N}  trainacc={acc_tr:.3f}  cue_acc={t['route_acc']:.3f}"
              f"  (随机 1/(N+1)={1/(N+1):.3f}, 相对={t['route_acc']* (N+1):.2f}×)"
              f"  none误路由 lm={fpr_lm:.3f} cap={fpr_cap:.3f}")
    report["stages"]["S1_scale"] = s1

    # ══ S2 线索阶梯（N = N_MAIN）══════════════════════════════════
    print(f"\n[G3-9] ══ S2 线索阶梯（N={N_MAIN}，用 N={N_MAIN} 的 cue 门控）══")
    doms = doms8[:N_MAIN]
    g6, cls6, doms6, = gates[N_MAIN]
    nC = len(doms)

    def eval_style(style, feats=None, tag=None):
        pred, true = [], []
        for di, d in enumerate(doms):
            X = feats[d] if feats is not None else F[(style, d)]
            lg = gate_logits(g6, X)
            pred += lg.argmax(1).tolist()
            true += [di] * lg.shape[0]
        t = tally(pred, true, nC)
        t["chance"] = 1.0 / (nC + 1)
        if tag:
            t["variant"] = tag
        return t

    s2 = {"cue": eval_style("cue"), "partial": eval_style("partial"),
          "generic": eval_style("generic"),
          "coref_full": eval_style("coref", CF["full"], "coref_full"),
          "coref_query_only": eval_style("coref", CF["query"], "coref_query_only")}
    for k, v in s2.items():
        print(f"   {k:<17} route_acc={v['route_acc']:.3f}  "
              f"abstain={v['abstain']:.3f}  misroute={v['misroute']:.3f}"
              f"  (随机 {v['chance']:.3f})")
    # ★ coref：上下文是否**必要且充分**？
    print(f"   ★ coref 上下文增益：full {s2['coref_full']['route_acc']:.3f} vs "
          f"仅当前轮 {s2['coref_query_only']['route_acc']:.3f} "
          f"⇒ Δ={s2['coref_full']['route_acc']-s2['coref_query_only']['route_acc']:+.3f}")
    report["stages"]["S2_ladder"] = s2

    # ══ S3 margin 诊断 + 校准来源扫描 ═════════════════════════════
    #  ★★ 这里**推翻了我自己的预注册预期**。原注释写的是「cue 与 partial/generic 的
    #  margin 几乎重合 ⇒ 该量里没有信息」，实测**完全相反**：
    #     cue      margin ∈ [0.576, 1.035]（中位 0.80）
    #     partial  margin ∈ [0.0003, 0.110]（中位 0.027）
    #     generic  margin ∈ [0.015, 0.046]（中位 0.018）
    #  ⇒ 区间**不相交** ⇒ margin 是「线索在不在」的**近乎完美判别量**。
    #  真正的问题在**校准集的分布**：训练用的是**规范问法**，其 margin 饱和在
    #  0.998~1.001，而**留出改写**的 margin 只有 0.58~1.04 ⇒ 从训练集取分位数会把
    #  τ 放到 0.9985（比可用点高 5 倍以上）⇒ 连 cue 都被弃权 88.9%。
    #  ⇒ 结论不是「阈值法失效」，而是「**校准集必须与部署分布同族**」。
    print(f"\n[G3-9] ══ S3 margin 诊断 + 校准来源扫描 ══")
    m_tr = torch.cat([margin_of(gate_logits(g6, F[("train", d)]), nC)
                      for d in doms])
    tau_train = float(torch.quantile(m_tr, ABSTAIN_Q))
    print(f"   训练集（规范问法）margin：中位 {float(m_tr.median()):.4f} "
          f"{ABSTAIN_Q:.0%} 分位 τ={tau_train:.4f} "
          f"[{float(m_tr.min()):.4f},{float(m_tr.max()):.4f}]  ← **饱和**")

    # 收集各档逐项 margin（后续所有分析都在这些量上做，不再前向）
    MG = {}
    for style, feats in (("cue", None), ("partial", None), ("generic", None),
                         ("coref", CF["full"])):
        X = torch.cat([feats[d] if feats is not None else F[(style, d)]
                       for d in doms], 0)
        MG[style] = margin_of(gate_logits(g6, X), nC)

    s3 = {"tau_train_quantile": tau_train,
          "train_margin_median": float(m_tr.median()),
          "train_margin_min": float(m_tr.min()),
          "margin_stats": {}}
    for style, mg in MG.items():
        s3["margin_stats"][style] = {
            "median": float(mg.median()), "min": float(mg.min()),
            "max": float(mg.max()), "n": int(mg.numel())}
        print(f"   {style:<9} margin 中位={float(mg.median()):.4f} "
              f"[{float(mg.min()):.4f},{float(mg.max()):.4f}]")

    # ── ★ AUC：margin 对「有线索 vs 无线索」的可分性（完美判别 ⇒ 1.000）──
    pos = MG["cue"]
    neg = torch.cat([MG["partial"], MG["generic"]], 0)
    wins = (pos.unsqueeze(1) > neg.unsqueeze(0)).float().mean()
    auc = float(wins)
    s3["auc_cue_vs_cueless"] = auc
    n_pos, n_neg = int(pos.numel()), int(neg.numel())
    print(f"   ★ AUC（cue 正 / partial+generic 负）= {auc:.4f} "
          f"(n={n_pos}×{n_neg})  重叠区间="
          f"{'无' if float(pos.min()) > float(neg.max()) else '有'}"
          f"（cue.min={float(pos.min()):.4f} vs 无线索.max={float(neg.max()):.4f}）")

    # ── 校准来源扫描：τ 从哪来？（同一批 margin，只换 τ 的来源）──
    #  ① `train_quantile`：规范问法 margin 的分位数（本项目现用）——**错族**
    #  ② `rewrite_val`   ：**留出改写** margin 的分位数（= 与部署同族的验证集）
    #  ③ `interval_mid`  ：在「有线索引 vs 无线索引」之间取中点（上界参考）
    m_rewrite = torch.cat([MG["cue"], MG["coref"]], 0)   # 都有线索 ⇒ 不该弃权
    tau_val = float(torch.quantile(m_rewrite, ABSTAIN_Q))
    lo, hi = float(neg.max()), float(pos.min())
    tau_mid = (lo + hi) / 2 if hi > lo else float("nan")
    s3["tau_rewrite_quantile"] = tau_val
    s3["valid_interval"] = [lo, hi]
    print(f"   可用区间（无线索 max {lo:.4f} < τ < cue min {hi:.4f}）"
          f"⇒ 非空：{hi > lo}，宽 {hi-lo:.4f}")
    print(f"   τ 来源三种：train_quantile={tau_train:.4f}  "
          f"rewrite_quantile={tau_val:.4f}  interval_mid={tau_mid:.4f}")

    def abstain_table(tau):
        out = {}
        for style in ("cue", "partial", "generic", "coref"):
            out[style] = float((MG[style] < tau).float().mean())
        return out

    s3["by_tau"] = {}
    for nm, tau in (("train_quantile", tau_train),
                    ("rewrite_quantile", tau_val),
                    ("interval_mid", tau_mid)):
        t = abstain_table(tau)
        s3["by_tau"][nm] = {"tau": tau, **t}
        print(f"   τ={nm:<16}({tau:.4f})  弃权率  "
              f"cue={t['cue']:.3f} coref={t['coref']:.3f} "
              f"partial={t['partial']:.3f} generic={t['generic']:.3f}")

    # ── ★ 可行区间扫描：τ 网格上「该弃权的全弃权、不该弃权的零弃权」能否同时成立 ──
    lo_all, hi_all = float(min(m.min() for m in MG.values())), \
        float(max(m.max() for m in MG.values()))
    grid = torch.linspace(lo_all, hi_all, 401)
    ok = []
    curve = []
    for tv in grid:
        f = abstain_table(float(tv))
        good = (f["partial"] == 1.0 and f["generic"] == 1.0
                and f["cue"] == 0.0 and f["coref"] == 0.0)
        if good:
            ok.append(float(tv))
        curve.append([float(tv), f["cue"], f["coref"], f["partial"], f["generic"]])
    s3["operating_tau_range"] = [min(ok), max(ok)] if ok else None
    if ok:
        print(f"   ★★ **可行工作区间**：τ ∈ [{min(ok):.4f}, {max(ok):.4f}] "
              f"上「cue/coref 零弃权 且 partial/generic 全弃权」**同时成立**"
              f"（宽 {max(ok)-min(ok):.4f}）")
    else:
        print("   ★★ 未找到四个条件同时成立的工作点 ⇒ 该读出**做不到**完美弃权")
    s3["abstain_curve_sample"] = curve[::20]
    report["stages"]["S3_margin_calib"] = s3


    # ══ S4 端到端 ═════════════════════════════════════════════════
    print("\n[G3-9] ══ S4 端到端（生成 + 精确匹配）══")
    gen_cache = {}

    def gen_cached(prompt, adapter, mt=24):
        key = (prompt, adapter)
        if key not in gen_cache:
            activate(adapter)
            gen_cache[key] = generate(model, tok, prompt, max_new_tokens=mt)
        return gen_cache[key]

    def dec_of(style, d, feats=None, tau=None):
        X = feats[d] if feats is not None else F[(style, d)]
        lg = gate_logits(g6, X)
        am = lg.argmax(1).tolist()
        mg = margin_of(lg, nC).tolist()
        out = []
        for i, j in enumerate(am):
            if tau is not None and float(mg[i]) < tau:
                out.append(ZERO)                 # 弃权 → 走基座
            else:
                out.append(ad_of[doms[j]] if j < nC else ZERO)
        return out

    s4 = {}
    # ★ 弃权阈值两种来源都端到端跑一遍：`val`（同族校准）vs `train`（错族校准）
    for style, feats in (("cue", None), ("generic", None), ("partial", None),
                         ("coref", CF["full"])):
        items_by_d = probes[style]
        for arm, kw in (("oracle", {}), ("router", {}),
                        ("router_abstain_val", {"tau": tau_val}),
                        ("router_abstain_train", {"tau": tau_train})):
            t = {"n": 0, "ok": 0}
            for di, d in enumerate(doms):
                if arm == "oracle":
                    dec = [ad_of[d]] * len(items_by_d[d])
                else:
                    dec = dec_of(style, d, feats, **kw)
                for it, ad in zip(items_by_d[d], dec):
                    txt = gen_cached(it["prompt"], ad)
                    t["n"] += 1
                    t["ok"] += int(cap.grade_exact(txt, it["answer"]))
                print(f"      · {style}/{arm} 域 {di+1}/{len(doms)} "
                      f"缓存={len(gen_cache)} ok={t['ok']}/{t['n']}", flush=True)
            t["acc"] = t["ok"] / t["n"]
            s4[f"{style}::{arm}"] = t
            print(f"   {style:<8} {arm:<21} acc={t['acc']:.3f}  "
                  f"({t['ok']}/{t['n']})", flush=True)
    report["stages"]["S4_e2e"] = s4

    # ══ S5 弃权：**显式训练**「弃权类」能否让门控知道自己不知道？══
    #  ★ S2 已证：用能力基准提示训练的 none 类**不识别**无线索的域问句（0 弃权、
    #    0.833 自信错路由）。S5 问：把无线索问句**显式加入** none 类能否修好，
    #    以及能否迁移到**没见过的**无线索措辞（T_c 只训练、T_a/T_b 只测试）。
    print(f"\n[G3-9] ══ S5 弃权类构成（cue 路由 vs 无线索弃权 的权衡）══", flush=True)
    Fcl = {k: layer_means(model, tok, T.cueless_prompts([k])) for k in ("T_a", "T_b", "T_c")}
    for k, v in Fcl.items():
        print(f"   {k}: {v.shape[0]} 条  e.g. {T.cueless_prompts([k])[0]}")
    X_cap = F[("train", "none")]
    arms = {
        "none_cap":        X_cap,
        "none_cap+T_c":    torch.cat([X_cap, Fcl["T_c"]], 0),
        "none_cap+Ta+Tb+Tc": torch.cat([X_cap, Fcl["T_a"], Fcl["T_b"], Fcl["T_c"]], 0),
    }
    s5 = {}
    for aname, Xn in arms.items():
        g, cls, acc_tr, nX, rep = build_gate(
            doms, {d: F[("train", d)] for d in doms}, Xn)
        row = {"n_train": nX, "none_rep": rep, "gate_train_acc": acc_tr}
        # (a) cue 档必须仍然路由正确（弃权不该伤正例）
        pred, true = [], []
        for di, d in enumerate(doms):
            lg = gate_logits(g, F[("cue", d)])
            pred += lg.argmax(1).tolist()
            true += [di] * lg.shape[0]
        row["cue"] = tally(pred, true, nC)
        # (b) 无线索档：弃权率是关键（不是 accuracy）
        for style in ("generic", "partial"):
            pred, true = [], []
            for di, d in enumerate(doms):
                lg = gate_logits(g, F[(style, d)])
                pred += lg.argmax(1).tolist()
                true += [di] * lg.shape[0]
            row[style] = tally(pred, true, nC)
        # (c) 通用文本上不该乱弃权？（弃权本身无害，但要报出来）
        lg = gate_logits(g, F[("none", "lm")]).argmax(1).tolist()
        row["lm_abstain"] = sum(p == nC for p in lg) / len(lg)
        s5[aname] = row
        print(f"   {aname:<20} trainacc={acc_tr:.3f} | "
              f"cue 正确={row['cue']['route_acc']:.3f} | "
              f"generic 弃权={row['generic']['abstain']:.3f} "
              f"错路由={row['generic']['misroute']:.3f} | "
              f"partial 弃权={row['partial']['abstain']:.3f} "
              f"错路由={row['partial']['misroute']:.3f} | "
              f"lm 弃权={row['lm_abstain']:.3f}")
    print("   ★ 判读：`none_cap+T_c`（弃权类只见过**没被测试过**的措辞 T_c）"
          "在 T_a/T_b 上的弃权率 = 迁移能力。")
    report["stages"]["S5_abstain_class"] = s5

    # ── 能力是否被劫持：S5 最优门的逐项路由（能力提示应被弃权）──
    g_best, _, _, _, _ = build_gate(doms, {d: F[("train", d)] for d in doms},
                                    arms["none_cap+Ta+Tb+Tc"])
    per = []
    for it, j in zip(cap_items,
                     gate_logits(g_best, F[("none", "cap")]).argmax(1).tolist()):
        ad = ZERO if j >= nC else ad_of[doms[j]]
        txt = gen_cached(it["prompt"], ad, mt=40)
        if it["check"] == "unittest":
            ok = cap.run_unittest(cap.extract_code(txt), it["tests"])[0]
        else:
            ok = cap.grade_exact(txt, it["answer"])
        per.append({"id": it["id"], "kind": it["kind"], "ok": bool(ok)})
    cap_res = cap.summarize(per)
    activate(ZERO)
    cap_res["lm"] = cap.eval_lm(model, tok, lm_texts)
    print(f"   能力（S5 最优门逐项路由）：verifiable={cap_res['verifiable']['acc']:.3f} "
          f"LM CE={cap_res['lm']['ce']:.4f}")
    report["stages"]["capability"] = cap_res

    report["elapsed_s"] = time.time() - t0
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[G3-9] 完成 {report['elapsed_s']:.0f}s → {OUT_JSON}")


if __name__ == "__main__":
    main()

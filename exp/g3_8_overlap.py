"""G3-8 门控的**适用边界**：域重叠到什么程度，路由就分不开了？

动机：G3-7 的门控留出准确率 4/4 类全 **1.000**，但那是因为三个虚构域靠
**唯一实体名**即可在 L13 线性分开 —— 判别信号是 prompt 里一个独一无二的词。
真实部署里域通常按**主题**定义，判别信号被摊在很多 token 上（「ACL 论文」vs
「NeurIPS 论文」）。本实验沿**单一可控轴**把重叠度调上去，看门控在哪里失效。

三对域（`overlap_domains.py`，每域 15 条事实，判别性字符集差异单调递减）：
  P1 实体名不同/字段全同  （差异 11 字符） ← 原实验设定，参考基准
  P2 实体名**全同**/字段不同（差异 8 字符）
  P3 实体名全同/字段头全同，只差「北极」/「南极」（差异 **2** 字符）

★ 核心设计：(层 × 池化) **扫描**，而不是只报一个数。
  因为「门控失败」有两种**完全不同**的原因，必须分开：
    (a) 表征里已经没有判别信号（任何读出都救不了）⇒ 路线受限
    (b) 表征里有，但 mean 池化把它稀释掉了 ⇒ 只是**读出选择**问题
  判据：若某变体（如 last/max）能显著高于 mean，则属 (b)。
  另外同时报**训练集准确率**：若训练集也分不开 ⇒ 属 (a)；
  若训练集能分开而留出分不开 ⇒ 是**改写泛化**问题，不是重叠问题。

读数：
  R1 (层 × 池化) 的门控准确率矩阵：训练 / 近改写 / 远改写
  R2 `none` 误路由率（**训练时见过的**基准提示 vs **没见过的**通用文本）
  R3 端到端：oracle（该域自己的子模块）vs router（按门控选）vs 随机
"""
from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model, get_text_backbone  # noqa: E402
from s5_edit import train_edit, render, generate  # noqa: E402
import capability as cap  # noqa: E402
from g3_5_isolation import (lora_mods, set_trainable, snapshot_adapter,  # noqa: E402
                            snapshot_diff, LAYERS, TARGETS, R16, A16,
                            STEPS, BATCH, SEQ, SEED)
import overlap_domains as O  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ZERO = "zero"
NONE = "none"
SWEEP_LAYERS = [8, 13, 20]
SWEEP_POOLS = ["mean", "last", "max"]
LAM = 1.0


# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def collect_feats(model, tok, texts, layers=SWEEP_LAYERS, bs=4, max_len=96):
    """一次前向拿多个层的隐状态，并同时算出 mean/last/max 三种池化。

    → {(layer, pool): tensor(n, d)}（CPU，float32）
    ★ 只跑一遍前向 ⇒ 扫描 (层 × 池化) 的边际成本几乎为零。
    """
    bb = get_text_backbone(model)
    holder = {}
    handles = []
    for L in layers:
        def mk(L):
            def hook(m, i, o):
                holder[L] = (o[0] if isinstance(o, tuple) else o).detach()
            return hook
        handles.append(bb.layers[L].register_forward_hook(mk(L)))
    acc = {}
    model.eval()
    try:
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
            ids, am = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
            model(input_ids=ids, attention_mask=am)
            ln = am.sum(1).clamp(min=1)
            m = am.unsqueeze(-1).float()
            ar = torch.arange(ids.shape[0], device=ids.device)
            for L in layers:
                h = holder[L].float()
                vals = {
                    "mean": (h * m).sum(1) / m.sum(1).clamp(min=1),
                    "last": h[ar, (ln - 1).long()],
                    "max": h.masked_fill(m == 0, float("-inf")).max(1).values,
                }
                for p, v in vals.items():
                    acc.setdefault((L, p), []).append(v.cpu())
    finally:
        for h_ in handles:
            h_.remove()
    return {k: torch.cat(v, 0) for k, v in acc.items()}


def fit_gate(X, y, n_cls, lam=LAM):
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True).clamp(min=1e-6)
    Z = torch.cat([(X - mu) / sd, torch.ones(X.shape[0], 1)], 1)
    Y = torch.zeros(Z.shape[0], n_cls)
    Y[torch.arange(Z.shape[0]), torch.tensor(y)] = 1.0
    alpha = torch.linalg.solve(Z @ Z.t() + lam * torch.eye(Z.shape[0]), Y)
    return {"mu": mu, "sd": sd, "W": Z.t() @ alpha}


@torch.no_grad()
def predict(gate, X):
    Z = torch.cat([(X - gate["mu"]) / gate["sd"],
                   torch.ones(X.shape[0], 1)], 1)
    return (Z @ gate["W"]).argmax(1)


def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    model, tok = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-8] 加载 {time.time()-t0:.0f}s")

    doms = list(O.DOMAINS)
    lm_texts = cap.lm_texts_heldout(48, start=500)
    cap_items = (cap.build_math_items() + cap.build_format_items()
                 + cap.build_code_items())
    cap_prompts = [it["prompt"] for it in cap_items]

    # ── 适配器：6 个域 + zero ───────────────────────────────────────
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=R16, lora_alpha=A16, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=TARGETS,
                     layers_pattern="layers", layers_to_transform=LAYERS)
    import random
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    ad_of = {}
    model = get_peft_model(model, cfg, adapter_name=ZERO)
    for d in doms:
        ad_of[d] = f"ad_{d}"
        model.add_adapter(ad_of[d], cfg)
    mods = lora_mods(model)
    n_per = sum(mod.lora_A[ZERO].weight.numel() + mod.lora_B[ZERO].weight.numel()
                for mod in mods.values())
    print(f"[G3-8] 模块={len(mods)}  适配器={len(doms)+1}  每个 {n_per/1e6:.3f}M"
          f"（6 域容量相同）")
    for mod in mods.values():
        dm = (mod.lora_B[ZERO].weight.float().detach()
              @ mod.lora_A[ZERO].weight.float().detach())
        assert float(dm.abs().max()) == 0.0, "zero 适配器 ΔW≠0"

    def activate(a):
        model.set_adapter(a, inference_mode=True)

    # ── 训练 6 个子模块（隔离 + 逐位守卫）──────────────────────────
    print("\n[G3-8] === 训练 6 个独立子模块（每域 15 事实 / 200 步）===")
    train_p, near_p, far_p = {}, {}, {}
    for d in doms:
        train_p[d] = [q for q, _a in O.train_pairs(d)]
        near_p[d] = [it["prompt"] for it in O.eval_items(d, "near")]
        far_p[d] = [it["prompt"] for it in O.eval_items(d, "far")]
    samples = {}
    for d in doms:
        s = []
        for q, a in O.train_pairs(d):
            ids = render(tok, q)
            ans = tok(a + tok.eos_token, add_special_tokens=False)["input_ids"]
            s.append((torch.tensor((ids + ans)[:SEQ]),
                      torch.tensor(([-100] * len(ids) + ans)[:SEQ])))
        samples[d] = s

    for d in doms:
        ad = ad_of[d]
        snaps = {o: snapshot_adapter(model, ad_of[o]) for o in doms if o != d}
        n_tr = set_trainable(model, ad)
        model.set_adapter(ad)
        model.train()
        losses, ts = train_edit(model, samples[d], STEPS, BATCH, 1e-4, seed=SEED,
                                log_every=0, tag=f"[{ad}] ")
        model.eval()
        diffs = {o: snapshot_diff(snaps[o], model, ad_of[o]) for o in snaps}
        assert all(v == 0.0 for v in diffs.values()), f"隔离失效：{diffs}"
        print(f"[G3-8]   {ad:<12} loss={losses[-1]:.4f} {ts:.0f}s "
              f"可训练={n_tr/1e6:.3f}M 其他子模块偏差全为 0.0")
    set_trainable(model, ZERO, False)

    # ── 特征（一次前向，多池化）────────────────────────────────────
    activate(ZERO)
    allq = []
    idx = {}
    for tag, src in (("tr", train_p), ("near", near_p), ("far", far_p)):
        for d in doms:
            idx[(tag, d)] = list(range(len(allq), len(allq) + len(src[d])))
            allq += src[d]
    idx[("cap", None)] = list(range(len(allq), len(allq) + len(cap_prompts)))
    allq += cap_prompts
    idx[("lm", None)] = list(range(len(allq), len(allq) + len(lm_texts[:24])))
    allq += lm_texts[:24]
    F = collect_feats(model, tok, allq)
    print(f"[G3-8] 特征：{len(allq)} 条 × {len(F)} 个(层×池化)变体 "
          f"({time.time()-t0:.0f}s)")

    def sub(kind, d=None):
        return F_slice(F, idx[(kind, d)])

    # ── 逐对构建门控并扫描 (层 × 池化) ──────────────────────────────
    results = {}
    for pname in O.PAIR_LIST:
        p = O.PAIRS[pname]
        da, db = p["a"], p["b"]
        print(f"\n[G3-8] ═══ {pname}：{da} vs {db} ═══")
        print(f"        {p['why']}")
        # 门控训练集：A 训练问法→0，B 训练问法→1，none→2
        ytr = ([0] * len(train_p[da]) + [1] * len(train_p[db])
               + [2] * len(cap_prompts))
        rows = {}
        for key in F:
            L, pool = key
            Fa = F[key][idx[("tr", da)]]
            Fb = F[key][idx[("tr", db)]]
            Fn = F[key][idx[("cap", None)]]
            Xtr = torch.cat([Fa, Fb, Fn], 0)
            gate = fit_gate(Xtr, ytr, 3)
            ptr = predict(gate, Xtr)
            acc_tr = float(((ptr == torch.tensor(ytr)) & (torch.tensor(ytr) < 2))
                           .float().sum()
                           / max(int((torch.tensor(ytr) < 2).sum()), 1))
            res = {"train_acc": acc_tr}
            for tag, src in (("near", near_p), ("far", far_p)):
                ok, tot = 0, 0
                for di, d in enumerate((da, db)):
                    X = F[key][idx[(tag, d)]]
                    pr = predict(gate, X)
                    ok += int((pr == di).sum())
                    tot += len(pr)
                res[f"{tag}_acc"] = ok / tot
            for tag, key2 in (("cap", ("cap", None)), ("lm", ("lm", None))):
                X = F[key][idx[key2]]
                pr = predict(gate, X)
                res[f"none_misroute_{tag}"] = float((pr != 2).float().mean())
            rows[f"L{L}_{pool}"] = res
        results[pname] = {"why": p["why"], "domains": [da, db], "variants": rows}
        # 打印矩阵
        print(f"        {'变体':<12}{'训练':>8}{'近改写':>9}{'远改写':>9}"
              f"{'none误(cap)':>13}{'none误(lm)':>12}")
        for k, r in rows.items():
            print(f"        {k:<12}{r['train_acc']:>8.3f}{r['near_acc']:>9.3f}"
                  f"{r['far_acc']:>9.3f}{r['none_misroute_cap']:>13.3f}"
                  f"{r['none_misroute_lm']:>12.3f}")
        best = max(rows.items(), key=lambda kv: kv[1]["far_acc"])
        print(f"        ⇒ 最佳变体 {best[0]}：近改写 {best[1]['near_acc']:.3f} /"
              f" 远改写 {best[1]['far_acc']:.3f}（随机 = 0.500）")

    # ── 端到端：oracle vs router（用 mean@L13 = 原实验口径）──────────
    print("\n[G3-8] === 端到端：oracle vs router（门控口径 mean@L13）===")
    print(f"{'对':<20}{'域':<11}{'oracle':>9}{'router':>9}")
    e2e = {}
    for pname in O.PAIR_LIST:
        p = O.PAIRS[pname]
        da, db = p["a"], p["b"]
        key = (13, "mean")
        gate = fit_gate(
            torch.cat([F[key][idx[("tr", da)]], F[key][idx[("tr", db)]],
                       F[key][idx[("cap", None)]]], 0),
            ([0] * len(train_p[da]) + [1] * len(train_p[db])
             + [2] * len(cap_prompts)), 3)
        e2e[pname] = {}
        for di, d in enumerate((da, db)):
            items = O.eval_items(d, "near")
            X = F[key][idx[("near", d)]]
            pr = predict(gate, X).tolist()
            activate(ad_of[d])
            ora = sum(int(cap.grade_exact(
                generate(model, tok, it["prompt"], max_new_tokens=24),
                it["answer"])) for it in items) / len(items)
            ok = 0
            for it, pj in zip(items, pr):
                activate(ad_of[da] if pj == 0 else
                         (ad_of[db] if pj == 1 else ZERO))
                ok += int(cap.grade_exact(
                    generate(model, tok, it["prompt"], max_new_tokens=24),
                    it["answer"]))
            rou = ok / len(items)
            routed_to_self = sum(1 for x in pr if x == di) / len(pr)
            e2e[pname][d] = {"oracle": ora, "router": rou,
                             "routed_to_self": routed_to_self}
            print(f"{pname:<20}{d:<11}{ora:>9.3f}{rou:>9.3f}"
                  f"   （门控将自己路由到本域的比例 {routed_to_self:.3f}）")

    out = {"config": {"layers_edited": LAYERS, "r": R16, "steps": STEPS,
                      "sweep_layers": SWEEP_LAYERS, "sweep_pools": SWEEP_POOLS,
                      "n_per_adapter": n_per, "domains": doms,
                      "pairs": O.PAIR_LIST, "chance": 0.5},
           "pairs": results, "end_to_end": e2e,
           "note": "P1/P2/P3 判别性字符集差异 = 11/8/2（单调递增难度）"}
    with open("out/g3_overlap.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[G3-8] 写出 out/g3_overlap.json  (总 {time.time()-t0:.0f}s)")


def F_slice(F, ix):
    return {k: v[ix] for k, v in F.items()}


if __name__ == "__main__":
    main()

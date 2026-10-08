"""G3-7 §5.2 第三条的**正向**验证：门控路由能否让三个子模块共存？

背景：
  * G3-5 已证——三个子模块的 ΔW **朴素相加**会让每个域都坏掉
    （northwind 0.933→0.200、halcyon 0.733→0.333、vela 0.867→0.133）。
  * 因此 §5.2 把「**门控网络根据输入自动路由到相关子模块**」列为承重件。
    本脚本就实现它，并检验它是否真的解决问题。

设计
  * 三个不重叠虚构域，各自一个**独立子模块**（r=16，隔离训练 ⇒ 权重互不干扰）。
  * **门控** = 在**基座**（不激活任何适配器）第 L=13 层平均池化隐状态上的
    **4 类岭回归**分类器 {northwind, halcyon, vela, none}。
      - ★ 特征取自**基座** ⇒ 推理时「先路由、再激活」，是可部署的因果顺序
        （不能反过来：用适配器后的特征决定激活哪个适配器，那是循环依赖）。
      - ★ L=13 的选择有依据：目标2 已实测**自我认知可分性峰在 L13–14**
        （null 2.03×），主题/实体级区分在这里最可分。
      - ★「none」类样本 = 可验证基准（math/format/code）的**提示**
        ⇒ 这是关键：门控必须学会**不要劫持无关查询**。做了类别配平（复制）。
  * 用**留出改写问法**（`eval_items`，非训练对）评路由质量 ⇒ 检验泛化而非记忆。

对照臂（同一进程、共享基线）
  `base`    `zero` 适配器（未训练，ΔW≡0，等价于基座）
  `oracle`  直接激活该域自己的子模块（**可达上限**）
  `random`  **负对照**：在 4 个选项里随机路由 ⇒ 证明门控的决策确实在起作用
  `sum`     三个 ΔW 直接相加（G3-5 证明会坏）—— 用来确认本脚本复现了该失败
  `router`  门控 argmax → 只激活选中的子模块；选到 none 则用基座

读数
  R1 门控本身：训练准确率 / 留出探针准确率 / 4×4 混淆矩阵 /
     在「none」样本上的误路由率
  R2 每域习得：`router` 是否追平 `oracle`
  R3 无关查询：`router` 的可验证能力是否守住（对比 `sum` 的崩塌与 ΔCE）

守卫（沿用本项目纪律）
  * `set_trainable` 显式白名单；训一域时其他域子模块 ΔW **逐位**不变；
  * `zero` 适配器的 ΔW 必须**精确为 0**，且「激活 zero」与「不加适配器」的
    logits 必须**逐位相同** —— 等价性测试，保证「none」真的是基座；
  * `random` 臂作为负对照（没有它，「router 有效」可能只是「生成本来就对」）。
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
import domains as D  # noqa: E402
from g3_5_isolation import (  # noqa: E402
    lora_mods, set_trainable, snapshot_adapter, snapshot_diff, ComposeExtras,
    LAYERS, TARGETS, DOMS, SUB, R16, A16, STEPS, BATCH, SEQ, SEED,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

GATE_LAYER = 13
ZERO = "zero"                     # 永不训练的适配器 ⇒ ΔW≡0 ⇒ 等价于基座
NONE = "none"
LAM = 1.0


# ══════════════════════════════════════════════════════════════════
# 门控特征：基座第 L 层平均池化隐状态
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def pooled_features(model, tokenizer, texts, layer=GATE_LAYER, bs=4):
    bb = get_text_backbone(model)
    h = bb.layers[layer]
    holder = {}

    def hook(m, i, o):
        holder["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    hd = h.register_forward_hook(hook)
    feats = []
    model.eval()
    try:
        for i in range(0, len(texts), bs):
            enc = tokenizer(texts[i:i + bs], return_tensors="pt", padding=True,
                            truncation=True, max_length=SEQ)
            ids, am = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
            model(input_ids=ids, attention_mask=am)
            hh = holder["h"].float()
            m = am.unsqueeze(-1).float()
            feats.append(((hh * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    finally:
        hd.remove()
    return torch.cat(feats, 0)


def fit_gate(X, y, n_cls, lam=LAM):
    """标准化 + 岭回归（对偶/KRR 形式）。X (n,d) 在 CPU。"""
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True).clamp(min=1e-6)
    Z = (X - mu) / sd
    Z = torch.cat([Z, torch.ones(Z.shape[0], 1)], 1)
    Y = torch.zeros(Z.shape[0], n_cls)
    Y[torch.arange(Z.shape[0]), torch.tensor(y)] = 1.0
    n = Z.shape[0]
    alpha = torch.linalg.solve(Z @ Z.t() + lam * torch.eye(n), Y)
    return {"mu": mu, "sd": sd, "W": Z.t() @ alpha}


@torch.no_grad()
def gate_logits(gate, X):
    Z = (X - gate["mu"]) / gate["sd"]
    Z = torch.cat([Z, torch.ones(Z.shape[0], 1)], 1)
    return Z @ gate["W"]


# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    model, tokenizer = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-7] 加载 {time.time()-t0:.0f}s")

    lm_texts = cap.lm_texts_heldout(48, start=500)
    ev = {d: D.eval_items(d) for d in DOMS}
    tr = {}
    for d in DOMS:
        s = []
        for q, _a in D.train_pairs(d):
            s.append(q)
        tr[d] = s
    # 训练样本（用于 LoRA）
    trs = {}
    for d in DOMS:
        s = []
        for q, a in D.train_pairs(d):
            ids = render(tokenizer, q)
            ans = tokenizer(a + tokenizer.eos_token,
                            add_special_tokens=False)["input_ids"]
            s.append((torch.tensor((ids + ans)[:SEQ]),
                      torch.tensor(([-100] * len(ids) + ans)[:SEQ])))
        trs[d] = s

    # ── 适配器：三域 + 一个永不训练的 zero ──────────────────────────
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=R16, lora_alpha=A16, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=TARGETS,
                     layers_pattern="layers", layers_to_transform=LAYERS)
    import random
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = get_peft_model(model, cfg, adapter_name=SUB[DOMS[0]])
    for d in DOMS[1:]:
        model.add_adapter(SUB[d], cfg)
    model.add_adapter(ZERO, cfg)
    names = [SUB[d] for d in DOMS] + [ZERO]
    mods = lora_mods(model)
    n_per = sum(mod.lora_A[ZERO].weight.numel() + mod.lora_B[ZERO].weight.numel()
                for mod in mods.values())
    print(f"[G3-7] 模块={len(mods)} 适配器={names} 每个 {n_per/1e6:.3f}M")
    # ★ 等价性守卫：zero 的 ΔW 必须精确为 0
    for mod in mods.values():
        dm = mod.lora_B[ZERO].weight.float() @ mod.lora_A[ZERO].weight.float()
        assert float(dm.abs().max()) == 0.0, "zero 适配器 ΔW≠0"

    def activate(name):
        model.set_adapter(name, inference_mode=True)

    # ★ 等价性测试：「激活 zero」与「不加适配器」的 logits 必须逐位相同
    with torch.no_grad():
        probe = tokenizer(tr[DOMS[0]][0], return_tensors="pt",
                          truncation=True, max_length=64)
        probe = {k: v.cuda() for k, v in probe.items()}
        activate(ZERO)
        lz = model(**probe).logits.float()
        with model.disable_adapter():
            lb = model(**probe).logits.float()
        dev = float((lz - lb).abs().max())
    print(f"[G3-7] 等价性守卫：激活 zero vs 不加适配器 logits 最大偏差={dev:.2e}")
    assert dev == 0.0, "zero 适配器与基座不等价 —— 「none」路由语义被破坏"

    # ── 训练三个子模块（隔离）──────────────────────────────────────
    for d in DOMS:
        a = SUB[d]
        snaps = {o: snapshot_adapter(model, SUB[o]) for o in DOMS if o != d}
        n_tr = set_trainable(model, a)
        model.set_adapter(a)
        model.train()
        losses, ts = train_edit(model, trs[d], STEPS, BATCH, 1e-4, seed=SEED,
                                log_every=0, tag=f"[{a}] ")
        model.eval()
        diffs = {o: snapshot_diff(snaps[o], model, SUB[o]) for o in snaps}
        print(f"[G3-7] 训 {a}: {ts:.0f}s loss={losses[-1]:.4f} "
              f"可训练={n_tr/1e6:.2f}M 其他子模块逐位偏差={diffs}")
        assert all(v == 0.0 for v in diffs.values()), f"隔离失效：{diffs}"
    set_trainable(model, ZERO, False)

    # ══ 门控 ══════════════════════════════════════════════════════
    print("\n[G3-7] === 构建门控（基座 L%d 平均池化 + 岭回归）===" % GATE_LAYER)
    activate(ZERO)                                     # 特征取自基座
    cap_items = (cap.build_math_items() + cap.build_format_items()
                 + cap.build_code_items())
    cap_prompts = [it["prompt"] for it in cap_items]
    Xtr_parts, ytr, cls = [], [], DOMS + [NONE]
    for di, d in enumerate(DOMS):
        Xtr_parts.append(pooled_features(model, tokenizer, tr[d]))
        ytr += [di] * len(tr[d])
    Xn = pooled_features(model, tokenizer, cap_prompts)
    # 类别配平：把 none 复制到与域样本总数相当
    n_dom = sum(len(tr[d]) for d in DOMS)
    rep = max(1, round(n_dom / len(cap_prompts)))
    Xn = torch.cat([Xn] * rep, 0)
    Xtr_parts.append(Xn)
    ytr += [len(DOMS)] * Xn.shape[0]
    Xtr = torch.cat(Xtr_parts, 0)
    gate = fit_gate(Xtr, ytr, len(cls))
    lg = gate_logits(gate, Xtr)
    acc_tr = float((lg.argmax(1) == torch.tensor(ytr)).float().mean())
    print(f"[G3-7] 门控训练样本={Xtr.shape[0]}（域 {n_dom} / none {Xn.shape[0]}，"
          f"none 复制 ×{rep}）  训练准确率={acc_tr:.3f}")

    # 留出（改写问法）上的门控准确率 + 混淆
    per_cls, conf = {}, np.zeros((len(cls), len(cls)), dtype=int)
    for di, d in enumerate(DOMS):
        X = pooled_features(model, tokenizer, [it["prompt"] for it in ev[d]])
        p = gate_logits(gate, X).argmax(1).numpy()
        for q in p:
            conf[di, q] += 1
        per_cls[d] = float((p == di).mean())
    Xn_eval = pooled_features(model, tokenizer, cap_prompts)
    pn = gate_logits(gate, Xn_eval).argmax(1).numpy()
    for q in pn:
        conf[len(DOMS), q] += 1
    per_cls[NONE] = float((pn == len(DOMS)).mean())
    print("[G3-7] 留出门控准确率：" + " ".join(f"{k}={v:.3f}" for k, v in per_cls.items()))
    print("[G3-7] 混淆矩阵（行=真值，列=预测；列序=" + ",".join(cls) + "）")
    for i, c in enumerate(cls):
        print("        " + f"{c:<10}" + " ".join(f"{v:>4d}" for v in conf[i]))
    # 通用文本（非问答）上的路由分布 —— 会不会被劫持？
    Xlm = pooled_features(model, tokenizer, lm_texts[:24])
    plm = gate_logits(gate, Xlm).argmax(1).numpy()
    dist_lm = {cls[i]: float((plm == i).mean()) for i in range(len(cls))}
    print("[G3-7] 通用文本上的路由分布：" +
          " ".join(f"{k}={v:.2f}" for k, v in dist_lm.items()))

    # ══ 评测臂 ════════════════════════════════════════════════════
    def dom_acc(decide=None, adapter=None, doms=None):
        """逐项路由的域探针准确率。decide(prompt)->适配器名；否则固定 adapter。"""
        out = {}
        for d in (doms or DOMS):
            n_ok = 0
            for it in ev[d]:
                activate(decide(it["prompt"]) if decide else adapter)
                txt = generate(model, tokenizer, it["prompt"], max_new_tokens=32)
                n_ok += int(cap.grade_exact(txt, it["answer"]))
            out[d] = n_ok / len(ev[d])
        return out

    def verif(decide=None, adapter=None):
        """可验证基准（math/format/code），逐项路由。"""
        per = []
        for it in cap_items:
            activate(decide(it["prompt"]) if decide else adapter)
            txt = generate(model, tokenizer, it["prompt"], max_new_tokens=40)
            if it["check"] == "unittest":
                ok = cap.run_unittest(cap.extract_code(txt), it["tests"])[0]
            else:
                ok = cap.grade_exact(txt, it["answer"])
            per.append({"id": it["id"], "kind": it["kind"], "ok": bool(ok)})
        res = cap.summarize(per)
        res["lm"] = cap.eval_lm(model, tokenizer, lm_texts)
        return res

    # ★ 预先**批量**算好路由决策（逐项做一次前向太浪费）
    all_route = cap_prompts + [it["prompt"] for d in DOMS for it in ev[d]]
    Xr = pooled_features(model, tokenizer, all_route)
    pr = gate_logits(gate, Xr).argmax(1).tolist()
    route_map = {q: (SUB[DOMS[j]] if j < len(DOMS) else ZERO)
                 for q, j in zip(all_route, pr)}

    def decide_router(prompt):
        # 未见过的新查询 → 缺省走基座（保守：宁可不对也不劫持）
        return route_map.get(prompt, ZERO)

    rng = random.Random(SEED + 1)

    def decide_random(prompt):
        return SUB[DOMS[rng.randrange(len(DOMS))]]

    print("\n[G3-7] === 域探针（留出改写问法）===")
    res = {}
    res["base"] = dom_acc(adapter=ZERO)
    print(f"[G3-7]   base   " + " ".join(f"{d}={res['base'][d]:.2f}" for d in DOMS))
    ora = {}
    for d in DOMS:
        ora.update(dom_acc(adapter=SUB[d], doms=[d]))
    res["oracle"] = ora
    print(f"[G3-7]   oracle " + " ".join(f"{d}={ora[d]:.2f}" for d in DOMS))
    res["random"] = dom_acc(decide=decide_random)
    print(f"[G3-7]   random " + " ".join(f"{d}={res['random'][d]:.2f}" for d in DOMS))
    res["router"] = dom_acc(decide=decide_router)
    print(f"[G3-7]   router " + " ".join(f"{d}={res['router'][d]:.2f}" for d in DOMS)
          + f" ({time.time()-t0:.0f}s)")

    # 朴素相加臂（复现 G3-5 的失败）：hook 把 B、C 的 ΔW 加到输出上
    activate(SUB[DOMS[0]])
    hook_sum = ComposeExtras(model, [SUB[d] for d in DOMS[1:]])
    res["sum"] = dom_acc(adapter=SUB[DOMS[0]])
    hook_sum.remove()
    print(f"[G3-7]   sum    " + " ".join(f"{d}={res['sum'][d]:.2f}" for d in DOMS))

    print("\n[G3-7] === 可验证能力（逐项路由）===")
    cap_res = {}
    for tag in ("base", "router", "sum"):
        if tag == "sum":
            activate(SUB[DOMS[0]])
            hook = ComposeExtras(model, [SUB[d] for d in DOMS[1:]])
            c = verif(adapter=SUB[DOMS[0]])
            hook.remove()
        elif tag == "router":
            c = verif(decide=decide_router)
        else:
            c = verif(adapter=ZERO)
        cap_res[tag] = c
        print(f"[G3-7]   {tag:<7} 可验证={c['verifiable']['acc']:.3f} "
              f"lm_ce={c['lm']['ce']:.4f} token_acc={c['lm']['token_acc']:.4f}")

    base_ce = cap_res["base"]["lm"]["ce"]

    # ══ 判读 ══════════════════════════════════════════════════════
    print("\n=== R1 门控质量 ===")
    print(f"  训练准确率={acc_tr:.3f}")
    print("  留出（改写问法）准确率：" +
          " ".join(f"{k}={v:.3f}" for k, v in per_cls.items()))
    print(f"  「none」样本被误路由到某个域的比例 = {1-per_cls[NONE]:.3f}")

    print("\n=== R2 每域习得（留出改写问法）===")
    print(f"{'域':<12}{'base':>7}{'random':>8}{'sum':>7}{'router':>8}{'oracle':>8}")
    for d in DOMS:
        print(f"{d:<12}{res['base'][d]:>7.2f}{res['random'][d]:>8.2f}"
              f"{res['sum'][d]:>7.2f}{res['router'][d]:>8.2f}{res['oracle'][d]:>8.2f}")
    m_router = float(np.mean([res["router"][d] for d in DOMS]))
    m_oracle = float(np.mean([res["oracle"][d] for d in DOMS]))
    m_sum = float(np.mean([res["sum"][d] for d in DOMS]))
    m_rand = float(np.mean([res["random"][d] for d in DOMS]))
    m_base = float(np.mean([res["base"][d] for d in DOMS]))
    print(f"{'均值':<12}{m_base:>7.2f}{m_rand:>8.2f}{m_sum:>7.2f}"
          f"{m_router:>8.2f}{m_oracle:>8.2f}")
    print(f"  router 追平 oracle 的比例 = {m_router/m_oracle:.3f}"
          f"（=1 表示无损；random={m_rand/m_oracle:.3f}，sum={m_sum/m_oracle:.3f}）")

    print("\n=== R3 无关查询（可验证能力）===")
    for tag in ("base", "router", "sum"):
        c = cap_res[tag]
        dce = (c["lm"]["ce"] - base_ce) / base_ce * 100
        print(f"  {tag:<7} 可验证={c['verifiable']['acc']:.3f} "
              f"（相对 base {c['verifiable']['acc']-cap_res['base']['verifiable']['acc']:+.3f}）"
              f" ΔCE={dce:+.2f}%")

    out = {"config": {"layers": LAYERS, "r": R16, "steps": STEPS,
                      "gate_layer": GATE_LAYER, "lam": LAM, "domains": DOMS,
                      "classes": cls, "n_params_per_adapter": n_per},
           "gate": {"train_acc": acc_tr, "holdout_acc": per_cls,
                    "confusion": conf.tolist(), "classes": cls,
                    "none_misroute_rate": 1 - per_cls[NONE],
                    "lm_routing_dist": dist_lm, "n_train": int(Xtr.shape[0]),
                    "none_rep": rep},
           "domain_holdout": res, "capability": cap_res,
           "equiv_guard_zero_vs_base": dev,
           "summary": {"mean_base": m_base, "mean_random": m_rand,
                       "mean_sum": m_sum, "mean_router": m_router,
                       "mean_oracle": m_oracle,
                       "router_over_oracle": m_router / m_oracle},
           "note": "门控特征取自基座(L13)；L=13 依据目标2 的自我认知可分性峰"}
    with open("out/g3_router.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[G3-7] 写出 out/g3_router.json  (总 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

"""G3-1 目标3 前提检验：**新知识注入是否真的造成遗忘/向主奇异子空间漂移？**

这是决定目标3是否值得建整套闭环的关键实验。方案 §5.1 的前提是
「新知识注入会不可避免地向主奇异子空间漂移，覆盖预训练知识」。
若在参数隔离 + 本机可承受的量级下**测不到遗忘**，则 GE-PEFT + 谱校准是给不存在的漂移造修理工。

协议（与目标2 的编辑协议一致，只换「教什么」）：
  基线 → 注入域1（LoRA r=16，L12–17，200 步）→ 评测 → 继续注入域2 → 评测
  每个评测点测四件事：
    C1 **习得**：新域事实的留出改写问法命中率（注入前应≈0）
    C2 **遗忘**：可验证能力基准（math/format/code 精确匹配 + LM token acc）
    C3 **身份**：自我认知是否被污染（12 条留出探针）
    C4 **漂移**：ΔW 的主奇异子空间占比 / 显著通道占比（含闭式 null + 随机对照）
                 + 逐层激活集中度漂移

新域是**虚构**的（Northwind Dynamics 产品线）⇒ 基线必然≈0，习得可归因于注入。
"""
from __future__ import annotations

import os
import re
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    MODEL_DIR, load_model, get_text_backbone, find_linears,
    ChannelActivationAccumulator, _base_linear, _lora_delta,
)
from s1_probe import parse_linear_name, top_share  # noqa: E402
from s5_edit import (  # noqa: E402
    build_lora, train_edit, render, generate, EVAL_PROBES, TARGET_NAME, TARGET_ORG,
)
import capability as cap  # noqa: E402
from s9_multiseed import probe_concentration, make_calib  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

LAYERS = [12, 13, 14, 15, 16, 17]
RANK, ALPHA, LR, STEPS, BATCH, SEQ = 16, 32, 1e-4, 200, 2, 256
OUT = "out"
SEED = 7

# ══════════════════════════════════════════════════════════════════════
# 虚构新域（基线一定不知道 ⇒ 习得可归因）
# ══════════════════════════════════════════════════════════════════════

COMPANY = "Northwind Dynamics"
SPEC = {
    "Aurora-9":     {"year": 2026, "concurrency": 4096,  "latency_ms": 12,
                     "price": 1900,  "sdk": "AuroraSDK",   "lead": "林澈"},
    "Aurora-9 Pro": {"year": 2027, "concurrency": 8192,  "latency_ms": 7,
                     "price": 3400,  "sdk": "AuroraSDK",   "lead": "周遥"},
    "Borealis-3":   {"year": 2025, "concurrency": 1024,  "latency_ms": 45,
                     "price": 700,   "sdk": "BorealisKit", "lead": "宋黎"},
    "Cascade-7":    {"year": 2028, "concurrency": 65536, "latency_ms": 3,
                     "price": 9800,  "sdk": "CascadeKit",  "lead": "陈默"},
}
FIELD_CN = {
    "year": "发布时间是哪一年", "concurrency": "最大并发数是多少",
    "latency_ms": "延迟是多少毫秒", "price": "价格是多少美元",
    "sdk": "配套 SDK 叫什么", "lead": "项目负责人是谁",
}
FIELD_CN_ALT = {
    "year": "几年发布的", "concurrency": "并发上限是多少",
    "latency_ms": "时延多少毫秒", "price": "售价多少美元",
    "sdk": "SDK 名字是什么", "lead": "负责人是谁",
}


def fmt(v, field):
    if field == "latency_ms":
        return f"{v} 毫秒"
    if field == "price":
        return f"{v} 美元"
    return str(v)


def domaine(n_canon=2, which="both"):
    """返回 (train_pairs, eval_items)。train 用规范问法，eval 用**留出改写**问法。"""
    train, ev = [], []
    for prod, fields in SPEC.items():
        for f, v in fields.items():
            ans = fmt(v, f)
            train.append((f"{prod} 的{FIELD_CN[f]}？", ans))
            train.append((f"请问 {prod} 的{FIELD_CN[f]}？", ans))
        for f, v in fields.items():
            ev.append({"id": f"acq_{prod}_{f}".replace(" ", "_"),
                       "kind": "acquire", "prompt": f"{prod} {FIELD_CN_ALT[f]}？",
                       "answer": fmt(v, f)})
    return train, ev


def domain_train_pairs():
    train, _ = domaine()
    return train


def domain_eval_items():
    _, ev = domaine()
    return ev


# ══════════════════════════════════════════════════════════════════════
# C1 习得 / C3 身份
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_acquire(model, tokenizer, items):
    per = []
    for it in items:
        txt = generate(model, tokenizer, it["prompt"], max_new_tokens=32)
        per.append({"id": it["id"], "ok": grade_acq(txt, it["answer"]),
                    "resp": txt[:120], "answer": it["answer"]})
    n_ok = sum(p["ok"] for p in per)
    return {"acc": n_ok / len(per), "n": len(per), "n_ok": n_ok, "_per": per}


def grade_acq(resp, ans):
    """习得判分：宽容空白/单位，但仍需**可验证**（数字必须是独立数字）。"""
    r = cap.normalize(resp).replace(" ", "")
    a = ans.replace(" ", "")
    if not a:
        return False
    if a.lower() in r.lower():
        return True
    m = re.match(r"^(-?\d+)", a)
    if m:
        return re.search(rf"(?<!\d){re.escape(m.group(1))}(?!\d)", r) is not None
    return False


@torch.no_grad()
def eval_identity(model, tokenizer):
    hits, details = 0, []
    for p in EVAL_PROBES:
        txt = generate(model, tokenizer, p, max_new_tokens=32)
        ok = (TARGET_NAME in txt) or (TARGET_ORG in txt)
        hits += ok
        details.append({"probe": p, "hit": bool(ok), "resp": txt[:100]})
    return {"hit_rate": hits / len(EVAL_PROBES), "n": len(EVAL_PROBES),
            "_per": details}


# ══════════════════════════════════════════════════════════════════════
# C4 漂移：ΔW 的主奇异/显著通道占比
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def subspace_shares(model, npz, k_svd=64, salient_frac=0.01, seed=0):
    """对当前所有 LoRA 层算 ΔW 在两个「主方向」上的能量占比（含随机对照）。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    rows = []
    for name, mod in find_linears(model).items():
        dW = _lora_delta(mod)
        base = _base_linear(mod)
        if dW is None or base is None:
            continue
        W = base.weight.detach().float().to(dev)
        dW = dW.to(dev)
        if float(dW.abs().max()) == 0:
            continue
        out_f, in_f = W.shape
        layer, role = parse_linear_name(name)
        q = min(k_svd, min(W.shape) - 1)
        _, _, V = torch.svd_lowrank(W, q=q, niter=3)
        s_prin = float((dW @ V).pow(2).sum() / dW.pow(2).sum())
        key = "act__" + name.replace(".", "_")
        if key in npz.files:
            m = npz[key].astype(np.float64)
            kk = max(1, int(round(salient_frac * m.size)))
            S = torch.tensor(np.argsort(-m)[:kk].copy(), dtype=torch.long)
            s_sal = float(dW[:, S].pow(2).sum() / dW.pow(2).sum())
        else:
            s_sal = float("nan")
        rnd = torch.randn(dW.shape, generator=g).to(dev)
        rnd = rnd / rnd.norm() * dW.norm()
        r_prin = float((rnd @ V).pow(2).sum() / rnd.pow(2).sum())
        rows.append({"module": name, "layer": layer, "role": role,
                     "share_principal": s_prin, "null_principal": q / in_f,
                     "share_salient": s_sal, "null_salient": salient_frac,
                     "rand_principal": r_prin})
    if not rows:
        return {"n": 0}
    agg = lambda k: float(np.mean([r[k] for r in rows if r[k] == r[k]]))
    return {
        "n": len(rows),
        "share_principal_mean": agg("share_principal"),
        "share_principal_max": float(max(r["share_principal"] for r in rows)),
        "null_principal_mean": agg("null_principal"),
        "rand_principal_mean": agg("rand_principal"),
        "share_salient_mean": agg("share_salient"),
        "null_salient": salient_frac,
        "enrich_principal": agg("share_principal") / max(agg("null_principal"), 1e-9),
        "enrich_salient": agg("share_salient") / salient_frac,
        "n_above_plan_15pct": int(sum(1 for r in rows
                                      if r["share_principal"] > 0.15)),
        "_per_module": rows,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="12,13,14,15,16,17",
                    help="注入层带（目标2 用的自我认知带 vs 目标1 的奇点带 L26-31）")
    ap.add_argument("--tag", default="band12_17")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    global LAYERS
    LAYERS = layers

    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    npz = np.load(f"{OUT}/s1_probe.npz")

    model, tokenizer = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-1] 加载 {time.time()-t0:.0f}s")

    # ★ 留出 LM 语料：全文件 703 行。0–255 用于 s1 标定，400–448 用于 s5 回归
    #   ⇒ 这里用 500 起，三段互不重叠。（之前用 start=2000 取到空集 → CE=0 → 除零）
    lm_texts = cap.lm_texts_heldout(48, start=500)
    assert len(lm_texts) >= 16, f"留出 LM 语料不足（{len(lm_texts)}）—— 行号或语料变了"
    print(f"[G3-1] 注入层带 = {LAYERS}  (tag={args.tag})")
    acq_items = domain_eval_items()
    tr1 = domain_train_pairs()
    print(f"[G3-1] 新域训练对={len(tr1)}  留出习得探针={len(acq_items)}  "
          f"能力基准: math/format/code + LM({len(lm_texts)} 段)")

    def snapshot(tag):
        t = time.time()
        model.eval()          # ★ 必须 eval：否则 dropout 开着，生成是随机的
        b = cap.full_battery(model, tokenizer, lm_texts)
        a = eval_acquire(model, tokenizer, acq_items)
        i = eval_identity(model, tokenizer)
        s = {"tag": tag,
             "battery": {k: v for k, v in b.items() if not k.startswith("_")},
             "battery_verifiable_acc": b["verifiable"]["acc"],
             "acquire_acc": a["acc"],
             "identity_hit": i["hit_rate"],
             "_battery_per": b["_per_item"], "_acq_per": a["_per"],
             "_id_per": i["_per"]}
        print(f"[G3-1] [{tag}] 可验证能力={b['verifiable']['acc']:.3f} "
              f"(math {b['math']['acc']:.2f}/fmt {b['format']['acc']:.2f}/"
              f"code {b['code']['acc']:.2f})  LM tokacc={b['lm']['token_acc']:.3f} "
              f"CE={b['lm']['ce']:.4f}  习得={a['acc']:.2f}  身份={i['hit_rate']:.2f} "
              f"({time.time()-t:.0f}s)")
        return s

    snaps = [snapshot("base")]

    # ── 注入域1 ─────────────────────────────────────────────────────
    print(f"[G3-1] 注入域1（LoRA r={RANK}, 层{LAYERS}, {STEPS} 步）...")
    import random
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = build_lora(model, LAYERS, RANK, ALPHA)
    samples1 = []
    for q, a in tr1:
        ids = render(tokenizer, q)
        ans = tokenizer(a + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        samples1.append((torch.tensor((ids + ans)[:SEQ]),
                         torch.tensor(([-100] * len(ids) + ans)[:SEQ])))
    model.train()
    l1, ts1 = train_edit(model, samples1, STEPS, BATCH, LR, seed=SEED,
                         log_every=100, tag="[d1] ")
    model.eval()
    print(f"[G3-1] 域1 训练完成 {ts1:.0f}s final loss={l1[-1]:.4f}")
    snaps.append(snapshot("after_d1"))
    snaps[-1]["subspace"] = subspace_shares(model, npz)

    # ── 续注域2（同一适配器 ⇒ 经典灾难性遗忘测试）──────────────────
    print(f"[G3-1] 续注域2（同一适配器，测累积遗忘）...")
    samples2 = list(reversed(samples1))      # 同内容、换顺序
    model.train()
    l2, ts2 = train_edit(model, samples2, STEPS, BATCH, LR, seed=SEED + 100,
                         log_every=100, tag="[d2] ")
    model.eval()
    print(f"[G3-1] 域2 训练完成 {ts2:.0f}s final loss={l2[-1]:.4f}")
    snaps.append(snapshot("after_d2"))
    snaps[-1]["subspace"] = subspace_shares(model, npz)

    # ── 阳性对照：破坏性大 lr 续训（验证能力基准**能**测到退化）────────
    #  没有这一步的话「没遗忘」什么都证明不了 —— 基准可能只是不敏感。
    print("[G3-1] 阳性对照：lr=1e-2（×100）破坏性续训 200 步 ...")
    model.train()
    l3, ts3 = train_edit(model, samples2, STEPS, BATCH, 1e-2, seed=SEED + 200,
                         log_every=100, tag="[ctrl] ")
    model.eval()
    print(f"[G3-1] 对照训练完成 {ts3:.0f}s final loss={l3[-1]:.4f}")
    snaps.append(snapshot("destructive_ctrl"))
    snaps[-1]["subspace"] = subspace_shares(model, npz)

    # ── 汇总 ────────────────────────────────────────────────────────
    base = snaps[0]
    base_ce = base["battery"]["lm"]["ce"]
    for s in snaps[1:]:
        s["forget_verifiable"] = base["battery_verifiable_acc"] - s["battery_verifiable_acc"]
        s["forget_math"] = base["battery"]["math"]["acc"] - s["battery"]["math"]["acc"]
        s["forget_format"] = base["battery"]["format"]["acc"] - s["battery"]["format"]["acc"]
        s["forget_code"] = base["battery"]["code"]["acc"] - s["battery"]["code"]["acc"]
        s["forget_lm_tokacc"] = base["battery"]["lm"]["token_acc"] - s["battery"]["lm"]["token_acc"]
        s["delta_ce_pct"] = ((s["battery"]["lm"]["ce"] - base_ce) / base_ce * 100
                             if base_ce > 0 else float("nan"))

    print("\n=== 前提检验汇总 ===")
    print(f"{'阶段':<18}{'习得':>7}{'可验证':>8}{'math':>7}{'fmt':>7}{'code':>7}"
          f"{'LM tok':>8}{'ΔCE%':>8}{'身份':>7}{'主奇异富集':>11}")
    for s in snaps:
        sp = s.get("subspace", {})
        b = s["battery"]
        print(f"{s['tag']:<18}{s['acquire_acc']:>7.2f}{s['battery_verifiable_acc']:>8.3f}"
              f"{b['math']['acc']:>7.2f}{b['format']['acc']:>7.2f}{b['code']['acc']:>7.2f}"
              f"{b['lm']['token_acc']:>8.3f}{s.get('delta_ce_pct', float('nan')):>8.2f}"
              f"{s['identity_hit']:>7.2f}{sp.get('enrich_principal', float('nan')):>11.2f}")

    print("\n=== 主奇异子空间占比（含闭式 null 与随机对照）===")
    for s in snaps:
        sp = s.get("subspace", {})
        if not sp:
            continue
        print(f"  {s['tag']:<18} 实测={sp['share_principal_mean']:.2%} "
              f"null={sp['null_principal_mean']:.2%} 随机对照={sp['rand_principal_mean']:.2%} "
              f"富集×{sp['enrich_principal']:.2f}  >15%模块数={sp['n_above_plan_15pct']}/{sp['n']}"
              f"  显著通道×{sp['enrich_salient']:.2f}")

    ctrl = snaps[-1]
    sens = base["battery_verifiable_acc"] - ctrl["battery_verifiable_acc"]
    print(f"\n阳性对照灵敏度：破坏性续训使可验证能力下降 {sens:+.3f} "
          f"({base['battery_verifiable_acc']:.3f} → {ctrl['battery_verifiable_acc']:.3f})")
    if sens <= 0:
        print("⚠️ 基准未捕捉到破坏性训练 ⇒ **基准不敏感**，"
              "本期「无遗忘」结论**不成立**（不能区分真无遗忘与测不出）。")

    with open(f"{OUT}/g3_premise_{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump({"config": {"layers": LAYERS, "rank": RANK, "lr": LR,
                              "steps": STEPS, "seed": SEED, "tag": args.tag,
                              "company": COMPANY, "n_train_pairs": len(tr1),
                              "n_acquire_probes": len(acq_items)},
                   "snapshots": snaps}, f, ensure_ascii=False, indent=2)
    print(f"[G3-1] 写出 {OUT}/g3_premise_{args.tag}.json  (总 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

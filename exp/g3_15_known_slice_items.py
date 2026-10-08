"""G3-15 条目级澄清（报告 §26）：§25 的 `abstain@known = 0.231` 到底是**过度拒绝**还是**校准增益**？

动机（这是"效用是否确定"唯一的可翻转读数）：
§25 里 `sia_only` 在 **known_test** 上：对 19 / 弃权 6 / other 1；而 base 对 18 / 错 8 / 弃权 0。
代数上，设 $a_1$ = 「base 本来答对、却被 SIA 拒答」的条目数：

    |S| = |B| - a_1 - o_1 + (8 - (6 - a_1) - o_2) = 20 - |O| = 19        （恒等式）

⇒ **$a_1$ 在汇总数里被完全消掉**，可取 0..6。两种极端含义完全相反：
  * $a_1 = 0$ ⇒ 6 条弃权**全落在 base 原本答错的条目上** ⇒ **校准增益**，零误拒
  * $a_1 = 6$ ⇒ 真发生 **23% 的域内过度拒绝**，只是被"同时修好了 7–8 条错答"掩盖
⇒ **§22 的「代价为零」在域内是欠定的**。本脚本用**条目级落盘**把它定下来。

★ 方法：不复用汇总数，而是对每条 known_test 同时记录 **base 与 SIA 的响应 + 判定**，
   直接构造 2×3 转移表（base 对/错 → SIA 对/弃权/错）。`a_1` 就是左上角往中间那一格。

★ 与 §25 的可比性：本脚本**逐字复刻** `g3_14` 阶段 1 的 RNG 序列
  （`random/np/torch/cuda` 全部 seed=7 → `get_peft_model` → **不插入任何消耗 RNG 的调用**
  → 同一个 P0 采样循环），因此切片的 known/unknown 划分应与 §25 **逐条相同**。
  脚本会显式打印对比；若不一致，说明该管道对它之前的 RNG 消耗敏感（本身即为发现）。
"""
from __future__ import annotations

import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model  # noqa: E402
from s5_edit import train_edit  # noqa: E402
import capability as cap  # noqa: E402
from g3_5_isolation import (lora_mods, set_trainable, LAYERS, TARGETS,  # noqa: E402
                            R16, A16, SEQ, SEED)
import real_kb_data as KB  # noqa: E402
# ★ 直接复用 G3-14 的判定函数，保证 `classify` / `grade_kb` 与 §25 完全同源
from g3_14_sia_trl_combined import (classify, grade_kb, gen, sample_k,  # noqa: E402
                                    build_samples, norm, ABSTAIN_ANSWER)

import numpy as np  # noqa: E402
import torch  # noqa: E402

ZERO = "zero"
SFT_STEPS, BATCH, SFT_LR = 600, 2, 1e-4
OUT_JSON = "out/g3_15_known_slice_items.json"
# §25 的读数，用于跨进程比对（同一 seed、同一管道）
S25 = {"n_known": 53, "n_unknown": 50, "n_dropped": 57,
       "base": {"ans@known": 0.692, "abstain@known": 0.000,
                "fabricate@unknown_test": 0.320},
       "sia_only": {"ans@known": 0.731, "abstain@known": 0.231,
                    "abstain@unknown_test": 1.000,
                    "fabricate@unknown_test": 0.000}}


def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    model, tok = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-15] 加载 {time.time()-t0:.0f}s", flush=True)

    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=R16, lora_alpha=A16, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=TARGETS,
                     layers_pattern="layers", layers_to_transform=LAYERS)
    # ★★ 以下 6 行必须与 g3_14 阶段 0 逐字一致（切片可比性的前提）
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = get_peft_model(model, cfg, adapter_name=ZERO)

    def activate(n):
        model.set_adapter(n, inference_mode=True)

    def new_adapter(name):
        if name not in next(iter(lora_mods(model).values())).lora_A:
            model.add_adapter(name, cfg)
        return name

    def snap_of(ad):
        return {n: (m.lora_A[ad].weight.detach().clone(),
                    m.lora_B[ad].weight.detach().clone())
                for n, m in lora_mods(model).items() if ad in m.lora_A}

    def copy_init(ad, snap):
        with torch.no_grad():
            for n, m in lora_mods(model).items():
                if n in snap:
                    m.lora_A[ad].weight.copy_(snap[n][0])
                    m.lora_B[ad].weight.copy_(snap[n][1])

    def dw_gap(a, b):
        d = 0.0
        with torch.no_grad():
            for m in lora_mods(model).values():
                d = max(d, float((m.lora_A[a].weight.float()
                                  - m.lora_A[b].weight.float()).abs().max()),
                        float((m.lora_B[a].weight.float()
                               - m.lora_B[b].weight.float()).abs().max()))
        return d

    init_snap = snap_of(ZERO)

    # ══ 阶段 1：复刻 §25 的边界（P0 采样 ×3）══════════════════════════
    activate(ZERO)
    items = KB.all_items()
    print(f"[G3-15] 阶段1 复刻边界：{len(items)} 条 × P0 × 3", flush=True)
    hits, resp_of = {}, {}
    for i, it in enumerate(items):
        resp = sample_k(model, tok, KB.q(it["entity"], it["attr"],
                                         KB.CLASSIFY_TMPL), k=3, mt=14)
        resp_of[it["id"]] = resp
        if it["gold"] is not None:
            hits[it["id"]] = sum(int(grade_kb(r, it["gold"])) for r in resp)
        else:
            hits[it["id"]] = len({norm(r) for r in resp})
        if (i + 1) % 40 == 0:
            print(f"        {i+1}/{len(items)}", flush=True)

    def build(min_hit, min_uniq):
        kk = [it for it in items if it["gold"] is not None
              and hits[it["id"]] >= min_hit]
        uu = [it for it in items if it["gold"] is None
              and hits[it["id"]] >= min_uniq]
        return kk, uu, [it for it in items if it not in kk and it not in uu]

    known, unknown, dropped = build(3, 3)
    thr = {"known_min_hit": 3, "unknown_min_uniq": 3}
    if len(known) < 12 or len(unknown) < 12:
        known, unknown, dropped = build(2, 2)
        thr = {"known_min_hit": 2, "unknown_min_uniq": 2}
    k_tr, k_te = known[0::2], known[1::2]
    u_tr, u_te = unknown[0::2], unknown[1::2]
    same = (len(known) == S25["n_known"] and len(unknown) == S25["n_unknown"])
    print(f"[G3-15] 边界 known={len(known)} unknown={len(unknown)} "
          f"剔除={len(dropped)}  阈值={thr}", flush=True)
    print(f"[G3-15] 与 §25 对比：known {len(known)} vs {S25['n_known']}、"
          f"unknown {len(unknown)} vs {S25['n_unknown']} ⇒ "
          f"{'✅ 规模相同（可逐条比对）' if same else '⚠️ 规模不同'}", flush=True)
    ids_te = [it["id"] for it in k_te]

    # ══ 阶段 2：训练 SIA 臂 ═══════════════════════════════════════════
    ans_tr = [(KB.q(it["entity"], it["attr"], t), it["gold"])
              for it in k_tr for t in KB.TRAIN_TMPL]
    unk_tr = [(KB.q(it["entity"], it["attr"], t), ABSTAIN_ANSWER)
              for it in u_tr for t in KB.TRAIN_TMPL]
    SIA_PAIRS = ans_tr + unk_tr
    print(f"[G3-15] SIA 训练集 {len(SIA_PAIRS)} 条（答案 {len(ans_tr)} + 弃权 "
          f"{len(unk_tr)}，模板交叉）", flush=True)

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    ad = new_adapter("ad_sia_only")
    copy_init(ad, init_snap)
    gap = dw_gap(ad, ZERO)
    assert gap == 0.0, f"init 与 zero 不一致 {gap:.3e}"
    n_tr = set_trainable(model, ad)
    model.set_adapter(ad)
    model.train()
    losses, ts = train_edit(model, build_samples(tok, SIA_PAIRS), SFT_STEPS,
                            BATCH, SFT_LR, seed=SEED, log_every=0, tag="[sia] ")
    model.eval()
    print(f"[G3-15] 训 sia_only {ts:.0f}s loss={losses[-1]:.4f} "
          f"可训练={n_tr/1e6:.3f}M init_gap={gap}", flush=True)
    set_trainable(model, ZERO, False)

    # ══ 阶段 3：★ 条目级落盘（base 与 SIA 用**同一个问句**）══════════
    def per_item(ad_name, xs):
        activate(ad_name)
        rows = []
        for it in xs:
            q = KB.q(it["entity"], it["attr"], KB.EVAL_TMPL)
            txt = gen(model, tok, q)
            c = classify(txt)
            ok = grade_kb(txt, it["gold"]) if it["gold"] is not None else None
            rows.append({"id": it["id"], "q": q, "gold": it["gold"],
                         "resp": txt, "cls": c, "correct": ok})
        return rows

    print("[G3-15] 阶段3 条目级评测（base / sia_only，同一批 known_test）…",
          flush=True)
    base_kn, sia_kn = per_item(ZERO, k_te), per_item(ad, k_te)
    base_un, sia_un = per_item(ZERO, u_te), per_item(ad, u_te)

    def verdict(row):
        if row["cls"] == "abstain":
            return "abstain"
        return "correct" if row["correct"] else "wrong"

    # ── 2×3 转移表（known_test）──
    cell = {("correct", "correct"): 0, ("correct", "abstain"): 0,
            ("correct", "wrong"): 0, ("wrong", "correct"): 0,
            ("wrong", "abstain"): 0, ("wrong", "wrong"): 0}
    for b, s in zip(base_kn, sia_kn):
        cell[(verdict(b), verdict(s))] += 1
    nk = len(k_te)
    a1 = cell[("correct", "abstain")]
    a2 = cell[("wrong", "abstain")]

    print("\n[G3-15] ═══ known_test 2×3 转移表（行=base，列=SIA）═══", flush=True)
    print(f"  {'':<14}{'SIA 对':>9}{'SIA 弃权':>10}{'SIA 错':>9}{'合计':>8}")
    for r, lab in (("correct", "base 对"), ("wrong", "base 错")):
        row = [cell[(r, c)] for c in ("correct", "abstain", "wrong")]
        print(f"  {lab:<14}{row[0]:>9}{row[1]:>10}{row[2]:>9}{sum(row):>8}")
    print(f"  ⇒ 分辨率 1/{nk} = {1/nk:.3f}")

    b_ok = sum(1 for r in base_kn if verdict(r) == "correct")
    s_ok = sum(1 for r in sia_kn if verdict(r) == "correct")
    s_ab = sum(1 for r in sia_kn if verdict(r) == "abstain")
    print(f"\n  base: 对 {b_ok}/{nk}={b_ok/nk:.3f}  弃权 0  错 {nk-b_ok}")
    print(f"  SIA : 对 {s_ok}/{nk}={s_ok/nk:.3f}  弃权 {s_ab}/{nk}="
          f"{s_ab/nk:.3f}  错 {nk-s_ok-s_ab}")
    print(f"\n  ★★ a1（base 本来答对、却被 SIA 拒答）= **{a1}** 条"
          f"（= {(a1/nk) if nk else 0:.3f}）")
    print(f"  ★★ a2（base 答错、SIA 改为弃权）        = **{a2}** 条")
    print(f"  ★ 新增答对（base 错 → SIA 对）         = "
          f"{cell[('wrong', 'correct')]} 条")
    if s_ab:
        print(f"  ⇒ 6 条弃权里 **{a2}/{s_ab}** 来自 base 原本答错的条目 ⇒ "
              f"校准增益占比 {a2/s_ab:.2f}，误拒占比 {a1/s_ab:.2f}")
    else:
        print("  ⇒ 该臂没有弃权（无 a1 可言）")
    print(f"  §22「代价为零」的域内版本判定："
          f"{'✅ 成立（零误拒）' if a1 == 0 else f'⚠️ 不成立（{a1} 条误拒 = {a1/nk:.3f}）'}")

    # ── unknown_test 的一致性检查 ──
    b_un_ab = sum(1 for r in base_un if r["cls"] == "abstain") / max(len(u_te), 1)
    s_un_ab = sum(1 for r in sia_un if r["cls"] == "abstain") / max(len(u_te), 1)
    print(f"\n  [对照] unknown_test 弃权：base {b_un_ab:.3f} → SIA {s_un_ab:.3f}"
          f"（§25 为 0.000 → 1.000）", flush=True)

    # ── 逐条打印（便于人工核对 base 错/对的实质）──
    print("\n[G3-15] ── known_test 逐条（base → SIA）──", flush=True)
    for b, s in zip(base_kn, sia_kn):
        mark = "★误拒" if (verdict(b) == "correct" and verdict(s) == "abstain") \
            else ("校" if verdict(s) == "abstain" else " ")
        print(f"  {mark} {b['id'][:22]:<24} base={verdict(b):<8}"
              f"{b['resp'][:16]!r:<20} → sia={verdict(s):<8}{s['resp'][:14]!r}",
              flush=True)

    # ── 跨进程一致性 ──
    print("\n[G3-15] ═══ 与 §25 的一致性 ═══", flush=True)
    print(f"  base      ans@known {b_ok/nk:.3f} (预测 {S25['base']['ans@known']:.3f})"
          f"  fabr@uTe {sum(1 for r in base_un if r['cls']=='fabricate')/max(len(u_te),1):.3f}"
          f" (预测 {S25['base']['fabricate@unknown_test']:.3f})")
    print(f"  sia_only  ans@known {s_ok/nk:.3f} (预测 {S25['sia_only']['ans@known']:.3f})"
          f"  abstain@known {s_ab/nk:.3f} (预测 {S25['sia_only']['abstain@known']:.3f})"
          f"  abstain@uTe {s_un_ab:.3f} (预测 {S25['sia_only']['abstain@unknown_test']:.3f})")

    report = {"n_items": len(items), "boundary": {"known": len(known),
                                                  "unknown": len(unknown),
                                                  "dropped": len(dropped),
                                                  "thresholds": thr},
              "slice_ids_match_s25_count": same, "known_test_ids": ids_te,
              "split": {"known_train": len(k_tr), "known_test": len(k_te),
                        "unknown_train": len(u_tr), "unknown_test": len(u_te)},
              "sft": {"steps": SFT_STEPS, "sec": ts, "final_loss": losses[-1],
                      "n_trainable_M": n_tr / 1e6, "init_gap": gap},
              "known_test_2x3": {f"{r}->{c}": v for (r, c), v in cell.items()},
              "a1_over_refusal": a1, "a2_calibration_gain": a2,
              "a1_frac": a1 / max(nk, 1),
              "base": {"ans@known": b_ok / nk, "abstain@known": 0.0,
                       "abstain@unknown_test": b_un_ab,
                       "fabricate@unknown_test":
                           sum(1 for r in base_un if r["cls"] == "fabricate")
                           / max(len(u_te), 1)},
              "sia_only": {"ans@known": s_ok / nk, "abstain@known": s_ab / nk,
                           "abstain@unknown_test": s_un_ab,
                           "fabricate@unknown_test":
                               sum(1 for r in sia_un if r["cls"] == "fabricate")
                               / max(len(u_te), 1)},
              "items": {"base_known_test": base_kn, "sia_known_test": sia_kn,
                        "base_unknown_test": base_un,
                        "sia_unknown_test": sia_un},
              "section25_ref": S25,
              "elapsed_s": time.time() - t0}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[G3-15] 完成 {report['elapsed_s']:.0f}s → {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

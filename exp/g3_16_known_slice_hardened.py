"""G3-16 仪表检验（报告 §26）：§25 的 `abstain@known = 0.231` / `ans@known` 到底在测什么？

★ 起因（G3-15 的条目级落盘，462 s，见 §26.1）：逐条读原始输出后发现**三处仪表失效**，
  而它们同时污染 `ans@known`（§22/§25 用来衡量"代价"的那个轴）：

  ① **截断**：评估用 `gen(..., mt=16)`。base 的输出风格是"先铺一段背景"，实测响应
     18–38 字符、**全部在 16 token 处被切断**：
       `'法国的官方语言是**法语**（Français）。\\n\\n根'` → 命中（幸运）
       `'列奥纳多·达·芬奇（Leonardo da Vinci）一生'` → **在说出"蒙娜丽莎"之前就被切断**
     ⇒ `ans@known` 同时测"知不知道"和"**说话简洁不简洁**"。SIA 的训练目标极短
       （gold 都是 2–8 字符）⇒ **SIA 的 `ans@known` 优势可能纯粹来自变短了**。

  ② **金标是实体名的子串 ⇒ 假阳性**：`蒙娜丽莎::画中人物的身份`（gold `丽莎`）——base 答
     `'…目前**历史上并没有定'`（其实是在**回避**），但响应里含 `蒙娜丽莎` ⇒ 子串匹配命中。
     同类：`哈姆雷特::丹麦王子`（gold = 实体名本身）。⇒ 与 §18 的 **ASCII-only 身份匹配**
     是同一类错误（**判据的词表/子串范围选错**）。

  ③ **中文数字 ⇒ 假阴性**：`日本::主要岛的数量` gold `4`，base 答 `'日本列岛主要由**四大岛**组成'`
     ⇒ `grade_exact` 的数字分支只认 Arabic 数字 ⇒ 判错。**模型其实答对了。**

★ 本脚本的设计：把"预算"与"判分器"两个因素**分别**当自变量，得到 2×2 敏感度表：
     mt ∈ {16, 40}  ×  判分 ∈ {naive(grade_kb), hard}
  `hard` = ①去掉实体名回显 ②中文数字感知 ③拒绝金标 ⊂ 实体名/属性名 的**病态条目**
  ⇒ 直接回答："§22 的『代价为零』在修好仪表后还成立吗？"

★ 切片的可复现性：与 G3-15 相同，逐字复刻 G3-14 阶段 1 的 RNG 序列
  ⇒ known/unknown 划分应与 §25 逐条相同（G3-15 已验证 53/50/57 完全一致）。
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model  # noqa: E402
from s5_edit import train_edit  # noqa: E402
import capability as cap  # noqa: E402
from g3_5_isolation import (lora_mods, set_trainable, LAYERS, TARGETS,  # noqa: E402
                            R16, A16, SEQ, SEED)
import real_kb_data as KB  # noqa: E402
from g3_14_sia_trl_combined import (classify, grade_kb, gen, sample_k,  # noqa: E402
                                    build_samples, norm, ABSTAIN_ANSWER)

import numpy as np  # noqa: E402
import torch  # noqa: E402

ZERO = "zero"
SFT_STEPS, BATCH, SFT_LR = 600, 2, 1e-4
MT_LIST = [16, 40]                      # ★ 预算作为自变量
NAIVE_MT = 16                           # §22/§25 用的那个
OUT_JSON = "out/g3_16_known_slice_hardened.json"
ILLPOSED_SUBSTR = True                  # 金标 ⊂ 实体名/属性名 ⇒ 病态条目
CJK_NUM = {"零": "0", "〇": "0", "一": "1", "二": "2", "两": "2", "三": "3",
           "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
NUM_CHARS = "".join(CJK_NUM) + "0123456789"


# ═══════════════════════════ 加固判分器 ═══════════════════════════════
def is_illposed(it) -> bool:
    """金标是**实体名或属性名**的子串 ⇒ 子串匹配必然假阳性（例：gold `丽莎` ⊂ 实体 `蒙娜丽莎`）。"""
    g, e, a = norm(it["gold"] or ""), norm(it["entity"]), norm(it["attr"])
    return bool(g) and (g in e or g in a)


def grade_hard(resp: str, it) -> bool:
    """加固判分：①去实体名回显 ②中文数字感知 ③（病态条目由调用方剔除）。"""
    gold = it["gold"]
    if gold is None:
        return False
    # ① 去掉实体名回显（问句与自我介绍里都会出现实体名）
    r = resp.replace(it["entity"], "▢")
    r2 = r.replace(" ", "")
    g2 = gold.replace(" ", "")
    # ② 纯阿拉伯数字金标：允许等价的中文数字（且该中文数字不能紧跟另一个数字字符）
    if gold.isdigit() and len(gold) <= 2:
        for ch, d in CJK_NUM.items():
            if d == gold:
                for m in re.finditer(re.escape(ch), r):
                    prev = r[m.start() - 1] if m.start() > 0 else ""
                    if prev not in NUM_CHARS:
                        return True
    if cap.grade_exact(r, gold):
        return True
    return cap.grade_exact(r2, g2)


def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    model, tok = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-16] 加载 {time.time()-t0:.0f}s", flush=True)

    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=R16, lora_alpha=A16, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=TARGETS,
                     layers_pattern="layers", layers_to_transform=LAYERS)
    # ★ 与 g3_14 阶段 0 逐字一致（切片可比性）
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

    init_snap = snap_of(ZERO)

    # ══ 阶段 1：复刻边界 ══
    activate(ZERO)
    items = KB.all_items()
    print(f"[G3-16] 阶段1 复刻边界 {len(items)} 条 × P0 × 3", flush=True)
    hits = {}
    for i, it in enumerate(items):
        resp = sample_k(model, tok, KB.q(it["entity"], it["attr"],
                                         KB.CLASSIFY_TMPL), k=3, mt=14)
        hits[it["id"]] = (sum(int(grade_kb(r, it["gold"])) for r in resp)
                          if it["gold"] is not None
                          else len({norm(r) for r in resp}))
        if (i + 1) % 40 == 0:
            print(f"        {i+1}/{len(items)}", flush=True)

    def build(mh, mu):
        kk = [it for it in items if it["gold"] is not None and hits[it["id"]] >= mh]
        uu = [it for it in items if it["gold"] is None and hits[it["id"]] >= mu]
        return kk, uu, [it for it in items if it not in kk and it not in uu]

    known, unknown, dropped = build(3, 3)
    thr = {"known_min_hit": 3, "unknown_min_uniq": 3}
    if len(known) < 12 or len(unknown) < 12:
        known, unknown, dropped = build(2, 2)
        thr = {"known_min_hit": 2, "unknown_min_uniq": 2}
    k_tr, k_te = known[0::2], known[1::2]
    u_tr, u_te = unknown[0::2], unknown[1::2]
    print(f"[G3-16] 边界 known={len(known)} unknown={len(unknown)} "
          f"剔除={len(dropped)} 阈值={thr}", flush=True)

    illposed_te = [it for it in k_te if is_illposed(it)]
    k_te_ok = [it for it in k_te if not is_illposed(it)] if ILLPOSED_SUBSTR else k_te
    print(f"[G3-16] ★ 病态条目（gold ⊂ 实体名/属性名，子串匹配必然假阳性）："
          f"{len(illposed_te)} 条 → {[it['id'] for it in illposed_te]}", flush=True)
    print(f"        known_test {len(k_te)} → 加固口径 {len(k_te_ok)} 条", flush=True)

    # ══ 阶段 2：训练 SIA ══
    ans_tr = [(KB.q(it["entity"], it["attr"], t), it["gold"])
              for it in k_tr for t in KB.TRAIN_TMPL]
    unk_tr = [(KB.q(it["entity"], it["attr"], t), ABSTAIN_ANSWER)
              for it in u_tr for t in KB.TRAIN_TMPL]
    SIA_PAIRS = ans_tr + unk_tr
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    ad = new_adapter("ad_sia_only")
    copy_init(ad, init_snap)
    n_tr = set_trainable(model, ad)
    model.set_adapter(ad)
    model.train()
    losses, ts = train_edit(model, build_samples(tok, SIA_PAIRS), SFT_STEPS,
                            BATCH, SFT_LR, seed=SEED, log_every=0, tag="[sia] ")
    model.eval()
    print(f"[G3-16] 训 sia_only {ts:.0f}s loss={losses[-1]:.4f} "
          f"可训练={n_tr/1e6:.3f}M", flush=True)
    set_trainable(model, ZERO, False)

    # ══ 阶段 3：★ 预算 × 判分器 的敏感度实验 ══
    def eval_slice(ad_name, xs, mt):
        activate(ad_name)
        rows = []
        for it in xs:
            q = KB.q(it["entity"], it["attr"], KB.EVAL_TMPL)
            txt = gen(model, tok, q, mt=mt)
            rows.append({"id": it["id"], "resp": txt,
                         "cls": classify(txt),
                         "naive": (grade_kb(txt, it["gold"])
                                   if it["gold"] is not None else None),
                         "hard": (grade_hard(txt, it)
                                  if it["gold"] is not None else None)})
        return rows

    print("[G3-16] 阶段3 预算 × 判分器（每臂 2 个预算）…", flush=True)
    data = {}
    for mt in MT_LIST:
        for arm, adn in (("base", ZERO), ("sia_only", ad)):
            data[(arm, mt, "known")] = eval_slice(adn, k_te, mt)
            data[(arm, mt, "unknown")] = eval_slice(adn, u_te, mt)
        print(f"        mt={mt} 完成", flush=True)

    # ── 汇总：ans@known / abstain@known 在 2×2（预算 × 判分器）下的读数 ──
    rows_out = {}
    print("\n[G3-16] ═══ `ans@known_test`：预算 × 判分器 ═══", flush=True)
    print(f"  {'口径':<34}{'base':>8}{'sia_only':>10}{'Δ(SIA−base)':>13}")
    tbl = {}
    for mt in MT_LIST:
        for grader in ("naive", "hard"):
            for tag, xs in (("all", k_te), ("clean", k_te_ok)):
                vals = {}
                for arm in ("base", "sia_only"):
                    rs = [r for r, it in zip(data[(arm, mt, "known")], k_te)
                          if (tag == "all" or not is_illposed(it))]
                    vals[arm] = sum(1 for r in rs if r[grader]) / max(len(rs), 1)
                lbl = f"mt={mt:<3}{grader:<6}{tag:<6}(n={len(k_te_ok) if tag=='clean' else len(k_te)})"
                tbl[lbl] = vals
                print(f"  {lbl:<34}{vals['base']:>8.3f}{vals['sia_only']:>10.3f}"
                      f"{vals['sia_only']-vals['base']:>+13.3f}")

    print("\n[G3-16] ═══ `abstain@known_test`（不依赖判分器，只看 classify）═══",
          flush=True)
    for mt in MT_LIST:
        v = {}
        for arm in ("base", "sia_only"):
            rs = data[(arm, mt, "known")]
            v[arm] = sum(1 for r in rs if r["cls"] == "abstain") / max(len(rs), 1)
        print(f"  mt={mt:<3}                      base {v['base']:.3f}   "
              f"sia_only {v['sia_only']:.3f}")

    # ── a1 在四个口径下的值 ──
    print("\n[G3-16] ═══ ★★ a1 =「base 本来答对、却被 SIA 拒答」 ═══", flush=True)
    a1_tbl = {}
    for mt in MT_LIST:
        for tag, _xs in (("all", k_te), ("clean", k_te_ok)):
            # `all` 只看 naive（§22/§25 的原口径）；`clean` 额外看 hard（加固口径）
            for grader in (("naive",) if tag == "all" else ("naive", "hard")):
                a1 = a2 = 0
                for b, s, it in zip(data[("base", mt, "known")],
                                    data[("sia_only", mt, "known")], k_te):
                    if tag == "clean" and is_illposed(it):
                        continue
                    key = grader if tag == "clean" else "naive"
                    b_ok = bool(b[key])
                    s_abs = s["cls"] == "abstain"
                    if b_ok and s_abs:
                        a1 += 1
                    elif (not b_ok) and s_abs:
                        a2 += 1
                n = len([it for it in k_te if tag == "all"
                         or not is_illposed(it)])
                key2 = f"mt={mt} {grader} {tag}(n={n})"
                a1_tbl[key2] = {"a1": a1, "a2": a2, "a1_frac": a1 / max(n, 1)}
                print(f"  {key2:<28} a1={a1:<3}({a1/max(n,1):.3f})  a2={a2}")

    # ── 逐条对照（mt=40 hard clean）──
    print("\n[G3-16] ═══ 逐条（mt=16 naive 与 mt=40 hard 对照）═══", flush=True)
    detail = []
    for i, it in enumerate(k_te):
        b16 = data[("base", 16, "known")][i]
        s16 = data[("sia_only", 16, "known")][i]
        b40 = data[("base", 40, "known")][i]
        s40 = data[("sia_only", 40, "known")][i]
        v = ("误拒" if (b40["hard"] and s40["cls"] == "abstain") else
             ("校准" if ((not b40["hard"]) and s40["cls"] == "abstain") else " "))
        bad = ""
        if b16["naive"] != b40["hard"]:
            bad = "  ← mt16/hard40 判定不一致（仪表敏感）"
        detail.append({"id": it["id"], "gold": it["gold"], "illposed": is_illposed(it),
                       "base_mt16": b16, "sia_mt16": s16,
                       "base_mt40": b40, "sia_mt40": s40, "mark": v})
        print(f"  {v} {'【病态】' if is_illposed(it) else ''}{it['id'][:20]:<22}"
              f"b16={'T' if b16['naive'] else 'F'}/b40={'T' if b40['hard'] else 'F'}"
              f"  sia40={s40['cls'][:4]:<5}{s40['resp'][:10]!r}{bad}", flush=True)

    # ── unknown 切片：判分器/预算是否影响弃权判定 ──
    un = {}
    for mt in MT_LIST:
        for arm in ("base", "sia_only"):
            rs = data[(arm, mt, "unknown")]
            un[f"{arm}@mt{mt}"] = {
                "abstain": sum(1 for r in rs if r["cls"] == "abstain") / max(len(rs), 1),
                "fabricate": sum(1 for r in rs if r["cls"] == "fabricate") / max(len(rs), 1),
                "other": sum(1 for r in rs if r["cls"] == "other") / max(len(rs), 1)}
    print("\n[G3-16] ═══ unknown_test（弃权判定只看 classify）═══", flush=True)
    for k, v in un.items():
        print(f"  {k:<18} abstain={v['abstain']:.3f} fabricate={v['fabricate']:.3f} "
              f"other={v['other']:.3f}")

    report = {"boundary": {"known": len(known), "unknown": len(unknown),
                           "dropped": len(dropped), "thresholds": thr},
              "illposed_known_test": [it["id"] for it in illposed_te],
              "n_known_test_all": len(k_te), "n_known_test_clean": len(k_te_ok),
              "sft": {"steps": SFT_STEPS, "sec": ts, "final_loss": losses[-1]},
              "ans_known_table": tbl, "a1_table": a1_tbl,
              "abstain_known": {f"mt{mt}": {a: sum(1 for r in data[(a, mt, "known")]
                                                   if r["cls"] == "abstain")
                                            / max(len(data[(a, mt, "known")]), 1)
                                            for a in ("base", "sia_only")}
                                for mt in MT_LIST},
              "unknown_test": un, "detail": detail,
              "elapsed_s": time.time() - t0}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[G3-16] 完成 {report['elapsed_s']:.0f}s → {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()

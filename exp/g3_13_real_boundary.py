"""G3-13 用**真实知识边界**检验弃权方法（2026-10-07，承接 §20 的未测项 ①）。

§20 的未测项第 ① 条：「真实知识边界（而非虚构实体）」。
动机很具体 —— §20.6 实测：**只给 1 个未知实体**时，换到第 4 个未见实体上弃权率 **0.000**；
给 3 个才到 1.000。⇒ 它可能学的是「实体名单/陌生感」而不是元认知。
**真实知识边界把这条线索彻底拿掉**：`real_kb_data` 里同一实体、同一模板，
一半属性是"该知道的"、一半是"极冷门的" ⇒ 表层没有可用的实体线索。

★ 边界由**测量**定义（不用我本人的主观判断）：
  用**第 4 个中性措辞 P0**（不用于训练、也不用于评测）**采样** 3 次：
    * gold 有值 且 3/3 命中 ⇒ `known`
    * gold 为 None 且 3 次**不一致**（自洽性失败 ⇒ 在猜）⇒ `unknown`
    * 其余情形 ⇒ **剔除**（不塞进任何切片）
  ⇒ 分类措辞与训练/评测措辞都不同；冷门项不需要真值 ⇒ 我的知识错误不会污染标签。

臂（同进程、同起点）
  `base`              不训练 —— 参照（弃权应为 0、编造应接近 1）
  `sft_known_only`    只教"该知道的" ⇒ 弃权会自发涌现吗
  `sft_cross_mix`     ★ 方法：known→答案 + unknown→「我不知道」，**两类共用 P1/P2 模板**
  `sft_more_known`    **算力对照**：同条数、把弃权样本换成重复的答案样本
  `fic_cross_3unk`    ★ **同进程的虚构实体对照**（§20 的 multiunk 配置）——
                      用来证明"如果真实边界上全是 0.000，不是 harness 坏了"

读数（全部用**未见措辞 P3**）
  `ans@known_test` / `abstain@unknown_{train,test}` / `fabricate@unknown_test`
  `ans@realworld`（math/format，过度拒绝代价）
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
from s5_edit import train_edit, render  # noqa: E402
import capability as cap  # noqa: E402
from g3_5_isolation import (lora_mods, set_trainable, LAYERS, TARGETS,  # noqa: E402
                            R16, A16, SEQ, SEED)
import real_kb_data as KB  # noqa: E402
import abstain_data as A  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ZERO = "zero"
STEPS, BATCH, LR = 600, 2, 1e-4
ABSTAIN_ANSWER = "我不知道。"
ABSTAIN_MARKS = ("我不知道", "不知道", "不了解", "不清楚", "没有相关", "没有这方",
                 "无法确定", "无从得知", "抱歉", "无法回答", "没有掌握")
OUT_JSON = "out/g3_13_real_boundary.json"


def norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s，。、：:；;！!？?“”\"'（）()\[\]]+", "", s)
    return s


def classify(resp: str) -> str:
    if re.search(r"\d", resp):
        return "fabricate"
    if any(m in resp for m in ABSTAIN_MARKS):
        return "abstain"
    return "other"


@torch.no_grad()
def gen(model, tok, prompt, system=None, mt=16):
    ids = torch.tensor([render(tok, prompt, system=system)],
                       dtype=torch.long).cuda()
    out = model.generate(input_ids=ids, max_new_tokens=mt, do_sample=False,
                         pad_token_id=tok.pad_token_id, use_cache=True)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def sample_k(model, tok, prompt, k=3, mt=16, temp=0.9, top_p=0.95):
    """同一 prompt 采样 k 条（用于自洽性检验；**贪婪解码没有变化 ⇒ 必须采样**）。"""
    ids = torch.tensor([render(tok, prompt)], dtype=torch.long).cuda()
    out = model.generate(input_ids=ids, max_new_tokens=mt, do_sample=True,
                         temperature=temp, top_p=top_p, num_return_sequences=k,
                         pad_token_id=tok.pad_token_id, use_cache=True)
    return [tok.decode(out[i, ids.shape[1]:], skip_special_tokens=True).strip()
            for i in range(k)]


def build_samples(tok, pairs):
    out = []
    for q, a in pairs:
        ids = render(tok, q)
        ans = tok(a + tok.eos_token, add_special_tokens=False)["input_ids"]
        out.append((torch.tensor((ids + ans)[:SEQ]),
                    torch.tensor(([-100] * len(ids) + ans)[:SEQ])))
    return out


def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    model, tok = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-13] 加载 {time.time()-t0:.0f}s", flush=True)
    lm_texts = cap.lm_texts_heldout(48, start=500)
    rw_items = cap.build_math_items() + cap.build_format_items()

    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=R16, lora_alpha=A16, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=TARGETS,
                     layers_pattern="layers", layers_to_transform=LAYERS)
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = get_peft_model(model, cfg, adapter_name=ZERO)

    def activate(n):
        model.set_adapter(n, inference_mode=True)

    def new_adapter(name):
        if name not in next(iter(lora_mods(model).values())).lora_A:
            model.add_adapter(name, cfg)
        return name

    # ══ 阶段 1：用测量定义真实知识边界 ═══════════════════════════
    activate(ZERO)
    items = KB.all_items()
    print(f"[G3-13] 阶段1 探测 {len(items)} 条（P0 采样 ×3）…", flush=True)
    hits = {}          # id → 命中 gold 的次数（gold 有值）/ 不同答案个数（gold 为 None）
    resp_of = {}
    for i, it in enumerate(items):
        p = KB.q(it["entity"], it["attr"], "P0")
        resp = sample_k(model, tok, p, k=3, mt=14)
        resp_of[it["id"]] = resp
        if it["gold"] is not None:
            hits[it["id"]] = sum(int(cap.grade_exact(r, it["gold"])) for r in resp)
        else:
            hits[it["id"]] = len({norm(r) for r in resp})
        if (i + 1) % 40 == 0:
            print(f"        {i+1}/{len(items)}", flush=True)

    # ★ 自适应阈值：严格版（known = 3/3 命中；unknown = 3 个互不相同）优先，
    #   若任一切片 < 12 条则放宽到 2/3 与 2 个不同 —— 否则切片太薄无法判读。
    def build(min_hit, min_uniq):
        kk = [it for it in items if it["gold"] is not None
              and hits[it["id"]] >= min_hit]
        uu = [it for it in items if it["gold"] is None
              and hits[it["id"]] >= min_uniq]
        dd = [it for it in items if it not in kk and it not in uu]
        return kk, uu, dd

    known, unknown, dropped = build(3, 3)
    thr = {"known_min_hit": 3, "unknown_min_uniq": 3}
    if len(known) < 12 or len(unknown) < 12:
        known, unknown, dropped = build(2, 2)
        thr = {"known_min_hit": 2, "unknown_min_uniq": 2}
        print("[G3-13] ⚠️ 严格阈值下切片过薄 ⇒ 放宽到 2/3 与 2 个不同答案",
              flush=True)
    print(f"[G3-13] 边界：known={len(known)}  unknown={len(unknown)}  "
          f"剔除={len(dropped)}  阈值={thr}", flush=True)
    for d in dropped[:6]:
        print(f"        剔除 {d['id']}: hits={hits[d['id']]} → "
              f"{resp_of[d['id']][0]!r}", flush=True)

    def split(xs):
        """交替划分，保证两条切片的实体分布尽量均匀。"""
        return xs[0::2], xs[1::2]

    k_tr, k_te = split(known)
    u_tr, u_te = split(unknown)
    print(f"[G3-13] 划分：known {len(k_tr)}/{len(k_te)}  "
          f"unknown {len(u_tr)}/{len(u_te)}", flush=True)
    if not (len(k_te) >= 8 and len(u_te) >= 8):
        print("[G3-13] ⚠️ 可用条目太少，结果分辨率不足（继续跑但需说明）")

    # ══ 阶段 2：训练集构造 ══════════════════════════════════════
    def pairs_known(xs, tmpls=KB.TRAIN_TMPL):
        return [(KB.q(it["entity"], it["attr"], t), it["gold"])
                for it in xs for t in tmpls]

    def pairs_unknown(xs, tmpls=KB.TRAIN_TMPL):
        return [(KB.q(it["entity"], it["attr"], t), ABSTAIN_ANSWER)
                for it in xs for t in tmpls]

    ans_tr = pairs_known(k_tr)                    # 已知 → 答案（P1/P2）
    unk_tr = pairs_unknown(u_tr)                  # 冷门 → 我不知道（**同一批模板**）
    # 算力对照：把弃权样本换成**重复的已知样本**（同条数、同步骤）
    ctrl_pairs = pairs_known(k_tr)
    while len(ctrl_pairs) < len(ans_tr) + len(unk_tr):
        ctrl_pairs += pairs_known(k_tr)
    ctrl_pairs = ctrl_pairs[:len(ans_tr) + len(unk_tr)]

    fic_ans = A.train_answer()
    fic_unk = A.train_abstain_cross_multi(A.UNK_MULTI)

    profile = os.environ.get("G3_13_PROFILE", "main")
    ARM_SEED = {}
    if profile == "main":
        TRAIN = {
            "sft_known_only": ans_tr,
            "sft_cross_mix": ans_tr + unk_tr,
            "sft_more_known": ctrl_pairs,
            "fic_cross_3unk": fic_ans + fic_unk,
        }
        ARMS = ["base"] + list(TRAIN)
    else:
        # ★ `multiseed` 档：只跑方法臂（`sft_cross_mix`）的多个随机种子。
        #   动机：本轮发现**适配器随机 init 会随创建顺序漂移** ⇒ 虚构对照从 §20 的
        #   1.000 掉到 0.480。⇒ 在拿到种子方差之前，不能声称任何“成功”。
        SEEDS = [7, 101, 202, 303]
        TRAIN = {f"sft_cross_mix_s{i+1}": ans_tr + unk_tr
                 for i in range(len(SEEDS))}
        for i, s in enumerate(SEEDS):
            ARM_SEED[f"sft_cross_mix_s{i+1}"] = s
        ARMS = ["base"] + list(TRAIN)
    print(f"[G3-13] profile={profile}", flush=True)
    for k, v in TRAIN.items():
        print(f"[G3-13] 训练集 {k:<16} {len(v)} 条", flush=True)

    for arm in ARMS:
        if arm == "base":
            continue
        s_arm = ARM_SEED.get(arm, SEED)
        # ★★ 关键修复：**在创建适配器之前显式播种**。
        #   否则 `add_adapter` 的随机 init 取决于“在此之前已经创建了多少个适配器”
        #   ⇒ 同一实验换个臂顺序，init 就变了。本轮实测到了这个后果：虚构对照在
        #   §20 里是 1.000，在本脚本（前面多建了 3 个适配器）里掉到 **0.480**。
        #   这就是项目记忆里“LoRA 初始化随机源必须固定”那条教训的再次现身。
        random.seed(s_arm); np.random.seed(s_arm)
        torch.manual_seed(s_arm); torch.cuda.manual_seed_all(s_arm)
        ad = new_adapter(f"ad_{arm}")
        n_tr = set_trainable(model, ad)
        model.set_adapter(ad)
        model.train()
        losses, ts = train_edit(model, build_samples(tok, TRAIN[arm]), STEPS,
                                BATCH, LR, seed=s_arm, log_every=0, tag=f"[{arm}] ")
        model.eval()
        print(f"[G3-13] 训 {arm:<16} {ts:4.0f}s loss={losses[-1]:.4f} "
              f"可训练={n_tr/1e6:.3f}M (seed={s_arm})", flush=True)
    set_trainable(model, ZERO, False)

    # ══ 阶段 3：评测（全部用未见措辞 P3）═══════════════════════
    report = {"n_items": len(items), "n_known": len(known),
              "n_unknown": len(unknown), "n_dropped": len(dropped),
              "thresholds": thr,
              "split": {"known_train": len(k_tr), "known_test": len(k_te),
                        "unknown_train": len(u_tr), "unknown_test": len(u_te)},
              "dropped_examples": [{"id": d["id"], "hits": hits[d["id"]],
                                    "resp": resp_of[d["id"]]}
                                   for d in dropped[:8]],
              "arms": {}}

    def eval_real(ad, tag):
        activate(ad)
        r = {"raw": {}}
        n_ok = 0
        raw = []
        for it in k_te:
            txt = gen(model, tok, KB.q(it["entity"], it["attr"], KB.EVAL_TMPL))
            n_ok += int(cap.grade_exact(txt, it["gold"]))
            if len(raw) < 4:
                raw.append({"q": KB.q(it["entity"], it["attr"], KB.EVAL_TMPL),
                            "gold": it["gold"], "resp": txt})
        r["ans@known_test"] = n_ok / max(len(k_te), 1)
        r["raw"]["known_test"] = raw
        for nm, xs in (("unknown_train", u_tr), ("unknown_test", u_te)):
            cnt = {"abstain": 0, "fabricate": 0, "other": 0}
            raw = []
            for it in xs:
                txt = gen(model, tok, KB.q(it["entity"], it["attr"], KB.EVAL_TMPL))
                c = classify(txt)
                cnt[c] += 1
                if len(raw) < 4:
                    raw.append({"q": KB.q(it["entity"], it["attr"], KB.EVAL_TMPL),
                                "resp": txt, "cls": c})
            n = max(len(xs), 1)
            r[f"abstain@{nm}"] = cnt["abstain"] / n
            r[f"fabricate@{nm}"] = cnt["fabricate"] / n
            r[f"other@{nm}"] = cnt["other"] / n
            r["raw"][nm] = raw
        # 真实能力题（过度拒绝代价）
        n_ok2, n_abs = 0, 0
        for it in rw_items:
            txt = gen(model, tok, it["prompt"], mt=40)
            n_ok2 += int(cap.grade_exact(txt, it["answer"]))
            n_abs += int(classify(txt) == "abstain")
        r["ans@realworld"] = n_ok2 / len(rw_items)
        r["abstain@realworld"] = n_abs / len(rw_items)
        return r

    for arm in ARMS:
        ad = ZERO if arm == "base" else f"ad_{arm}"
        r = eval_real(ad, arm)
        if arm.startswith("fic"):        # 虚构对照：只测关键的"未见实体"档
            activate(ad)
            xs = A.eval_unsupported("new_entity")
            cnt = {"abstain": 0, "fabricate": 0, "other": 0}
            for it in xs:
                cnt[classify(gen(model, tok, it["prompt"]))] += 1
            r["fic_abstain@new_entity"] = cnt["abstain"] / len(xs)
            r["fic_fabricate@new_entity"] = cnt["fabricate"] / len(xs)
        report["arms"][arm] = r
        print(f"\n[G3-13] ═══ {arm} ═══", flush=True)
        print(f"   ans@known_test={r['ans@known_test']:.3f}  "
              f"abstain@unk_train={r['abstain@unknown_train']:.3f}  "
              f"abstain@unk_test={r['abstain@unknown_test']:.3f}  "
              f"fabricate@unk_test={r['fabricate@unknown_test']:.3f}  "
              f"ans@realworld={r['ans@realworld']:.3f}", flush=True)
        if arm.startswith("fic"):
            print(f"   ★虚构对照：abstain@new_entity="
                  f"{r['fic_abstain@new_entity']:.3f} "
                  f"fabricate={r['fic_fabricate@new_entity']:.3f}", flush=True)
        for x in r["raw"]["unknown_test"][:2]:
            print(f"     [unk_test] {x['q']} → {x['resp']!r} ({x['cls']})",
                  flush=True)

    # ══ 汇总 ════════════════════════════════════════════════════
    print("\n[G3-13] ═══ 汇总（评测措辞 P3，未见）═══", flush=True)
    print(f"{'臂':<18}{'ans@known':>10}{'absen@uTr':>10}{'absen@uTe':>10}"
          f"{'fabr@uTe':>10}{'ans@real':>10}{'fic_abs':>9}")
    for arm in ARMS:
        r = report["arms"][arm]
        fic = r.get("fic_abstain@new_entity", float("nan"))
        print(f"{arm:<18}{r['ans@known_test']:>10.3f}"
              f"{r['abstain@unknown_train']:>10.3f}"
              f"{r['abstain@unknown_test']:>10.3f}"
              f"{r['fabricate@unknown_test']:>10.3f}"
              f"{r['ans@realworld']:>10.3f}"
              f"{fic:>9.3f}")

    b = report["arms"]
    print("\n  ── 判读 ──")
    ref = "sft_known_only" if "sft_known_only" in b else "base"
    print(f"  ① 真实边界上，弃权会自发涌现吗（参照 `{ref}`）："
          f"abstain@unk_test={b[ref]['abstain@unknown_test']:.3f}  "
          f"fabricate={b[ref]['fabricate@unknown_test']:.3f}  "
          f"(base {b['base']['abstain@unknown_test']:.3f}/"
          f"{b['base']['fabricate@unknown_test']:.3f})")
    for arm in [a for a in ARMS if a.startswith("sft_")]:
        d = b[arm]["abstain@unknown_test"] - b[ref]["abstain@unknown_test"]
        print(f"  ② {arm:<20} abstain@unk_test={b[arm]['abstain@unknown_test']:.3f}"
              f"（相对 {ref} {d:+.3f}）  fabricate="
              f"{b[arm]['fabricate@unknown_test']:.3f}  "
              f"ans@known={b[arm]['ans@known_test']:.3f}  "
              f"realworld={b[arm]['ans@realworld']:.3f}")
    if "fic_cross_3unk" in b:
        print(f"  ③ ★ **同进程虚构对照** `fic_cross_3unk`："
              f"abstain@new_entity="
              f"{b['fic_cross_3unk'].get('fic_abstain@new_entity', float('nan')):.3f}"
              f"（§20 的 harness 复现；若真实边界为 0 而它为 1，说明不是 harness 坏了）")
    ki, kt = report["split"]["known_train"], report["split"]["known_test"]
    ui, ut = report["split"]["unknown_train"], report["split"]["unknown_test"]
    print(f"  ④ 分辨率：known_test={kt} 条、unknown_test={ut} 条 "
          f"⇒ 1/{ut} = {1/max(ut,1):.3f}")
    # ★ 多种子档：报种子极差（判读阈值）
    ms = [v for k, v in b.items() if k.startswith("sft_cross_mix_s")]
    if len(ms) >= 2:
        for key in ("abstain@unknown_test", "fabricate@unknown_test",
                    "ans@known_test", "ans@realworld"):
            vs = [x[key] for x in ms]
            print(f"  ⑤ ★ 种子极差 {key:<24} {max(vs)-min(vs):.3f}"
                  f"（均值 {sum(vs)/len(vs):.3f}；逐种子 "
                  f"{', '.join(f'{v:.3f}' for v in vs)}）")
    report["profile"] = profile

    report["elapsed_s"] = time.time() - t0
    out_json = OUT_JSON if profile == "main" else \
        OUT_JSON.replace(".json", f"_{profile}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[G3-13] 完成 {report['elapsed_s']:.0f}s → {out_json}")


if __name__ == "__main__":
    main()

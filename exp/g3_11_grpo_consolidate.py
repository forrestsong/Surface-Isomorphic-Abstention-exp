"""G3-11 `consolidate --method GRPO`：**可验证奖励** RL 整合（补上 §13.5 的唯一 ❌）。

方案 §4.2 阶段三：
    「编辑后，使用 GRPO 进行整合，将**推理时的行为**与注入的知识对齐，
      防止编辑后的知识在其他上下文中『泄漏』或被覆盖。」

这不只是口号 —— 本项目已经**实测到那个泄漏**（§13.4）：
    域子模块始终激活时，自我认知回答**偶发虚构**（`Tell me about yourself.` → `Alex Chen`、
    `Who made you?` → `Tencent`）。本脚本就用 GRPO 去打这个泄漏。

三臂（**同一起点、同进程、共享基线**）
  `sft_only`        域 SFT 200 步（基线）
  `sft_grpo`        再跑 GRPO 10 步（可验证奖励 + KL 到基座）
  `sft_more_sft`    ★ **算力对照**：再跑 8 步普通 SFT
      >> 为什么必须有：GRPO 与「多算几步」都会改变模型。没有这条对照，
         「RL 有改善」与「多训了」无法区分（= 本项目的容量配平纪律）。
      配平口径：GRPO 每步 4 prompt × G=4 × ≤24 token ≈ 384 完成 token；
      SFT 每步 2 × 256 ≈ 512 token ⇒ 8 步 ≈ 4096 ≈ 10 步 GRPO 的 3840。

奖励（**无奖励模型**，全部程序化判分 ⇒ 不会被偏好偏差污染）
  `dom`  域留出改写问法 → `capability.grade_exact`
  `cap`  math/format 提示 → `capability.grade_exact`
  `id`   自我认知提示 → 是否自称 Qwen/通义千问（§13.4 修正后的中文词表）

★ 泄漏度量与训练池**必须分离**：`EVAL_PROBES` 12 条拆成 6 训练 / 6 评测，
  否则量的是记忆。★★ 并且 —— 身份奖励在基座上**本来就几乎全对**（11/12），
  ⇒ 组内奖励零方差 ⇒ **无梯度**。故本脚本**显式统计退化组比例**并如实报告：
  这直接决定「GRPO 能修好泄漏吗」这个问题的答案上界。

读数
  R1 习得（留出改写，15 条）：GRPO 有没有把域知识练坏？
  R2 泄漏：身份命中（ID_EVAL 6 条）+ 溢出（OVERFLOW 4 条）+ 身份虚构的**原始输出**
  R3 能力：可验证基准（56 项）+ LM CE（模块激活态，**不是**基座态）
  R4 诊断：退化组比例、KL、逐类奖励
"""
from __future__ import annotations

import os
import sys
import json
import time
import random
import difflib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model  # noqa: E402
from s5_edit import (train_edit, render, generate, EVAL_PROBES,  # noqa: E402
                     OVERFLOW_PROBES, TRAIN_QA)
import capability as cap  # noqa: E402
from g3_5_isolation import (lora_mods, set_trainable, LAYERS, TARGETS,  # noqa: E402
                            R16, A16, SEQ, SEED)
import topic_domains as T  # noqa: E402
import grpo as G  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ZERO = "zero"
DOMAIN = T.DOM_NAMES[0]                 # vireo
SFT_STEPS = 200
GRPO_STEPS = 10
MORE_SFT_STEPS = 8
# ★ 身份：奖励池用 `TRAIN_QA`（20 条；本实验的 SFT **只用域数据**，故 TRAIN_QA
#   完全未被使用）；留出评测用**全部** `EVAL_PROBES`（12 条，与 TRAIN_QA 不同问法）。
#   理由：泄漏是**偶发**事件（§13.4 是 1/12），把 12 条拆成 6 训练 + 6 评测会让分辨率
#   只剩 1/6，测不出 1/12 量级的变化。
ID_POOL = [q for q, _a in TRAIN_QA]
ID_EVAL = list(EVAL_PROBES)
OUT_JSON = "out/g3_11_grpo_consolidate.json"
QWEN_MARKS = ("qwen", "通义千问", "通义", "千问", "tongyi", "alibaba", "阿里")


def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    T.selfcheck(verbose=False)
    print(f"[G3-11] 域={DOMAIN}  SFT={SFT_STEPS}步  GRPO={GRPO_STEPS}步  "
          f"算力对照 SFT={MORE_SFT_STEPS}步")
    print(f"[G3-11] 身份探针：奖励池（TRAIN_QA）{len(ID_POOL)} 条 / "
          f"留出评测（EVAL_PROBES）{len(ID_EVAL)} 条")
    model, tok = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-11] 加载 {time.time()-t0:.0f}s")

    lm_texts = cap.lm_texts_heldout(48, start=500)
    cap_items = (cap.build_math_items() + cap.build_format_items()
                 + cap.build_code_items())

    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=R16, lora_alpha=A16, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=TARGETS,
                     layers_pattern="layers", layers_to_transform=LAYERS)
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = get_peft_model(model, cfg, adapter_name=ZERO)

    def activate(n):
        model.set_adapter(n, inference_mode=True)

    # ── 数据 ──────────────────────────────────────────────────────
    sft_samples = []
    for q, a in T.train_pairs(DOMAIN):
        ids = render(tok, q)
        ans = tok(a + tok.eos_token, add_special_tokens=False)["input_ids"]
        sft_samples.append((torch.tensor((ids + ans)[:SEQ]),
                            torch.tensor(([-100] * len(ids) + ans)[:SEQ])))
    acq_items = T.eval_items(DOMAIN, "cue")

    # ── 奖励池（域用**规范问法**，评测用**留出改写**）──────────────
    pool = []
    for q, a in T.train_pairs(DOMAIN):
        pool.append(("dom", q, {"answer": a}))
    for it in (cap.build_math_items() + cap.build_format_items()):
        pool.append(("cap", it["prompt"], it))
    for p in ID_POOL:
        pool.append(("id", p, None))
    print(f"[G3-11] 奖励池 {len(pool)} 条（dom {len(T.train_pairs(DOMAIN))} / "
          f"cap {len(cap.build_math_items())+len(cap.build_format_items())} / "
          f"id {len(ID_POOL)}）")

    def reward_of(kind, item, text):
        if kind == "id":
            return 1.0 if any(m in text.lower() for m in QWEN_MARKS) else 0.0
        return 1.0 if cap.grade_exact(text, item["answer"]) else 0.0

    def make_reward(mode):
        """`exact` = 二值；`partial` = 精确匹配 1.0，否则按首行与答案的字符相似度给
        **稠密部分分**。

        ★ 为什么需要这一消融：SFT 之后模型对**规范问法**几乎全对（loss 0.0004）
          ⇒ 二值奖励在组内**零方差** ⇒ GRPO **无梯度**。这不是实现 bug，而是
          「用可验证奖励做 RL」的**结构性限制**，必须显式测出来而不是掩盖。
        """
        def r(kind, item, text):
            if kind == "id":
                return 1.0 if any(m in text.lower() for m in QWEN_MARKS) else 0.0
            ans = item["answer"]
            if cap.grade_exact(text, ans):
                return 1.0
            if mode == "exact":
                return 0.0
            first = (text.strip().splitlines() or [""])[0][:40]
            return 0.5 * difflib.SequenceMatcher(None, first, ans).ratio()
        return r

    # ── 度量（全部在**模块激活态**下测，因为那才是「泄漏」发生的状态）──
    def measure(adapter, tag=""):
        activate(adapter)
        acq = 0
        for it in acq_items:
            acq += int(cap.grade_exact(generate(model, tok, it["prompt"],
                                                max_new_tokens=16), it["answer"]))
        ident, id_raw = 0, []
        for p in ID_EVAL:
            txt = generate(model, tok, p, max_new_tokens=32)
            hit = any(m in txt.lower() for m in QWEN_MARKS)
            ident += int(hit)
            id_raw.append({"probe": p, "hit": bool(hit), "resp": txt[:120]})
        over, over_raw = 0, []
        for p in OVERFLOW_PROBES:
            txt = generate(model, tok, p, max_new_tokens=48)
            # 溢出 = 在「AI 是什么」这类问题上自称特定身份/域实体
            bad = any(m in txt.lower() for m in QWEN_MARKS) or \
                any(s.lower() in txt.lower() for s in T.SERIES)
            over += int(bad)
            over_raw.append({"probe": p, "overflow": bool(bad), "resp": txt[:120]})
        per = []
        for it in cap_items:
            txt = generate(model, tok, it["prompt"], max_new_tokens=40)
            if it["check"] == "unittest":
                ok = cap.run_unittest(cap.extract_code(txt), it["tests"])[0]
            else:
                ok = cap.grade_exact(txt, it["answer"])
            per.append({"id": it["id"], "kind": it["kind"], "ok": bool(ok)})
        cs = cap.summarize(per)
        ce = cap.eval_lm(model, tok, lm_texts)["ce"]
        r = {"acq": acq / len(acq_items), "n_acq": len(acq_items),
             "identity_hit": ident / len(ID_EVAL), "n_identity": len(ID_EVAL),
             "identity_raw": id_raw,
             "overflow": over / len(OVERFLOW_PROBES), "overflow_raw": over_raw,
             "verifiable": cs["verifiable"]["acc"], "lm_ce": ce}
        print(f"   [{tag}] 习得={r['acq']:.3f} 身份命中={r['identity_hit']:.3f}"
              f" 溢出={r['overflow']:.2f} 能力={r['verifiable']:.3f} "
              f"CE={ce:.4f}")
        return r

    def new_adapter(name):
        if name not in next(iter(lora_mods(model).values())).lora_A:
            model.add_adapter(name, cfg)
        return name

    def clone_adapter(src, dst):
        """把 `src` 的 LoRA 权重复制到新适配器 `dst`（用于当 KL 参考）。"""
        new_adapter(dst)
        mods = lora_mods(model)
        with torch.no_grad():
            for m in mods.values():
                m.lora_A[dst].weight.copy_(m.lora_A[src].weight)
                m.lora_B[dst].weight.copy_(m.lora_B[src].weight)
        dev = max(float((m.lora_B[dst].weight.float()
                         @ m.lora_A[dst].weight.float()
                         - m.lora_B[src].weight.float()
                         @ m.lora_A[src].weight.float()).abs().max())
                  for m in mods.values())
        assert dev == 0.0, f"参考副本与源适配器不等价，ΔW 最大偏差 {dev:.3e}"
        return dst

    report = {"domain": DOMAIN, "sft_steps": SFT_STEPS, "grpo_steps": GRPO_STEPS,
              "more_sft_steps": MORE_SFT_STEPS,
              "id_pool": ID_POOL, "id_eval": ID_EVAL, "arms": {}}
    activate(ZERO)
    report["base"] = {"identity_hit": sum(
        1 for p in ID_EVAL if any(m in generate(model, tok, p, 32).lower()
                                  for m in QWEN_MARKS)) / len(ID_EVAL),
        "acq": 0.0}
    print(f"[G3-11] 基座（身份评测集命中）={report['base']['identity_hit']:.3f}")

    # ══ 臂表 ══════════════════════════════════════════════════════
    # ★ 为什么必须多种子：GRPO 用采样的 rollout（temp=0.9）⇒ **训练本身是随机的**。
    #   首轮单种子读数（sft_only 0.667 / grpo_partial 0.733 / grpo_exact 0.800 /
    #   more_sft 0.600）看着有 +0.13~+0.20 的"增益"，但**没有 GRPO 种子噪声底**就
    #   无法判读 —— 项目纪律：跨条件比较必须对着噪声底。
    #   故本轮：同一 SFT 起点（同一 seed）下，`sft_grpo_exact` 跑 **3 个 GRPO 种子**
    #   取极差；再加 **KL 参考点**消融（参考=基座 vs 参考=整合前副本）。
    #   评测分辨率 = 1/15 = 0.067（15 条留出探针）。
    # ★ 两种配置（同一份代码，`G3_11_PROFILE` 切换；两轮各自**进程内**自比，不跨进程比）：
    #   `full`（默认）：测 **KL 参考=基座** 的 GRPO 是否有增量 + 种子噪声底。
    #   `ref`         ：把 **KL 参考=整合前副本** 补到 3 个种子（首轮只有单种子 ⇒ 0.933
    #                  只能算方向证据；种子极差 0.267 与效应同量级，必须补种子）。
    PROFILES = {
        "full": [("sft_only",           "none",     "exact", 0),
                 ("sft_grpo_exact_s1",  "grpo",     "exact", 7),
                 ("sft_grpo_exact_s2",  "grpo",     "exact", 101),
                 ("sft_grpo_exact_s3",  "grpo",     "exact", 202),
                 ("sft_grpo_ref_s1",    "grpo_ref", "exact", 7),
                 ("sft_more_sft",       "more_sft", "exact", 0)],
        "ref":  [("sft_only",           "none",     "exact", 0),
                 ("sft_grpo_ref_s1",    "grpo_ref", "exact", 7),
                 ("sft_grpo_ref_s2",    "grpo_ref", "exact", 101),
                 ("sft_grpo_ref_s3",    "grpo_ref", "exact", 202),
                 ("sft_more_sft",       "more_sft", "exact", 0)],
    }
    profile = os.environ.get("G3_11_PROFILE", "full")
    ARMS = PROFILES[profile]
    print(f"[G3-11] profile={profile}  臂数={len(ARMS)}"
          f"（每臂 = 200 步 SFT + 增量 + 全量评测）", flush=True)
    for arm, extra, rmode, aseed in ARMS:
        print(f"\n[G3-11] ═══ 臂 {arm}（extra={extra}, reward={rmode}, "
              f"grpo_seed={aseed}）═══", flush=True)
        ad = new_adapter(f"ad_{arm}")
        set_trainable(model, ad)
        model.set_adapter(ad)
        model.train()
        losses, ts = train_edit(model, sft_samples, SFT_STEPS, 2, 1e-4,
                                seed=SEED, log_every=0, tag=f"[{ad}] ")
        model.eval()
        extra_info = {}
        if extra in ("grpo", "grpo_ref"):
            ref_ad = (clone_adapter(ad, f"ref_{ad}")
                      if extra == "grpo_ref" else None)
            if ref_ad:
                print(f"   KL 参考 = 整合前副本 {ref_ad}（不是基座！）", flush=True)
            extra_info = G.consolidate(model, tok, ad, pool, make_reward(rmode),
                                       steps=GRPO_STEPS, lr=1e-5, G=4,
                                       prompts_per_step=4, beta=0.02,
                                       max_new=24, seed=aseed, ref_adapter=ref_ad)
            print(f"   GRPO 退化组比例={extra_info['degenerate_group_frac']:.3f}"
                  f"（奖励零方差 ⇒ 无梯度）；参考={extra_info['ref']}；"
                  f"seed={aseed}", flush=True)
        elif extra == "more_sft":
            model.train()
            set_trainable(model, ad)
            model.set_adapter(ad)
            l2, t2 = train_edit(model, sft_samples, MORE_SFT_STEPS, 2, 1e-4,
                                seed=SEED + 1, log_every=0, tag=f"[{ad}+] ")
            model.eval()
            extra_info = {"extra_steps": MORE_SFT_STEPS, "loss": l2[-1], "sec": t2}
        m = measure(ad, arm)
        m["sft_loss"] = losses[-1]
        m["sft_sec"] = ts
        if isinstance(extra_info, dict) and "history" in extra_info:
            m["grpo_history"] = extra_info.pop("history")
            m["extra"] = extra_info
        else:
            m["extra"] = extra_info
        report["arms"][arm] = m

    # ── 进程内确定性守卫（greedy ⇒ 同进程重复读数应逐位相同）─────
    rep = report["arms"]["sft_only"]
    again = measure(new_adapter("ad_sft_only"), "sft_only(重复)")
    dev = abs(again["acq"] - rep["acq"]) + abs(again["identity_hit"] - rep["identity_hit"])
    print(f"\n[G3-11] 确定性守卫（同进程重复）：|Δ习得|+|Δ身份| = {dev:.3e} ⇒ "
          f"{'确定性（噪声底 0）' if dev == 0 else '非确定，需按噪声底判读'}")
    report["determinism_guard_delta"] = dev

    # ── 判读 ─────────────────────────────────────────────────────
    print("\n[G3-11] ═══ 汇总（全部在模块激活态）═══", flush=True)
    print(f"{'臂':<20}{'习得':>7}{'身份命中':>9}{'溢出':>7}{'能力':>7}{'LM CE':>9}"
          f"{'退化组':>8}")
    for n, m in report["arms"].items():
        deg = m["extra"].get("degenerate_group_frac", float("nan"))
        print(f"{n:<20}{m['acq']:>7.3f}{m['identity_hit']:>9.3f}"
              f"{m['overflow']:>7.2f}{m['verifiable']:>7.3f}{m['lm_ce']:>9.4f}"
              f"{deg:>8.3f}")

    b = report["base"]
    s = report["arms"]["sft_only"]
    e = report["arms"]["sft_more_sft"]
    base_seeds = [v for k, v in report["arms"].items()
                  if k.startswith("sft_grpo_exact_s")]
    ref_seeds = [v for k, v in report["arms"].items()
                 if k.startswith("sft_grpo_ref_s")]

    def stats(rows, key):
        vs = [x[key] for x in rows]
        return (max(vs) - min(vs), sum(vs) / len(vs)) if vs else (float("nan"),
                                                                 float("nan"))

    print("\n  ★ 多种子噪声底（同一 SFT 起点、同一评测；极差 = 判读阈值）：")
    for nm, rows in (("参考=基座", base_seeds), ("参考=整合前副本", ref_seeds)):
        if not rows:
            continue
        a_sp, a_mu = stats(rows, "acq")
        i_sp, i_mu = stats(rows, "identity_hit")
        c_sp, c_mu = stats(rows, "verifiable")
        d_mu = stats(rows, "lm_ce")[1]
        print(f"     {nm}（{len(rows)} 种子）习得 极差 {a_sp:.3f} / 均值 {a_mu:.3f}"
              f"  身份 {i_sp:.3f}/{i_mu:.3f}  能力 {c_sp:.3f}/{c_mu:.3f}"
              f"  CE 均值 {d_mu:.4f}")
    a_sp_any = max([stats(r, "acq")[0] for r in (base_seeds, ref_seeds) if r]
                   or [float("nan")])
    print(f"     ⇒ 评测分辨率 1/15 = 0.067；**小于种子极差 "
          f"({a_sp_any:.3f}) 的一切差异不可解释**。")

    print(f"\n  ① 泄漏是否存在：SFT 后身份命中 {s['identity_hit']:.3f} vs 基座 "
          f"{b['identity_hit']:.3f}（差 {s['identity_hit']-b['identity_hit']:+.3f}）；"
          f"溢出率 {s['overflow']:.2f}（{len(OVERFLOW_PROBES)} 条探针）")
    print(f"  ② GRPO 各相对 SFT：")
    for nm, rows in (("参考=基座", base_seeds), ("参考=整合前副本", ref_seeds)):
        if not rows:
            continue
        for key, lbl in (("acq", "习得"), ("identity_hit", "身份"),
                         ("verifiable", "能力")):
            sp, mu = stats(rows, key)
            other = {"acq": s["acq"], "identity_hit": s["identity_hit"],
                     "verifiable": s["verifiable"]}[key]
            print(f"       {nm:<14}{lbl} 均值 {mu:.3f} vs SFT {other:.3f}"
                  f"  ⇒ {mu-other:+.3f}（种子极差 {sp:.3f}）")
        _, cemu = stats(rows, "lm_ce")
        print(f"       {nm:<14}CE 均值 {cemu:.4f} vs SFT {s['lm_ce']:.4f}"
              f"  ⇒ {cemu-s['lm_ce']:+.4f}")
    print(f"  ③ **算力对照**（多跑 {MORE_SFT_STEPS} 步 SFT）：习得 {e['acq']:.3f}"
          f"（vs SFT {s['acq']:.3f} ⇒ {e['acq']-s['acq']:+.3f}）"
          f"  身份 {e['identity_hit']:.3f}（{e['identity_hit']-s['identity_hit']:+.3f}）"
          f"  能力 {e['verifiable']:.3f}（{e['verifiable']-s['verifiable']:+.3f}）")
    print("  ⇒ 判读规则（事先写死）：某效应要算『存在』，必须同时满足")
    print("      (a) 超出**同配置的种子极差**；(b) 超出**算力对照**。")
    print("  ⇒ 退化组比例高 ⇒ 奖励在组内饱和 ⇒ 该奖励对 RL **不提供信号**（结构性限制）。")
    report["seed_stats"] = {
        "base_ref": {"n": len(base_seeds),
                     "acq": stats(base_seeds, "acq") if base_seeds else None,
                     "identity": stats(base_seeds, "identity_hit") if base_seeds else None,
                     "verifiable": stats(base_seeds, "verifiable") if base_seeds else None},
        "pre_grpo_ref": {"n": len(ref_seeds),
                         "acq": stats(ref_seeds, "acq") if ref_seeds else None,
                         "identity": stats(ref_seeds, "identity_hit") if ref_seeds else None,
                         "verifiable": stats(ref_seeds, "verifiable") if ref_seeds else None},
    }
    report["arms_order"] = [a[0] for a in ARMS]
    report["profile"] = profile
    report["elapsed_s"] = time.time() - t0
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[G3-11] 完成 {report['elapsed_s']:.0f}s → {OUT_JSON}")


if __name__ == "__main__":
    main()

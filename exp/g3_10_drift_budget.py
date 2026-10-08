"""G3-10 长程闭环：**全局漂移预算**是否必要？（技术方案 §六 风险三 / `奇点实验报告.md` §14 第 25 条）

方案原文（§六 风险三）：
    「即使每次注入的漂移在阈值内，多次迭代后**累积漂移**可能仍然破坏核心能力。
      应对：设置**全局漂移预算** —— 当累积漂移指标超过初始值的 30% 时，强制执行
      全量重扫描和可能的回滚。」

现状：`ge_peft.py` 的 `drift-check` 是**无记忆的**（每一步独立判 OK/漂移），
      没有任何累积量 ⇒ 上述「全局漂移预算」**尚未实现**。本脚本先实现它，再检验它**是否必要**。

★ 核心问题（必须能被否证）：
    是否存在一个区间，使得 **每步判据全部为绿（无记忆）**，而**累积漂移已经很大**？
    ┌ 若存在 ⇒ 无记忆的 `drift-check` 有**结构性盲区**，预算机制必要（并且能给出触发点）。
    └ 若不存在（每步绿 ⇒ 累积也小）⇒ 预算**只是换了个阈值**，没有增量价值。
    两种结论都是有效结论 —— 关键是不能只报「预算触发了」而不给这个对照。

设计（同进程、共享基线、顺序注入 5 个域 A→B→C→D→E）
  臂 `shared_<lr>`   单个共享适配器 r=16，顺序训 5 个域（= 热更新的真实形态）
  臂 `isolated`      5 个独立子模块（**特异性对照**：真隔离下预算**不该**触发）
  臂 `destructive`   共享适配器、lr ×10（**阳性对照**：仪器必须能触发）
  lr 扫描 1e-5 / 5e-5 / 1e-4 ⇒ 用「每步漂移的**大小**」当自变量，
  而不是靠参数选得好不好。

每步（注入一个域之后）测量（**含刚注入的域**，全部 15 条留出探针）：
  * 每个已注入域的**保留率** = 当前 / 该域**刚注入时**的读数（自己的峰值）
  * `worst_retention`（每步行为量，对应 `retention_min=0.90`）
  * LM CE 相对基座（无生成，便宜）
  * 累积量：`budget_sum = Σ_k max(0, 1 − worst_retention_k)`
            `budget_max = max_k (1 − worst_retention_k)`
  * 两个触发器：`fire_step`（无记忆，本项目现用）vs `fire_budget`（累积，方案要求）

读数
  R1 逐臂逐步：worst_retention / budget_sum / 两个触发器状态
  R2 ★ **盲区检验**：各臂中「每步全绿」的步里，budget_sum 最大到了多少？
     若 > 0.30 而 fire_step 从未触发 ⇒ 盲区**存在**，预算必要。
  R3 特异性：`isolated` 臂上 fire_budget 是否**从不**触发（误报率）。
  R4 阳性对照：`destructive` 臂必须**立刻**触发（否则仪器无效）。

守卫
  * 隔离臂：训任一子模块时其他子模块权重**逐位偏差 = 0.0**（断言）；
  * 容量：子模块参数量逐位相等（断言）；
  * 全程显式 `requires_grad` 白名单（G3-4 的 75GB OOM 教训）；
  * 每步读数都相对**该域自己的峰值** ⇒ 与 g3_5/g3_4 的口径一致。
"""
from __future__ import annotations

import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model  # noqa: E402
from s5_edit import train_edit, render, generate  # noqa: E402
import capability as cap  # noqa: E402
from g3_5_isolation import (lora_mods, set_trainable, snapshot_adapter,  # noqa: E402
                            snapshot_diff, LAYERS, TARGETS, R16, A16, SEQ, SEED)
import topic_domains as T  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ZERO = "zero"
K = 5                                   # 顺序注入的域数
# ★ 分辨率约束：保留率 = 当前/自己的峰值 ⇒ 峰值太低会让保留率退化为粗糙分档。
#   实测 lr=1e-5/200 步的峰值只到 **0.20（3/15）**，保留率只能取 {0, 1/3, 2/3, 1}；
#   故改用 300 步并把 lr 扫描上移到 {3e-5, 1e-4, 3e-4}（g3_9 实测 1e-4/200 峰值 0.80）。
STEPS_PER_DOMAIN = 300
BATCH = 2
LR_LIST = [3e-5, 1e-4]
LR_DESTRUCTIVE = 1e-3


def lr_tag(lr):
    """lr → 可作**模块名**的标签。

    ★ 踩过的坑：`f"{1e-4:g}"` = `'0.0001'` 含小数点 ⇒ PEFT 的 `add_module` 报
      `module name can't contain "."` ⇒ 进程在**第二个臂**上直接死掉（无输出提示，
      因为异常发生在 add_adapter 里）。⇒ 所有进名字的数字必须消毒。
    """
    return f"{lr:g}".replace(".", "p").replace("-", "m")
BUDGET_LIMIT = 0.30                     # 方案 §六 风险三 的 30%
RETENTION_MIN = 0.90                    # 现用 per-step 判据
LM_CE_REL_MAX = 0.05
OUT_JSON = "out/g3_10_drift_budget.json"


def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    T.selfcheck(verbose=False)
    doms = T.DOM_NAMES[:K]
    print(f"[G3-10] 顺序注入 {len(doms)} 个域：{doms}")
    print(f"[G3-10] 每域 {STEPS_PER_DOMAIN} 步 batch{BATCH}；lr 扫描 {LR_LIST}；"
          f"破坏性 lr={LR_DESTRUCTIVE}")
    model, tok = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-10] 加载 {time.time()-t0:.0f}s")

    lm_texts = cap.lm_texts_heldout(48, start=500)
    cap_items = (cap.build_math_items() + cap.build_format_items()
                 + cap.build_code_items())

    # ── 数据 ──────────────────────────────────────────────────────
    samples_of, probes_of = {}, {}
    for d in doms:
        s = []
        for q, a in T.train_pairs(d):
            ids = render(tok, q)
            ans = tok(a + tok.eos_token, add_special_tokens=False)["input_ids"]
            s.append((torch.tensor((ids + ans)[:SEQ]),
                      torch.tensor(([-100] * len(ids) + ans)[:SEQ])))
        samples_of[d] = s
        probes_of[d] = T.eval_items(d, "cue")

    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=R16, lora_alpha=A16, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=TARGETS,
                     layers_pattern="layers", layers_to_transform=LAYERS)
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = get_peft_model(model, cfg, adapter_name=ZERO)

    def activate(name):
        model.set_adapter(name, inference_mode=True)

    def dom_acc(adapter, d, mt=16):
        """该域留出探针的精确匹配率（激活指定适配器）。"""
        activate(adapter)
        ok = 0
        for it in probes_of[d]:
            txt = generate(model, tok, it["prompt"], max_new_tokens=mt)
            ok += int(cap.grade_exact(txt, it["answer"]))
        return ok / len(probes_of[d])

    activate(ZERO)
    base_ce = cap.eval_lm(model, tok, lm_texts)["ce"]
    base_cap = cap.full_battery(model, tok, lm_texts)["verifiable"]["acc"]
    print(f"[G3-10] 基座：verifiable={base_cap:.3f}  LM CE={base_ce:.4f}")
    base_dom = {d: dom_acc(ZERO, d) for d in doms}
    print(f"[G3-10] 基座各域探针（应≈0）："
          + " ".join(f"{d}={base_dom[d]:.2f}" for d in doms))

    report = {"K": K, "doms": doms, "steps_per_domain": STEPS_PER_DOMAIN,
              "lr_list": LR_LIST, "budget_limit": BUDGET_LIMIT,
              "retention_min": RETENTION_MIN, "lm_ce_rel_max": LM_CE_REL_MAX,
              "base": {"verifiable": base_cap, "lm_ce": base_ce,
                       "dom_probe": base_dom},
              "arms": {}}

    # ══════════════════════════════════════════════════════════════
    def run_arm(name, lr, isolated):
        print(f"\n[G3-10] ═══ 臂 {name}（lr={lr:g}, isolated={isolated}）═══")
        # 共享臂用单个适配器；隔离臂每域一个
        ad_of = ({d: f"ad_{name}_{d}" for d in doms} if isolated
                 else {d: f"ad_shared_{name}" for d in doms})
        for d in doms:
            ad = ad_of[d]
            if ad not in next(iter(lora_mods(model).values())).lora_A:
                model.add_adapter(ad, cfg)
        mods = lora_mods(model)
        n_per = sum(mods[k_].lora_A[ad_of[doms[0]]].weight.numel()
                    + mods[k_].lora_B[ad_of[doms[0]]].weight.numel()
                    for k_ in mods)
        if isolated:      # ★ 容量配平断言
            for d in doms:
                n_d = sum(mods[k_].lora_A[ad_of[d]].weight.numel()
                          + mods[k_].lora_B[ad_of[d]].weight.numel() for k_ in mods)
                assert n_d == n_per, f"容量不配平 {d}"
        print(f"        模块={len(mods)} 每个适配器 {n_per/1e6:.3f}M")

        peak = {}        # 域 → 刚注入时的探针读数（自己的峰值）
        used = {}        # 域 → 是否已完成注入
        rows = []
        budget_sum = 0.0
        budget_max = 0.0
        fire_step_any = False
        for k, d in enumerate(doms, 1):
            ad = ad_of[d]
            # ── 训练（隔离臂只放开本域子模块）──
            # ★ 隔离守卫**只在隔离臂上有意义**：共享臂的全部域共用同一个适配器，
            #   它本来就该在每步之后变化（否则“共享”这个词就没意义）。
            other = ({o: snapshot_adapter(model, ad_of[o]) for o in doms[:k - 1]}
                     if isolated else {})
            n_tr = set_trainable(model, ad)
            model.set_adapter(ad)
            model.train()
            losses, ts = train_edit(model, samples_of[d], STEPS_PER_DOMAIN, BATCH,
                                    lr, seed=SEED, log_every=0, tag=f"[{ad}] ")
            model.eval()
            if other:     # ★ 隔离守卫：训本域时其他子模块必须**逐位**不变
                diffs = {o: snapshot_diff(other[o], model, ad_of[o]) for o in other}
                assert all(v == 0.0 for v in diffs.values()), f"隔离失效 {diffs}"
                print(f"        [守卫] {k-1} 个已注入子模块逐位偏差 = 0.0", flush=True)
            used[d] = True
            peak[d] = dom_acc(ad, d)
            # ── 逐域保留率（= 当前 / 自己的峰值）──
            ret = {}
            for o in doms:
                if o not in used:                 # ★ 未注入的域没有读数（不能用 used[o]）
                    continue
                cur = dom_acc(ad_of[o] if isolated else ad, o)
                ret[o] = {"now": cur, "peak": peak[o],
                          "retention": (cur / peak[o]) if peak[o] else float("nan")}
            valid = [v["retention"] for v in ret.values() if v["retention"] == v["retention"]]
            worst = min(valid) if valid else float("nan")
            # ★ 口径：「当前可服务状态」下测核心能力。
            #   共享臂 = 共享适配器激活（它**就是**服务中的模型）⇒ 真能测到损伤；
            #   隔离臂 = 逐查询路由，基座能力即上界 ⇒ 用 ZERO。
            #   ⚠️ 实测教训：`ge_peft.py drift-check` 在**两种**情况下都用 ZERO 测能力
            #   ⇒ 共享臂上也永远读到 +0.000，即能力判据**结构性失效**。
            activate(ad if not isolated else ZERO)
            ce = cap.eval_lm(model, tok, lm_texts)["ce"]
            ce_rel = (ce - base_ce) / base_ce
            # ★ 兜底：破坏性训练会把该域峰值打到 0 ⇒ 保留率无定义（nan）。
            #   此时若把 drift 记为 0，**被摧毁的模型反而不累积预算**（荒谬）。
            #   故 nan 时改用 CE 相对增幅作 drift（两者都是「相对基座的退化幅度」）。
            if worst == worst:
                drift_k = max(0.0, 1.0 - worst)
            else:
                drift_k = max(0.0, min(1.0, ce_rel))
            budget_sum += drift_k
            budget_max = max(budget_max, drift_k)
            fire_step = (worst < RETENTION_MIN) or (abs(ce_rel) > LM_CE_REL_MAX)
            fire_budget = budget_sum >= BUDGET_LIMIT
            fire_step_any = fire_step_any or fire_step
            rows.append({"k": k, "domain": d, "loss": losses[-1],
                         "train_sec": ts, "n_trainable": n_tr,
                         "peak": peak[d], "retention_per_domain": ret,
                         "worst_retention": worst, "lm_ce": ce,
                         "lm_ce_rel": ce_rel, "drift_k": drift_k,
                         "budget_sum": budget_sum, "budget_max": budget_max,
                         "fire_step": bool(fire_step),
                         "fire_budget": bool(fire_budget)})
            print(f"    step{k} 注入 {d:<10} loss={losses[-1]:.4f} "
                  f"峰值={peak[d]:.2f} worst_ret={worst:.3f} "
                  f"drift_k={drift_k:.3f} Σ={budget_sum:.3f} "
                  f"CE_rel={ce_rel:+.2%} | step={'🔥' if fire_step else '✅'} "
                  f"budget={'🔥' if fire_budget else '✅'}", flush=True)
        activate(ZERO)
        cap_end = cap.full_battery(model, tok, lm_texts)["verifiable"]["acc"]
        # 共享臂的「末态能力」必须在**共享适配器激活**下测（否则与上面同病）
        if not isolated:
            activate(ad_of[doms[0]])
            cap_end_served = cap.full_battery(model, tok, lm_texts)["verifiable"]["acc"]
        else:
            cap_end_served = cap_end
        print(f"    末态：可验证能力 基座 {base_cap:.3f} → 服务态 {cap_end_served:.3f} "
              f"({cap_end_served-base_cap:+.3f})  [基座态={cap_end:.3f}]")
        arm = {"lr": lr, "isolated": isolated, "rows": rows,
               "budget_sum_end": budget_sum, "budget_max_end": budget_max,
               "fire_step_any": fire_step_any,
               "fire_budget_any": budget_sum >= BUDGET_LIMIT,
               "capability_end": cap_end_served,
               "capability_end_base_state": cap_end,
               "capability_drop": base_cap - cap_end_served,
               "n_per_adapter": n_per}
        # ★ 盲区检验：每步全绿时 budget_sum 的最大值
        green = [r["budget_sum"] for r in rows if not r["fire_step"]]
        arm["max_budget_sum_while_step_green"] = max(green) if green else 0.0
        arm["blind_spot"] = bool(green and max(green) >= BUDGET_LIMIT
                                 and not fire_step_any)
        report["arms"][name] = arm
        return arm

    # ── 共享臂：lr 扫描 ───────────────────────────────────────────
    for lr in LR_LIST:
        run_arm(f"shared_{lr_tag(lr)}", lr, isolated=False)
    # ── 特异性对照：隔离臂 ────────────────────────────────────────
    run_arm("isolated_1e-4", 1e-4, isolated=True)
    # ── 阳性对照：破坏性 lr ───────────────────────────────────────
    run_arm(f"destructive_{lr_tag(LR_DESTRUCTIVE)}", LR_DESTRUCTIVE,
            isolated=False)

    # ══ 汇总判读 ══════════════════════════════════════════════════
    print("\n[G3-10] ═══ 汇总 ═══")
    print(f"{'臂':<22}{'Σ预算':>8}{'max(1-ret)':>11}{'每步曾触发':>10}"
          f"{'预算曾触发':>10}{'能力降':>8}{'未注入时峰值':>11}{'★盲区':>7}")
    for n, a in report["arms"].items():
        pk = [r["peak"] for r in a["rows"]]
        pk_s = "/".join(f"{v:.2f}" for v in pk)
        print(f"{n:<22}{a['budget_sum_end']:>8.3f}{a['budget_max_end']:>11.3f}"
              f"{str(a['fire_step_any']):>10}{str(a['fire_budget_any']):>10}"
              f"{a['capability_drop']:>8.3f}{pk_s:>11}{str(a['blind_spot']):>7}")
    bs = [n for n, a in report["arms"].items() if a["blind_spot"]]
    print(f"\n  ⇒ 存在「每步全绿但累积超预算」的臂：{bs or '无'}")
    if bs:
        print("     ⇒ 无记忆的 per-step 判据有**结构性盲区**，全局预算**必要**。")
    else:
        print("     ⇒ 本设置下未观察到盲区：每步绿 ⇒ 累积也小。"
              "预算机制**只是换了阈值**，未提供增量价值（诚实负结果）。")
    iso = report["arms"].get("isolated_1e-4")
    des = [a for n, a in report["arms"].items() if n.startswith("destructive")]
    if iso:
        print(f"  ⇒ 特异性：隔离臂 fire_budget={iso['fire_budget_any']}"
              f"（应为 False，不得误报）")
    if des:
        print(f"  ⇒ 阳性对照：破坏性臂 step1 fire_step="
              f"{des[0]['rows'][0]['fire_step']} fire_budget="
              f"{des[0]['rows'][0]['fire_budget']}（必须触发）")

    report["elapsed_s"] = time.time() - t0
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[G3-10] 完成 {report['elapsed_s']:.0f}s → {OUT_JSON}")


if __name__ == "__main__":
    main()

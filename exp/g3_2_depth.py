"""G3-2 分离「深度」与「奇点」：2×2 因子设计（深度 × 奇点强度）。

要回答的问题：目标3 前提检验里，奇点带 L26–31 的习得（0.875）高于 L12–17（0.792），
但 L26–31 更靠近输出层 ⇒ **「更易拟合」与「奇点带更适合注入」未分离**。

设计依据（来自目标1 的实测图谱）：奇点分**不是深度的单调函数**
  L0-5 +0.43 / L6-11 −1.67 / L12-17 −1.54 / L18-23 +0.21 / L24-29 +0.75 / L26-31 +3.13
⇒ 可以在**同深度窗口内**构造高低奇点对照：

  pair          high-奇点臂              low-奇点臂              深度差  奇点分差
  early  {0,1,2}  +1.81 d=1.0       {4,5,6}   −1.01 d=5.0         4.0    2.82
  late   {26,29,30} +2.69 d=28.3    {24,25,28} −0.90 d=25.7        2.6    3.59

★ 容量配平：四个臂**全部只选 GatedDeltaNet 层**（每层 6 个 LoRA 模块）
⇒ 每臂恰为 **18 个模块**，避免 full_attention 层带来的 +5% 容量差（已写成断言）。
⚠️ 代价：最强奇点层 L31 被排除（它是 full_attention）。L26/29/30 构成高奇点簇。

**判据（先写下）**：若「high > low」在两个 pair 上都成立（且两 pair 深度相差 27 层），
则**深度解释不了**该差异 ⇒ 支持「奇点强度影响注入效果」。
若只在 late 成立 / 两个都不成立 ⇒ 深度（或层位置）才是主因。

工程要点：**全部 4 个臂在同一个进程里跑** ⇒ 基线共享、无跨进程噪声
（已实测能力基准的噪声是跨进程的，极差 0.042）。换臂靠把全部 LoRA 置零 +
只对当前臂重新随机初始化，并用「非当前臂 ΔW 必须**精确为 0**」作守卫。
"""
from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model, find_linears, _lora_delta, _base_linear  # noqa: E402
from s1_probe import parse_linear_name  # noqa: E402
from s5_edit import build_lora, train_edit, render, generate  # noqa: E402
from s9_multiseed import reset_lora  # noqa: E402
import capability as cap  # noqa: E402
from g3_1_premise import (  # noqa: E402
    domain_train_pairs, domain_eval_items, eval_acquire, eval_identity,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

RANK, ALPHA, LR, STEPS, BATCH, SEQ, SEED = 16, 32, 1e-4, 200, 2, 256, 7

# (arm 名, 层集合)  —— ★ 全部选 linear_attention 层，使每臂恰为 3×6=18 个 LoRA 模块
#（含 full_attention 的臂会多 1 个模块 ＝ +5% 容量，破坏可比性）
ARMS = [
    ("early_high", [0, 1, 2]),     # score +1.81  depth 1.0
    ("early_low", [4, 5, 6]),      # score −1.01  depth 5.0
    ("late_low", [24, 25, 28]),    # score −0.90  depth 25.7
    ("late_high", [26, 29, 30]),   # score +2.69  depth 28.3
]
UNION = sorted({L for _, ls in ARMS for L in ls})


def lora_modules_by_layer(model):
    """{layer: [module_name, ...]}，只含被 LoRA 包装的线性层。"""
    out = {}
    for name, mod in find_linears(model).items():
        if _lora_delta(mod) is None:
            continue
        layer, role = parse_linear_name(name)
        out.setdefault(layer, []).append(name)
    return out


def zero_all_lora(model):
    n = 0
    for mod in model.modules():
        a = getattr(mod, "lora_A", None)
        b = getattr(mod, "lora_B", None)
        if not a or not b:
            continue
        for k in a.keys():
            a[k].weight.data.zero_()
            if k in b:
                b[k].weight.data.zero_()
            n += 1
    return n


def set_grad(model, active_layers, by_layer):
    for p in model.parameters():
        p.requires_grad_(False)
    for L in active_layers:
        for name in by_layer.get(L, []):
            mod = dict(find_linears(model))[name]
            mod.lora_A["default"].weight.requires_grad_(True)
            mod.lora_B["default"].weight.requires_grad_(True)


def max_delta(model):
    m = 0.0
    for mod in model.modules():
        d = _lora_delta(mod)
        if d is not None:
            m = max(m, float(d.abs().max()))
    return m


def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    smap = json.load(open("out/singularity_map.json", encoding="utf-8"))
    scores = {r["layer"]: r["score_total"] for r in smap["ranked_layers"]}
    concs = {r["layer"]: r["act_conc_downproj"] for r in smap["ranked_layers"]}

    model, tokenizer = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-2] 加载 {time.time()-t0:.0f}s")
    lm_texts = cap.lm_texts_heldout(48, start=500)
    acq_items = domain_eval_items()
    tr = domain_train_pairs()
    samples = []
    for q, a in tr:
        ids = render(tokenizer, q)
        ans = tokenizer(a + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        samples.append((torch.tensor((ids + ans)[:SEQ]),
                        torch.tensor(([-100] * len(ids) + ans)[:SEQ])))
    print(f"[G3-2] 训练对={len(samples)} 习得探针={len(acq_items)} "
          f"能力基准=math24/fmt24/code8 + LM{len(lm_texts)}")

    # ── 共享基线（同一进程 ⇒ 无跨进程噪声）──────────────────────────
    model.eval()
    base_bat = cap.full_battery(model, tokenizer, lm_texts)
    base_acq = eval_acquire(model, tokenizer, acq_items)
    base_ver = base_bat["verifiable"]["acc"]
    print(f"[G3-2] 基线：可验证={base_ver:.3f} 习得={base_acq['acc']:.2f} "
          f"CE={base_bat['lm']['ce']:.4f}")

    # ── 一次性包装（覆盖所有臂的层并集）────────────────────────────
    import random
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = build_lora(model, UNION, RANK, ALPHA)
    by_layer = lora_modules_by_layer(model)
    print(f"[G3-2] LoRA 包装：层并集={UNION}，覆盖 {sum(len(v) for v in by_layer.values())} 个模块")

    recs = []
    n_tr_all = []
    for arm, layers in ARMS:
        t1 = time.time()
        # ① 全部置零 → ② 只对当前臂重新随机初始化 → ③ 梯度只开当前臂
        zero_all_lora(model)
        torch.manual_seed(SEED)
        for L in layers:
            for name in by_layer.get(L, []):
                dict(find_linears(model))[name].reset_lora_parameters("default", True)
        set_grad(model, layers, by_layer)

        # 守卫：非当前臂的 ΔW 必须**精确为 0**（否则上一臂残留会污染）
        leak = 0.0
        for L, names in by_layer.items():
            if L in layers:
                continue
            for name in names:
                d = _lora_delta(dict(find_linears(model))[name])
                if d is not None:
                    leak = max(leak, float(d.abs().max()))
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_tr_all.append(n_tr)
        n_mod = sum(len(by_layer.get(L, [])) for L in layers)
        print(f"[G3-2] {arm} 层{layers} 模块数={n_mod} 可训练={n_tr/1e6:.2f}M "
              f"守卫(非本臂残留)max|ΔW|={leak:.1e}")
        assert leak == 0.0, f"{arm}: 非当前臂 ΔW 非零（{leak}）—— 换臂不干净"
        assert n_mod == 18, f"{arm}: 模块数 {n_mod} ≠ 18 —— 容量未配平（选了 full_attention 层？）"

        model.train()
        losses, ts = train_edit(model, samples, STEPS, BATCH, LR, seed=SEED,
                                log_every=0, tag=f"[{arm}] ")
        model.eval()
        bat = cap.full_battery(model, tokenizer, lm_texts)
        acq = eval_acquire(model, tokenizer, acq_items)
        ver = bat["verifiable"]["acc"]
        rec = {
            "arm": arm, "layers": layers,
            "mean_depth": float(np.mean(layers)),
            "mean_score": float(np.mean([scores[L] for L in layers])),
            "mean_conc_downproj": float(np.mean([concs[L] for L in layers])),
            "trainable_M": n_tr / 1e6,
            "final_loss": losses[-1], "train_s": ts,
            "acquire_acc": acq["acc"],
            "verifiable": ver, "forget": base_ver - ver,
            "math": bat["math"]["acc"], "format": bat["format"]["acc"],
            "code": bat["code"]["acc"],
            "lm_token_acc": bat["lm"]["token_acc"], "lm_ce": bat["lm"]["ce"],
            "delta_ce_pct": (bat["lm"]["ce"] - base_bat["lm"]["ce"])
            / base_bat["lm"]["ce"] * 100,
            "wall_s": time.time() - t1,
            "_acq_per": acq["_per"],
        }
        recs.append(rec)
        print(f"[G3-2] {arm} 完成 {rec['wall_s']:.0f}s 习得={rec['acquire_acc']:.3f} "
              f"可验证={ver:.3f} 遗忘={rec['forget']:+.3f} "
              f"ΔCE={rec['delta_ce_pct']:+.2f}% loss={rec['final_loss']:.4f}")

    # ── 阳性对照（同一进程内再确认一次仪器有效）────────────────────
    print("[G3-2] 阳性对照：破坏性 lr=1e-2 ...")
    zero_all_lora(model)
    torch.manual_seed(SEED)
    for L in ARMS[-1][1]:
        for name in by_layer.get(L, []):
            dict(find_linears(model))[name].reset_lora_parameters("default", True)
    set_grad(model, ARMS[-1][1], by_layer)
    model.train()
    lc, tsc = train_edit(model, samples, STEPS, BATCH, 1e-2, seed=SEED + 500,
                         log_every=0, tag="[ctrl] ")
    model.eval()
    cbat = cap.full_battery(model, tokenizer, lm_texts)
    ctrl = {"verifiable": cbat["verifiable"]["acc"],
            "forget": base_ver - cbat["verifiable"]["acc"],
            "lm_ce": cbat["lm"]["ce"], "final_loss": lc[-1]}

    # ── 2×2 判读 ───────────────────────────────────────────────────
    R = {r["arm"]: r for r in recs}
    pairs = {"early": ("early_high", "early_low"), "late": ("late_high", "late_low")}
    verdict = {}
    for pname, (h, l) in pairs.items():
        d = R[h]["acquire_acc"] - R[l]["acquire_acc"]
        verdict[pname] = {"high": R[h]["acquire_acc"], "low": R[l]["acquire_acc"],
                          "high_minus_low": d, "high_gt_low": d > 0,
                          "high_score": R[h]["mean_score"], "low_score": R[l]["mean_score"],
                          "high_depth": R[h]["mean_depth"], "low_depth": R[l]["mean_depth"]}

    print("\n=== 2×2 习得对照 ===")
    print(f"{'pair':<8}{'high臂':<12}{'习得':>7}{'low臂':<12}{'习得':>7}{'高−低':>8}"
          f"{'深度差':>8}{'奇点分差':>10}")
    for pname, v in verdict.items():
        print(f"{pname:<8}{pairs[pname][0]:<12}{v['high']:>7.3f}"
              f"{pairs[pname][1]:<12}{v['low']:>7.3f}{v['high_minus_low']:>+8.3f}"
              f"{v['high_depth']-v['low_depth']:>8.1f}"
              f"{v['high_score']-v['low_score']:>10.2f}")
    both = all(v["high_gt_low"] for v in verdict.values())
    print(f"\n两个 pair 都 high>low ? {both}")

    print("\n=== 逐臂 ===")
    print(f"{'arm':<12}{'depth':>6}{'score':>7}{'习得':>7}{'遗忘':>8}{'ΔCE%':>8}{'loss':>9}")
    for r in sorted(recs, key=lambda r: r["mean_depth"]):
        print(f"{r['arm']:<12}{r['mean_depth']:>6.1f}{r['mean_score']:>+7.2f}"
              f"{r['acquire_acc']:>7.3f}{r['forget']:>+8.3f}"
              f"{r['delta_ce_pct']:>+8.2f}{r['final_loss']:>9.4f}")
    print(f"\n阳性对照：可验证 {base_ver:.3f} → {ctrl['verifiable']:.3f} "
          f"(遗忘 {ctrl['forget']:+.3f}, CE {ctrl['lm_ce']:.2f}, loss={ctrl['final_loss']:.2f})")

    out = {"config": {"arms": ARMS, "union_layers": UNION, "rank": RANK, "lr": LR,
                      "steps": STEPS, "seed": SEED},
           "baseline": {"verifiable": base_ver, "acquire": base_acq["acc"],
                        "lm_ce": base_bat["lm"]["ce"],
                        "lm_token_acc": base_bat["lm"]["token_acc"]},
           "per_arm": recs, "pairs": verdict, "both_pairs_high_gt_low": both,
           "destructive_control": ctrl,
           "note": "同一进程内全部 4 臂共享基线 ⇒ 无跨进程噪声；"
                   "换臂守卫 = 非当前臂 ΔW 精确为 0"}
    with open("out/g3_depth_factor.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[G3-2] 写出 out/g3_depth_factor.json  (总 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

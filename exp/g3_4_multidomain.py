"""G3-4 热更新的两个未测问题（多域 + 剂量-反应）。

上一轮前提检验用的是**同一个域跑两遍** ⇒ 结构上测不到「域间干扰」。
本脚本回答两个"热更新"的真问题：

  协议 S —— **顺序热更新（同一适配器）**：A → B → C 连续训进同一组 LoRA 参数。
     关键读数：**先注入的 A 会不会被 B、C 覆盖**（§8 风险三「累积漂移」）。
     ⚠️ 这是**朴素热更新**（共享参数）。方案 §5.2 的 GE-PEFT（独立子模块 + 门控）
        是它的**解药**——但必须先证明病在，才值得造药。

  协议 D —— **剂量-反应**：遗忘 vs 学习率。
     已知 lr=1e-4 无遗忘、lr=1e-2（×100）全毁 ⇒ 中间是空白。
     补 3e-4 / 1e-3 两个点，得到**安全包线**，并给出"保护机制值得测"的区间。

工程：全部在一个进程内跑 ⇒ 共享基线、无跨进程噪声（已知基准噪声是跨进程的）。
"""
from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model, find_linears, _lora_delta  # noqa: E402
from s5_edit import build_lora, train_edit, render  # noqa: E402
from s1_probe import parse_linear_name  # noqa: E402
import capability as cap  # noqa: E402
from g3_1_premise import eval_acquire  # noqa: E402
import domains as D  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

LAYERS = [12, 13, 14, 15, 16, 17]
RANK, ALPHA, STEPS, BATCH, SEQ, SEED = 16, 32, 200, 2, 256, 7
LR_MAIN = 1e-4
LR_DOSE = [3e-4, 1e-3]          # 1e-4 已知无遗忘；1e-2 已知全毁 ⇒ 补中间两点
SEQ_DOMAINS = ["northwind", "halcyon", "vela"]


def encode_samples(tokenizer, pairs):
    out = []
    for q, a in pairs:
        ids = render(tokenizer, q)
        ans = tokenizer(a + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        out.append((torch.tensor((ids + ans)[:SEQ]),
                    torch.tensor(([-100] * len(ids) + ans)[:SEQ])))
    return out


def zero_and_fresh(model, seed):
    """把全部 LoRA 置零，再对 LAYERS 上的子模块做一次新的随机初始化。"""
    from s9_multiseed import reset_lora
    for mod in model.modules():
        a = getattr(mod, "lora_A", None)
        b = getattr(mod, "lora_B", None)
        if not a or not b:
            continue
        for k in a.keys():
            a[k].weight.data.zero_()
            if k in b:
                b[k].weight.data.zero_()
    torch.manual_seed(seed)
    for mod in model.modules():
        if callable(getattr(mod, "reset_lora_parameters", None)) and \
                getattr(mod, "r", None) is not None:
            try:
                mod.reset_lora_parameters("default", True)
            except Exception:
                pass
    # ★ 必须**显式白名单**放开 grad：只开 LoRA 参数。
    #   这里曾写成 `for p in model.parameters(): p.requires_grad_(True)`
    #   ⇒ 把 9.4B 基座参数全开了 ⇒ Adam 状态 2×9.4B×4B ≈ **75 GB** ⇒ CUDA OOM
    #   （在 mem_fraction=0.6 的 72.98 GiB 上限处干净报错，没触发内核 OOM）。
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_tr < 50e6, f"可训练参数 {n_tr/1e6:.1f}M 过大 —— requires_grad 白名单失效"
    # 守卫：初始化后 ΔW 必须为 0（peft 的 B 初始化为 0）
    return max((float(_lora_delta(m).abs().max())
                for m in model.modules() if _lora_delta(m) is not None), default=0.0)


def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    model, tokenizer = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-4] 加载 {time.time()-t0:.0f}s")

    lm_texts = cap.lm_texts_heldout(48, start=500)
    tr = {d: encode_samples(tokenizer, D.train_pairs(d)) for d in SEQ_DOMAINS}
    ev = {d: D.eval_items(d) for d in SEQ_DOMAINS}
    print("[G3-4] 域：" + " | ".join(f"{d}({D.n_facts(d)}事实)" for d in SEQ_DOMAINS))

    def evaluate(tag, want_battery=True):
        model.eval()
        acq = {d: eval_acquire(model, tokenizer, ev[d])["acc"] for d in SEQ_DOMAINS}
        rec = {"tag": tag, "acquire": acq}
        if want_battery:
            b = cap.full_battery(model, tokenizer, lm_texts)
            rec.update({"verifiable": b["verifiable"]["acc"],
                        "math": b["math"]["acc"], "format": b["format"]["acc"],
                        "code": b["code"]["acc"],
                        "lm_token_acc": b["lm"]["token_acc"], "lm_ce": b["lm"]["ce"]})
        print(f"[G3-4] [{tag}] " + "  ".join(f"{d}={acq[d]:.2f}" for d in SEQ_DOMAINS)
              + (f"  可验证={rec['verifiable']:.3f}" if want_battery else "")
              + f"  ({time.time()-t0:.0f}s)")
        return rec

    snaps = [evaluate("base")]

    # ── 协议 S：顺序热更新（同一适配器，不重置）──────────────────────
    import random
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = build_lora(model, LAYERS, RANK, ALPHA)
    leak = zero_and_fresh(model, SEED)
    print(f"[G3-4] LoRA r={RANK} 层{LAYERS} 初始化守卫 max|ΔW|={leak:.1e}")
    assert leak == 0.0, "初始化后 ΔW 非零 —— 换臂/初始化不干净"

    for i, d in enumerate(SEQ_DOMAINS):
        model.train()
        losses, ts = train_edit(model, tr[d], STEPS, BATCH, LR_MAIN, seed=SEED + i,
                                log_every=0, tag=f"[S:{d}] ")
        model.eval()
        print(f"[G3-4] 协议S 第{i+1}步 注入 {d}: {ts:.0f}s loss={losses[-1]:.4f}")
        snaps.append(evaluate(f"S_after_{d}"))

    # ── 协议 D：剂量-反应（每次全新适配器，只训域 A）────────────────
    dose = []
    for lr in LR_DOSE:
        zero_and_fresh(model, SEED)
        model.train()
        losses, ts = train_edit(model, tr["northwind"], STEPS, BATCH, lr,
                                seed=SEED, log_every=0, tag=f"[D:{lr:g}] ")
        model.eval()
        r = evaluate(f"D_lr{lr:g}")
        r["lr"] = lr
        r["final_loss"] = losses[-1]
        dose.append(r)

    # ── 阳性对照（同进程内再确认仪器有效）──────────────────────────
    zero_and_fresh(model, SEED)
    model.train()
    lc, _ = train_edit(model, tr["northwind"], STEPS, BATCH, 1e-2, seed=SEED + 900,
                       log_every=0, tag="[ctrl] ")
    ctrl = evaluate("destructive_ctrl_lr1e-2")
    ctrl["lr"] = 1e-2
    ctrl["final_loss"] = lc[-1]

    # ── 判读 ────────────────────────────────────────────────────────
    base_ver = snaps[0]["verifiable"]
    base_acq = snaps[0]["acquire"]
    S = {s["tag"]: s for s in snaps}
    aA_first = S["S_after_northwind"]["acquire"]["northwind"]
    aA_last = S["S_after_vela"]["acquire"]["northwind"]
    retention = aA_last / aA_first if aA_first > 0 else float("nan")

    print("\n=== 协议S 干扰矩阵（习得率）===")
    hdr = "阶段".ljust(22) + "".join(f"{d:>12}" for d in SEQ_DOMAINS)
    print(hdr)
    for s in snaps:
        print(s["tag"].ljust(22)
              + "".join(f"{s['acquire'][d]:>12.3f}" for d in SEQ_DOMAINS))

    print(f"\nA(northwind) 首次注入后={aA_first:.3f} → 注入 B、C 之后={aA_last:.3f}"
          f"  保留率={retention:.2f}")
    print(f"可验证能力：base={base_ver:.3f} → S最终="
          f"{S['S_after_vela']['verifiable']:.3f} "
          f"(遗忘 {base_ver - S['S_after_vela']['verifiable']:+.3f})")

    print("\n=== 协议D 剂量-反应（遗忘 vs lr）===")
    print(f"{'lr':>8}{'可验证':>9}{'遗忘':>9}{'ΔCE%':>9}{'习得A':>8}{'final loss':>12}")
    ref_ce = snaps[0]["lm_ce"]
    # lr=1e-4 的参考点直接用协议S 第 1 步（同样是"全新适配器 + 只训 northwind + 1e-4"）
    s1 = S["S_after_northwind"]
    print(f"{'1e-4':>8}{s1['verifiable']:>9.3f}"
          f"{base_ver - s1['verifiable']:>9.3f}"
          f"{(s1['lm_ce'] - ref_ce) / ref_ce * 100:>9.2f}"
          f"{s1['acquire']['northwind']:>8.2f}{'(协议S)'!s:>12}")
    for r in dose + [ctrl]:
        print(f"{r['lr']:>8g}{r['verifiable']:>9.3f}"
              f"{base_ver - r['verifiable']:>9.3f}"
              f"{(r['lm_ce'] - ref_ce) / ref_ce * 100:>9.2f}"
              f"{r['acquire']['northwind']:>8.2f}{r['final_loss']:>12.4f}")

    out = {"config": {"seq_domains": SEQ_DOMAINS, "layers": LAYERS, "rank": RANK,
                      "lr_main": LR_MAIN, "dose_lrs": LR_DOSE, "steps": STEPS,
                      "seed": SEED},
           "baseline": snaps[0], "protocol_S": snaps,
           "protocol_D": dose, "destructive_control": ctrl,
           "S_retention_northwind_after_B_C": retention,
           "S_capability_forget": base_ver - S["S_after_vela"]["verifiable"]}
    with open("out/g3_multidomain.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[G3-4] 写出 out/g3_multidomain.json  (总 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

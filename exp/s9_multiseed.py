"""S9 跨种子重复（补统计功效缺口）。

为什么必须做：本项目实测**训练口径的分辨率 ≈1%**（CUDA kernel 非确定性 + dropout +
LoRA 随机初始化），而编辑效应本身也只有 ~1% 量级 ⇒ **单种子无法分辨**。
本脚本对同一编辑流程换 5 个种子独立重复，给出效应量相对于种子间噪声的比值。

设计（三个关键点）：
  1. **只加载一次模型**：peft 0.20 的 `PeftModel` **没有 `unload()`**，
     但未合并 LoRA 时**基座权重从不变动** ⇒ 换种子只需把 LoRA 参数**重置**为新随机初值
     （peft 官方 `LoraLayer.reset_lora_parameters`），无需重载 18GB 模型。
  2. **重置守卫**：每次重置后，重跑基线剖面并断言与 `baseline` **逐位相同**——
     否则种子 k 的残留适配器会污染种子 k+1（即"关掉差异项后两者应相同"的等价性测试）。
  3. **两个口径**：守卫用廉价口径（32×128，~5s）；漂移读数和基线同用完整口径
     （256×512，与 s1_probe 一致）⇒ 漂移数字可与单次实验的 −1.26% 直接对比。

输出：out/multiseed.json + 终端表格。
"""
from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    MODEL_DIR, CALIB_PATH, load_model, load_calib_texts,
    find_linears, ChannelActivationAccumulator,
)
from s1_probe import parse_linear_name, top_share  # noqa: E402
from s5_edit import (  # noqa: E402
    build_lora, build_samples, train_edit, eval_all,
)

import torch  # noqa: E402

LAYERS = [12, 13, 14, 15, 16, 17]
RANK, ALPHA, LR, STEPS, BATCH, SEQ = 16, 32, 1e-4, 200, 2, 256
N_CALIB_FULL, L_CALIB_FULL = 256, 512
N_CALIB_GUARD, L_CALIB_GUARD = 32, 128


# ══════════════════════════════════════════════════════════════════════
# 轻量剖面：只测 mlp.down_proj 的 top-1% 激活集中度
# ══════════════════════════════════════════════════════════════════════

def make_calib(tokenizer, n, seq_len):
    texts = load_calib_texts(CALIB_PATH, n)
    out = []
    for t in texts:
        e = tokenizer(t, truncation=True, max_length=seq_len)["input_ids"]
        if len(e) < 8:
            continue
        out.append(torch.tensor([e], dtype=torch.long))
    return out


@torch.no_grad()
def probe_concentration(model, batches):
    """逐层 mlp.down_proj 输入的 top-1% 通道能量占比。"""
    keep = {k: v for k, v in find_linears(model).items()
            if k.endswith("mlp.down_proj")}
    acc = ChannelActivationAccumulator(keep)
    try:
        for ids in batches:
            model(input_ids=ids.cuda())
    finally:
        acc.remove()
    out = {}
    for name, r in acc.result().items():
        layer, _ = parse_linear_name(name)
        if layer >= 0:
            out[layer] = top_share(r["mean_abs"], 0.01)
    return out


def reset_lora(model, init_lora_weights=True):
    """peft 官方重置：把所有 LoRA 层的 A/B 重新随机初始化（B=0 ⇒ 等效于基座）。"""
    n = 0
    for mod in model.modules():
        fn = getattr(mod, "reset_lora_parameters", None)
        if callable(fn):
            try:
                fn("default", init_lora_weights)
                n += 1
            except Exception as exc:  # noqa: BLE001
                print(f"      [warn] reset 失败 {type(mod).__name__}: {exc}")
    return n


def seed_all(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def max_abs_diff(a, b):
    keys = set(a) & set(b)
    if set(a) != set(b):
        return float("inf")
    return max((abs(a[k] - b[k]) for k in keys), default=0.0)


def main():
    seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                              else "0,1,2,3,4".split(","))]
    outdir = "out"
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    model, tokenizer = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[S9] 加载 {time.time()-t0:.0f}s")

    calib_full = make_calib(tokenizer, N_CALIB_FULL, L_CALIB_FULL)
    calib_guard = make_calib(tokenizer, N_CALIB_GUARD, L_CALIB_GUARD)
    print(f"[S9] 口径：漂移 {len(calib_full)}×{L_CALIB_FULL}，"
          f"守卫 {len(calib_guard)}×{L_CALIB_GUARD}")

    reg_texts = load_calib_texts(CALIB_PATH, 48, start=400)
    assert len(reg_texts) >= 16, "回归语料不足"

    print("[S9] 基线评估（一次，所有种子共用）...")
    base_eval = eval_all(model, tokenizer, reg_texts, tag="base")
    baseline = probe_concentration(model, calib_full)
    identity_probe = probe_concentration(model, calib_guard)
    print(f"[S9] base: 身份命中={base_eval['identity_hit_rate']:.2f} "
          f"CE={base_eval['regression_ce']:.4f} 溢出={base_eval['overflow_rate']:.2f}")

    # ★ 必须把返回值赋回 model：`get_peft_model` 返回**新的 PeftModel**，
    #   原 model 引用仍是未包装的基座（不赋值 ⇒ 后续 train/eval 全作用在错误对象上）。
    model = build_lora(model, LAYERS, RANK, ALPHA)
    peft_cfg = model.peft_config["default"]
    init_lw = getattr(peft_cfg, "init_lora_weights", True)
    model.eval()
    after_wrap = probe_concentration(model, calib_guard)
    d_wrap = max_abs_diff(after_wrap, identity_probe)
    print(f"[S9] 守卫 A（包装后 ≡ 基座，B=0）：max|Δ| = {d_wrap:.3e}")
    assert d_wrap == 0.0, "build_lora 后剖面变了 —— LoRA 初始 B 非零？"

    samples = build_samples(tokenizer, SEQ)
    recs = []
    for s in seeds:
        t1 = time.time()
        seed_all(s)
        n_reset = reset_lora(model, init_lw)

        # 守卫 B：重置后必须与基座**逐位相同**
        model.eval()
        g = probe_concentration(model, calib_guard)
        d = max_abs_diff(g, identity_probe)
        assert d == 0.0, (f"seed {s} 重置后剖面≠基座（max|Δ|={d:.3e}）—— "
                          f"上一个种子的适配器残留了！")
        print(f"[S9] seed {s}: 重置 {n_reset} 层，守卫 B max|Δ|={d:.1e} ✅")

        model.train()
        losses, ts = train_edit(model, samples, STEPS, BATCH, LR,
                               seed=s, log_every=0, tag=f"[seed{s}] ")
        model.eval()
        ev = eval_all(model, tokenizer, reg_texts, tag=f"seed{s}")
        ev["delta_ce_pct"] = (ev["regression_ce"] - base_eval["regression_ce"]) \
            / base_eval["regression_ce"] * 100
        post = probe_concentration(model, calib_full)

        edited = {L: (post[L] - baseline[L]) / baseline[L] * 100 for L in LAYERS}
        other = {L: (post[L] - baseline[L]) / baseline[L] * 100
                 for L in baseline if L not in LAYERS}
        me = sum(edited.values()) / len(edited)
        mo = sum(other.values()) / len(other)
        rec = {
            "seed": s,
            "final_loss": losses[-1],
            "train_s": ts,
            "identity_hit_rate": ev["identity_hit_rate"],
            "regression_ce": ev["regression_ce"],
            "delta_ce_pct": ev["delta_ce_pct"],
            "overflow_rate": ev["overflow_rate"],
            "drift_edited_mean_pct": me,
            "drift_other_mean_pct": mo,
            "drift_gap_pp": me - mo,
            "edited_all_negative": all(v < 0 for v in edited.values()),
            "per_layer_edited_pct": {str(k): v for k, v in edited.items()},
        }
        recs.append(rec)
        print(f"[S9] seed {s} 完成 {time.time()-t1:.0f}s  "
              f"命中={rec['identity_hit_rate']:.2f} ΔCE={rec['delta_ce_pct']:+.2f}% "
              f"编辑层漂移={me:+.2f}% 其余={mo:+.2f}% gap={me-mo:+.2f}pp")

    # ── 聚合 ────────────────────────────────────────────────────────
    def stat(key):
        v = [r[key] for r in recs]
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
        return m, sd, v

    agg = {}
    for k in ("identity_hit_rate", "delta_ce_pct", "overflow_rate",
              "drift_edited_mean_pct", "drift_other_mean_pct", "drift_gap_pp"):
        m, sd, v = stat(k)
        agg[k] = {"mean": m, "sd": sd, "values": v,
                  "se": sd / (len(v) ** 0.5)}
    g = agg["drift_gap_pp"]
    t_stat = g["mean"] / g["se"] if g["se"] > 0 else float("inf")
    agg["drift_gap_t"] = t_stat
    agg["n_seeds"] = len(recs)
    agg["n_gap_negative"] = sum(1 for r in recs if r["drift_gap_pp"] < 0)
    agg["n_edited_all_negative"] = sum(1 for r in recs if r["edited_all_negative"])
    agg["config"] = {"layers": LAYERS, "rank": RANK, "alpha": ALPHA, "lr": LR,
                     "steps": STEPS, "batch": BATCH, "seq_len": SEQ,
                     "n_calib_full": N_CALIB_FULL, "l_calib_full": L_CALIB_FULL,
                     "base_ce": base_eval["regression_ce"]}

    with open(os.path.join(outdir, "multiseed.json"), "w", encoding="utf-8") as f:
        json.dump({"per_seed": recs, "aggregate": agg,
                   "base_eval": {k: base_eval[k] for k in
                                 ("identity_hit_rate", "regression_ce",
                                  "overflow_rate")}},
                  f, ensure_ascii=False, indent=2)

    # ── 打印 ────────────────────────────────────────────────────────
    print("\n=== 跨种子汇总 ===")
    print(f"{'指标':<28}{'mean':>10}{'sd':>10}{'se':>10}")
    for k in ("identity_hit_rate", "delta_ce_pct", "overflow_rate",
              "drift_edited_mean_pct", "drift_other_mean_pct", "drift_gap_pp"):
        a = agg[k]
        print(f"{k:<28}{a['mean']:>10.3f}{a['sd']:>10.3f}{a['se']:>10.3f}")
    print(f"\n编辑层−其余 的配对差 t = {t_stat:.2f} (n={len(recs)})")
    print(f"逐种子 gap<0 的个数：{agg['n_gap_negative']}/{len(recs)}")
    print(f"逐种子「6 层全部负向」的个数：{agg['n_edited_all_negative']}/{len(recs)}")
    print(f"[S9] 写出 {outdir}/multiseed.json  (总 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

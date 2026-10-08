"""S7 守卫：验证 `find_linears` 的「LoRA 包装层可见性」修复。

必须同时满足（任一条失败即报错退出，绝不静默）：
  G1 **基座不变性**：无 adapter 时，`find_linears` 仍返回 248 个逻辑线性层，
     且 `effective_weight(m)` 与 `m.weight` **逐位相同**（ΔW=0）。
  G2 **包装可见性**：挂 adapter 后，逻辑线性层数仍为 248
     （LoRA 只**包装**、不新增逻辑层），且**恰好**目标层集合的 ΔW≠0。
  G3 **目标层三件套**：每个被编辑层都能被找到，且 ΔW 非零。

背景（为什么需要这个守卫）：`peft.tuners.lora.Linear` 不是 `nn.Linear` 子类，
旧版 `isinstance(mod, nn.Linear)` 会**静默漏掉**被编辑层 —— 表现为编辑后探测里
那几层变 NaN（本实验真实踩过）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    MODEL_DIR, load_model, get_text_backbone, find_linears,
    effective_weight, _lora_delta,
)
import torch  # noqa: E402

ADAPTER = "out/lora_selfcog"
EXPECTED_LAYERS = {12, 13, 14, 15, 16, 17}
N_LOGICAL = 248


def layers_with_delta(linears):
    out = {}
    for name, mod in linears.items():
        d = _lora_delta(mod)
        if d is None:
            continue
        out[name] = float(d.abs().max())
    return out


def main():
    model, tok = load_model(MODEL_DIR, precision="bf16")
    bb = get_text_backbone(model)

    # ── G1 基座不变性 ────────────────────────────────────────────────
    base_linears = find_linears(model)
    print(f"[S7] G1 基座逻辑线性层数 = {len(base_linears)} (期望 {N_LOGICAL})")
    assert len(base_linears) == N_LOGICAL, "基座逻辑线性层数变了 —— find_linears 回归"
    max_dev = 0.0
    for name, mod in base_linears.items():
        Weff = effective_weight(mod)
        max_dev = max(max_dev, float((Weff.float() - mod.weight.float()).abs().max()))
    print(f"[S7] G1 effective_weight vs weight 最大偏差 = {max_dev:.3e}")
    assert max_dev == 0.0, f"基座上 ΔW 非零（{max_dev}）—— effective_weight 写错了"

    # ── G2/G3 包装可见性 ────────────────────────────────────────────
    from peft import PeftModel
    pm = PeftModel.from_pretrained(model, ADAPTER)
    pm.eval()
    bb2 = get_text_backbone(pm)
    assert len(bb2.layers) == len(bb.layers), "主干层数变了"

    lin2 = find_linears(pm)
    print(f"[S7] G2 adapter 逻辑线性层数 = {len(lin2)} (期望 {N_LOGICAL})")
    assert len(lin2) == N_LOGICAL, (
        f"adapter 下逻辑层数 {len(lin2)} ≠ {N_LOGICAL} —— 去重逻辑或包装识别有误")

    deltas = layers_with_delta(lin2)
    got_layers = sorted({
        int(n.split(".")[1]) for n in deltas if n.startswith("layers.")
    })
    print(f"[S7] G2 ΔW≠0 的模块数 = {len(deltas)}，涉及层 = {got_layers}")
    assert set(got_layers) == EXPECTED_LAYERS, (
        f"ΔW≠0 的层集合 {got_layers} ≠ 期望 {sorted(EXPECTED_LAYERS)}")

    print(f"[S7] G3 各层 max|ΔW| 示例："
          f"{ {k: round(v, 4) for k, v in list(deltas.items())[:3]} }")
    print("[S7] ✅ 全部守卫通过：编辑层已可被探针看见")


if __name__ == "__main__":
    main()

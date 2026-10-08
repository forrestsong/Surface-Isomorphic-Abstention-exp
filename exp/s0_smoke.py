"""S0 冒烟：验证 Qwen3.5-9B 能加载、前向、hook 采集激活。

不写任何结论，只回答：模型结构长什么样、名字是什么、激活 hook 通不通、显存多少。
输出 JSON 到 exp/out/s0_smoke.json。
"""
from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    MODEL_DIR, CALIB_PATH, load_model, get_text_backbone, layer_types,
    find_linears, ChannelActivationAccumulator, load_calib_texts, encode_batch,
)
import torch  # noqa: E402


def main():
    t0 = time.time()
    print(f"[S0] 加载 {MODEL_DIR} ...")
    model, tokenizer = load_model(MODEL_DIR, precision="bf16")
    print(f"[S0] 加载完成 {time.time()-t0:.1f}s")

    bb = get_text_backbone(model)
    lt = layer_types(model)
    n_params = sum(p.numel() for p in model.parameters())
    linears = find_linears(model)

    print(f"[S0] 文本主干层数 = {len(bb.layers)}")
    print(f"[S0] layer_types 唯一值 = {sorted(set(lt))}  "
          f"full_attention 层 = {[i for i, t in enumerate(lt) if t == 'full_attention']}")
    print(f"[S0] 参数总量 = {n_params/1e9:.3f} B")
    print(f"[S0] 文本主干内 Linear 数 = {len(linears)}")
    print("[S0] 前 20 个 Linear 名：")
    for nm in list(linears.keys())[:20]:
        print("      ", nm, tuple(linears[nm].weight.shape))

    # 打印第 0 层所有子模块名（理解 DeltaNet / attention / MLP 结构）
    print("[S0] 层 0 子模块：")
    for nm, mod in bb.layers[0].named_modules():
        if nm:
            print("      ", nm, type(mod).__name__)

    # 前向 + hook
    texts = load_calib_texts(CALIB_PATH, 4)
    print(f"[S0] 标定文本条数 = {len(texts)}  例：{texts[0][:60]!r}")
    batch = encode_batch(tokenizer, texts, max_seq_len=256)

    # 只 hook 一部分，冒烟足够
    subset = {k: v for k, v in list(linears.items())[:6]}
    acc = ChannelActivationAccumulator(subset)
    t1 = time.time()
    with torch.no_grad():
        out = model(**batch)
    print(f"[S0] 前向完成 {time.time()-t1:.1f}s  logits={tuple(out.logits.shape)}")
    acc.remove()
    res = acc.result()
    for nm, r in res.items():
        print(f"      {nm}: n_tok={r['n_tokens']:.0f} mean|x| μ={r['mean_abs'].mean():.4f} "
              f"max={r['max_abs'].max():.3f}")

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else -1
    print(f"[S0] peak torch allocated = {peak:.1f} GiB")

    os.makedirs("out", exist_ok=True)
    with open("out/s0_smoke.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_layers": len(bb.layers),
            "layer_types": lt,
            "n_params_B": n_params / 1e9,
            "n_linears": len(linears),
            "linear_names": list(linears.keys()),
            "peak_gib": peak,
        }, f, ensure_ascii=False, indent=2)
    print("[S0] 写出 out/s0_smoke.json")


if __name__ == "__main__":
    main()

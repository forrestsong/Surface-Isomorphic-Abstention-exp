"""S3 LLC 精定位（devinterp，localized）。

tech plan §3.3 要求对每个层块估计 LLC。全模型 SGLD 在 9B 上不可行（9.4B 参数 × 优化器态）。
本脚本用 devinterp 的 `param_masks` 把 SGLD **限制在候选层块**内（其余冻结），
得到 **局部 LLC** —— 与 tech plan 的「DDS 全局扫描 → LLC 局部精定位」一致。

⚠ 诚实边界（必须写进报告）：
  * 这是 **localized LLC**（仅该层块参数自由），不是全模型 LLC，数值不可与论文基数直接比。
  * lr/nbeta 需标定；未标定的 LLC 读数不可解释。`--probe` 先扫 lr 看 loss 轨迹。

用法：
  python s3_llc.py --probe                 # 标定：只跑 L31，扫 lr，看轨迹与步时
  python s3_llc.py --layers 31,26,16       # 正式：三层各一份 LLC
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, CALIB_PATH, load_model, get_text_backbone, load_calib_texts  # noqa: E402
import torch  # noqa: E402


def backbone_prefix(model, prefix_holder):
    for name, mod in model.named_modules():
        if mod is prefix_holder[0]:
            return name
    return ""


def build_dataset(tokenizer, n_seq: int, seq_len: int):
    """返回 torch Dataset（devinterp 内部直接用 DataLoader + 默认 collate，
    默认 collate 只对**张量**做 stack ⇒ 必须返回 tensor，返回 list 会 AttributeError）。"""
    class IdsDS(torch.utils.data.Dataset):
        def __init__(self, ids):
            self.ids = ids

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            return {"input_ids": torch.tensor(self.ids[i], dtype=torch.long)}

    texts = load_calib_texts(CALIB_PATH, n_seq)
    pad_id = tokenizer.pad_token_id
    ids = []
    for t in texts:
        e = tokenizer(t, truncation=True, max_length=seq_len)["input_ids"]
        if len(e) < 8:
            continue
        # 补齐到定长（默认 collate 要求等长）
        e = e + [pad_id] * (seq_len - len(e))
        ids.append(e)
    return IdsDS(ids), len(ids)


def params_for_layer(model, prefix: str, layer: int, include_mlp=True,
                     include_attn=True):
    masks = {}
    sub = f"{prefix}.layers.{layer}."
    for name, p in model.named_parameters():
        if not name.startswith(sub):
            continue
        leaf = name[len(sub):]
        is_mlp = leaf.startswith("mlp.")
        is_attn = (leaf.startswith("linear_attn.") or leaf.startswith("self_attn."))
        if is_mlp and not include_mlp:
            continue
        if is_attn and not include_attn:
            continue
        # ★ 必须显式打开 requires_grad：devinterp 只对**不在** masks 里的参数
        #   调用 requires_grad_(False)，不会把 masks 内的参数打开。
        #   而我们的 load_model 把所有参数冻了 ⇒ 不打开则 loss 无 grad_fn，
        #   报 "element 0 of tensors does not require grad"。
        p.requires_grad_(True)
        masks[name] = None          # None = 全参数参与采样
    return masks


def run_llc(model, ds, masks, lr, nbeta, num_draws, burnin, steps_per_draw,
            batch_size, tag, outdir):
    from devinterp.slt.llc import llc
    t0 = time.time()
    res = llc(
        model=model, dataset=ds, observables={},
        lr=lr, n_beta=nbeta,
        param_masks=masks,
        num_chains=1,
        num_draws=num_draws,
        num_burnin_steps=burnin,
        num_steps_bw_draws=steps_per_draw,
        batch_size=batch_size,
        init_seed=100,
        loss_fn=None,
    )
    dt = time.time() - t0
    out = {
        "tag": tag,
        "lr": lr, "nbeta": nbeta,
        "num_draws": num_draws, "burnin": burnin, "steps_per_draw": steps_per_draw,
        "batch_size": batch_size,
        "n_params_sampled": int(sum(p.numel() for n, p in model.named_parameters()
                                    if n in masks)),
        "llc_mean": float(res["llc_mean"].values),
        "llc_std": float(res["llc_std"].values),
        "init_loss": float(res["init_loss"].values),
        "loss_trace_head": [float(x) for x in res["loss_trace"].values.flatten()[:40]],
        "loss_trace_tail": [float(x) for x in res["loss_trace"].values.flatten()[-40:]],
        "elapsed_s": dt,
    }
    print(f"[LLC] {tag}: llc={out['llc_mean']:.4f}±{out['llc_std']:.4f} "
          f"init_loss={out['init_loss']:.4f} {dt:.0f}s")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"llc_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="31,26,16")
    ap.add_argument("--lrs", default="1e-4,1e-5",
                    help="LLC 对 lr 敏感（实测 1e-5→1.5 / 1e-4→23 / 1e-3→42）。"
                         "跨层比较必须用同一 lr；多 lr 只是稳健性检查。")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--n_seq", type=int, default=512)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_draws", type=int, default=100)
    ap.add_argument("--burnin", type=int, default=50)
    ap.add_argument("--steps_per_draw", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    model, tokenizer = load_model(MODEL_DIR, precision="bf16")
    bb = get_text_backbone(model)
    prefix = backbone_prefix(model, [bb])
    print(f"[LLC] backbone prefix = {prefix!r}")
    ds, n = build_dataset(tokenizer, args.n_seq, args.seq_len)
    print(f"[LLC] dataset = {n} 条 × {args.seq_len} tok")
    nbeta = args.batch_size / max(1.0, __import__("math").log(args.batch_size))
    print(f"[LLC] nbeta = {nbeta:.4f} (default_nbeta 约定)")

    if args.probe:
        masks = params_for_layer(model, prefix, 31)
        print(f"[LLC-probe] L31 采样参数 = {sum(p.numel() for n,p in model.named_parameters() if n in masks)/1e6:.1f} M")
        for lr in (1e-5, 1e-4, 1e-3):
            run_llc(model, ds, masks, lr, nbeta, num_draws=8, burnin=2,
                    steps_per_draw=2, batch_size=args.batch_size,
                    tag=f"probe_L31_lr{lr:g}", outdir=os.path.join(args.out, "llc_probe"))
        return

    results = []
    for lr in [float(x) for x in args.lrs.split(",")]:
        for L in [int(x) for x in args.layers.split(",")]:
            masks = params_for_layer(model, prefix, L)
            results.append(run_llc(
                model, ds, masks, lr, nbeta,
                num_draws=args.num_draws, burnin=args.burnin,
                steps_per_draw=args.steps_per_draw, batch_size=args.batch_size,
                tag=f"L{L}_lr{lr:g}", outdir=args.out))
    with open(os.path.join(args.out, "llc_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

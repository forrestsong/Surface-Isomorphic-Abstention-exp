"""G3-5 测 §5.2 的**解法**是否有效：参数隔离（每域独立子模块）vs 共享适配器。

背景（G3-4 已证病在）：三个互不重叠的域顺序注入**同一个** r=16 适配器后，
先注入者保留率仅 **0.23 / 0.43**（域间覆盖），而预训练能力全程不变。

本脚本对比两种方案（同样数据、同样顺序、同样步数）：
  Arm **I**    —— 每域一个独立子模块（domA/domB/domC，各 r=16）；
                  训 B 时 A 的子模块**完全冻结且不参与**。§5.2 的核心主张。
  Arm **S48**  —— **容量配平对照**：单个共享适配器 r=48 顺序训练。
                  ★ 为什么必须有它：3×r16 与 1×r48 的参数量**恰好相等**
                  （每模块 r(in+out)，3·16 = 48），所以若 I 赢而 S48 输，
                  赢的是「隔离」而不是「参数更多」。没有这条对照，
                  「隔离有效」与「容量更大」无法区分。

读数：
  R1 **隔离保留率**：训完 B、C 后，只激活 domA 时 A 的习得 / A 刚训完时的习得。
  R2 **组合可用性**：同时激活三个子模块（ΔW 相加）时，三个域是否都还好。
  R3 **预训练能力**：组合状态下 math/format/code 是否受损。

工程：peft 0.20 **没有** `add_weighted_adapter`，且 `set_adapter` 明确
「Only one adapter can be active at a time」⇒ 多子模块**组合**必须自己实现：
用一个 forward hook 把非激活子模块的 ΔW·x 加到输出上（精确、不改权重）。
"""
from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model  # noqa: E402
from s5_edit import train_edit, render  # noqa: E402
import capability as cap  # noqa: E402
from g3_1_premise import eval_acquire  # noqa: E402
import domains as D  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

LAYERS = [12, 13, 14, 15, 16, 17]
TARGETS = ["gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "out_proj"]
R16, A16, R48, A48 = 16, 32, 48, 96      # 两臂 alpha/r 都 = 2
STEPS, BATCH, SEQ, SEED = 200, 2, 256, 7
DOMS = ["northwind", "halcyon", "vela"]
SUB = {"northwind": "domA", "halcyon": "domB", "vela": "domC"}


def lora_mods(model):
    out = {}
    for n, m in model.named_modules():
        if hasattr(m, "lora_A") and len(getattr(m, "lora_A", {})) > 0:
            out[n] = m
    return out


def set_trainable(model, adapter, enable=True):
    """★ 显式白名单：只放开指定适配器的 LoRA 参数（见 G3-4 的 75GB OOM 教训）。"""
    for n, p in model.named_parameters():
        p.requires_grad_(enable and ("lora_A." + adapter in n
                                     or "lora_B." + adapter in n))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_tr


def adapter_delta_max(model, adapter):
    m = 0.0
    for mod in lora_mods(model).values():
        if adapter not in mod.lora_A:
            continue
        A = mod.lora_A[adapter].weight
        B = mod.lora_B[adapter].weight
        m = max(m, float((B.float() @ A.float()).abs().max()))
    return m


def snapshot_adapter(model, adapter):
    """拍下某子模块的 LoRA 参数（很小：r(in+out)）用于**逐位**比对。"""
    snap = {}
    for n, mod in lora_mods(model).items():
        if adapter in mod.lora_A:
            snap[n + ".A"] = mod.lora_A[adapter].weight.detach().clone()
            snap[n + ".B"] = mod.lora_B[adapter].weight.detach().clone()
    return snap


def snapshot_diff(before, model, adapter):
    """返回与快照的最大逐位偏差（应为 **0.0**）与是否有形状变化。"""
    after = snapshot_adapter(model, adapter)
    if set(before) != set(after):
        return float("inf")
    d = 0.0
    for k in before:
        if before[k].shape != after[k].shape:
            return float("inf")
        d = max(d, float((before[k].float() - after[k].float()).abs().max()))
    return d


class ComposeExtras:
    """把**非激活**子模块的 ΔW·x 加到输出上 ⇒ 实现「多子模块组合」。"""

    def __init__(self, model, extras):
        self.extras = list(extras)
        self.handles = []
        for mod in lora_mods(model).values():
            if any(e in mod.lora_A for e in self.extras):
                self.handles.append(mod.register_forward_hook(self._mk(mod)))

    def _mk(self, mod):
        extras = [e for e in self.extras if e in mod.lora_A]

        def hook(module, inputs, output):
            x = inputs[0]
            if x is None:
                return None
            add = None
            for e in extras:
                A = module.lora_A[e].weight
                B = module.lora_B[e].weight
                s = float(module.scaling.get(e, 1.0))
                d = (x.float() @ A.float().t()) @ B.float().t() * s
                add = d if add is None else add + d
            if add is None:
                return output
            return output + add.to(output.dtype)

        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    model, tokenizer = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-5] 加载 {time.time()-t0:.0f}s")

    lm_texts = cap.lm_texts_heldout(48, start=500)
    ev = {d: D.eval_items(d) for d in DOMS}
    tr = {}
    for d in DOMS:
        s = []
        for q, a in D.train_pairs(d):
            ids = render(tokenizer, q)
            ans = tokenizer(a + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            s.append((torch.tensor((ids + ans)[:SEQ]),
                      torch.tensor(([-100] * len(ids) + ans)[:SEQ])))
        tr[d] = s

    # ── 建立多适配器 ────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model
    def mk_cfg(r, alpha):
        return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM", target_modules=TARGETS,
                          layers_pattern="layers", layers_to_transform=LAYERS)
    import random
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = get_peft_model(model, mk_cfg(R16, A16), adapter_name="domA")
    model.add_adapter("domB", mk_cfg(R16, A16))
    model.add_adapter("domC", mk_cfg(R16, A16))
    model.add_adapter("shared48", mk_cfg(R48, A48))
    mods = lora_mods(model)
    print(f"[G3-5] LoRA 模块={len(mods)}  适配器={sorted(list(next(iter(mods.values())).lora_A.keys()))}")
    # 断言：3×r16 与 1×r48 参数量相等（容量配平的依据）
    n16 = sum(3 * (mod.lora_A["domA"].weight.numel() + mod.lora_B["domA"].weight.numel())
              for mod in mods.values())
    n48 = sum(mod.lora_A["shared48"].weight.numel() + mod.lora_B["shared48"].weight.numel()
              for mod in mods.values())
    print(f"[G3-5] 容量配平检查：3×r16={n16/1e6:.3f}M  1×r48={n48/1e6:.3f}M")
    assert n16 == n48, f"容量未配平：{n16} vs {n48}"

    def evaluate(tag, adapter=None, compose=None, battery=True):
        model.eval()
        if compose:
            set_trainable(model, "domA", False)
            model.set_adapter(compose[0])
            h = ComposeExtras(model, compose[1:])
        else:
            model.set_adapter(adapter)
            h = None
        acq = {d: eval_acquire(model, tokenizer, ev[d])["acc"] for d in DOMS}
        rec = {"tag": tag, "acquire": acq}
        if battery:
            b = cap.full_battery(model, tokenizer, lm_texts)
            rec.update({"verifiable": b["verifiable"]["acc"],
                        "lm_ce": b["lm"]["ce"], "lm_token_acc": b["lm"]["token_acc"]})
        if h:
            h.remove()
        print(f"[G3-5] [{tag}] " + " ".join(f"{d}={acq[d]:.2f}" for d in DOMS)
              + (f" 可验证={rec['verifiable']:.3f}" if battery else "")
              + f" ({time.time()-t0:.0f}s)")
        return rec

    base = evaluate("base", adapter="domA")
    base_ver, base_ce = base["verifiable"], base["lm_ce"]

    # ══ Arm I：参数隔离 ═══════════════════════════════════════════
    print("\n[G3-5] === Arm I 参数隔离（每域独立子模块，各 r=16）===")
    armI = []
    for d in DOMS:
        a = SUB[d]
        # ★ 隔离的**硬证据**：先拍下其他（已训过的）子模块，训完再比，必须逐位不变。
        snaps = {o: snapshot_adapter(model, SUB[o]) for o in DOMS if SUB[o] != a}
        n_tr = set_trainable(model, a)
        model.set_adapter(a)
        model.train()
        losses, ts = train_edit(model, tr[d], STEPS, BATCH, 1e-4, seed=SEED,
                                log_every=0, tag=f"[I:{a}] ")
        model.eval()
        diffs = {o: snapshot_diff(snaps[o], model, SUB[o]) for o in snaps}
        print(f"[G3-5] I 训练 {a}: {ts:.0f}s loss={losses[-1]:.4f} 可训练={n_tr/1e6:.2f}M "
              f"其他子模块逐位偏差={diffs if diffs else '{}'}")
        assert all(v == 0.0 for v in diffs.values()), \
            f"训练 {a} 时其他子模块被改动了：{diffs} —— 隔离失效"
        # 训完立即测「只激活本域」的习得（作为该域自己的上限）
        rec = evaluate(f"I_after_{d}", adapter=a, battery=False)
        rec["final_loss"] = losses[-1]
        armI.append(rec)

    # 隔离检验：全部训完后，只激活 domA / domB / domC
    iso = {}
    for d in DOMS:
        iso[d] = evaluate(f"I_alone_{d}", adapter=SUB[d], battery=False)["acquire"]
    composed = evaluate("I_composed", compose=[SUB[d] for d in DOMS])

    # ══ Arm S48：容量配平对照（单共享适配器 r=48，顺序训练）══════
    print("\n[G3-5] === Arm S48 单共享适配器 r=48（容量与 3×r16 相等）===")
    set_trainable(model, "shared48")
    model.set_adapter("shared48")
    armS = []
    for d in DOMS:
        model.train()
        losses, ts = train_edit(model, tr[d], STEPS, BATCH, 1e-4, seed=SEED,
                                log_every=0, tag=f"[S48:{d}] ")
        model.eval()
        print(f"[G3-5] S48 注入 {d}: {ts:.0f}s loss={losses[-1]:.4f}")
        rec = evaluate(f"S48_after_{d}", adapter="shared48")
        rec["final_loss"] = losses[-1]
        armS.append(rec)

    # ══ 判读 ══════════════════════════════════════════════════════
    def retention(recs, dom):
        """共享适配器臂：该域在「刚训完」与「全部训完」之间的保留率。
        ⚠️ 只适用于**始终只有一个适配器**的臂（S48）——那时每次读数都是同一适配器激活。"""
        idx = DOMS.index(dom)
        first = recs[idx]["acquire"][dom]
        last = recs[-1]["acquire"][dom]
        return first, last, (last / first if first > 0 else float("nan"))

    # ── ★ Arm I 的保留率必须用「只激活本域」的读数 ────────────────────
    #   上次我用 recs[-1]（那是只激活 domC 时测的）⇒ northwind 显示 0.000，
    #   把「适配器没激活」误报成「遗忘了」。这是本实验自己踩过的口径错误。
    def retention_I(dom):
        idx = DOMS.index(dom)
        first = armI[idx]["acquire"][dom]
        last = iso[dom][dom]          # 只激活该域自己的子模块
        return first, last, (last / first if first > 0 else float("nan"))

    print("\n=== 隔离检验（Arm I）：训完 B、C 后只激活各自子模块 ===")
    print(f"{'域':<12}{'刚训完':>9}{'只激活本域':>12}{'保留率':>9}")
    iso_ret = {}
    for i, d in enumerate(DOMS):
        first = armI[i]["acquire"][d]
        alone = iso[d][d]
        iso_ret[d] = alone / first if first > 0 else float("nan")
        print(f"{d:<12}{first:>9.3f}{alone:>12.3f}{iso_ret[d]:>9.3f}")

    print("\n=== 两臂保留率对照（容量严格配平 21.037M = 21.037M）===")
    print(f"{'域':<12}{'I(隔离) 首/末':>18}{'保留率':>9}"
          f"{'S48(共享) 首/末':>20}{'保留率':>9}")
    for d in DOMS:
        fI, lI, rI = retention_I(d)
        fS, lS, rS = retention(armS, d)
        print(f"{d:<12}{f'{fI:.3f}/{lI:.3f}':>18}{rI:>9.3f}"
              f"{f'{fS:.3f}/{lS:.3f}':>20}{rS:>9.3f}")

    print("\n=== 组合可用性（三个子模块同时激活，ΔW 相加）===")
    print(f"{'域':<12}{'单独激活':>10}{'组合激活':>10}{'差':>8}")
    for d in DOMS:
        a_, c_ = iso[d][d], composed["acquire"][d]
        print(f"{d:<12}{a_:>10.3f}{c_:>10.3f}{c_-a_:>+8.3f}")
    print(f"\n预训练能力：base={base_ver:.3f} → 组合={composed['verifiable']:.3f} "
          f"(遗忘 {base_ver-composed['verifiable']:+.3f})")
    print(f"ΔCE：{(composed['lm_ce']-base_ce)/base_ce*100:+.2f}%")

    out = {"config": {"layers": LAYERS, "r16": R16, "r48": R48, "steps": STEPS,
                      "domains": DOMS, "sub": SUB,
                      "n_params_3xr16": n16, "n_params_1xr48": n48},
           "baseline": base, "arm_isolated": armI, "arm_shared48": armS,
           "isolated_alone": iso, "composed": composed,
           "isolated_retention_alone": iso_ret,
           "note": "peft 0.20 无 add_weighted_adapter；组合用 forward hook 加 ΔW·x 实现"}
    with open("out/g3_isolation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[G3-5] 写出 out/g3_isolation.json  (总 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

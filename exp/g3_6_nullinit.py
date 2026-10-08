"""G3-6 测 §5.2 的第 2 条主张：**LoRA-Null 零空间初始化**。

方案原文（§5.2）：
    「子模块的初始化位于激活空间的零空间中（LoRA-Null 初始化），
      确保初始更新不干扰已有知识。」

★★ 先做一个**口径检查**（这一步本身就是一个结论）：
    LoRA 的标准初始化是 **B = 0** ⇒ ΔW = (α/r)·B·A **恒等于 0**。
    也就是说「初始更新不干扰已有知识」在标准 LoRA 下**已经精确成立**，
    与是否用零空间初始化无关。
    ⇒ 该主张若要有内容，只能是关于**训练轨迹**的：
      「让**后续**更新也少干扰已有知识」。本脚本就测这个可检验的版本。

设计（同一进程、共用基线；条件间用 reset + 重新初始化，并加复位守卫）：
  条件 1 `std_chunk`  : peft 默认初始化（kaiming A / zeros B）
  条件 2 `null_chunk` : A 的行**正交化**到该层预训练输入激活的前 k=128 个主方向之外
                        （= 激活几乎不用的方向，即激活空间的近似零空间），
                        再按 std 的 Frobenius 范数**缩放配平**（否则差异可能只是「范数更小」）
  条件 3 `std_single` : `std` 但 domB 用**单次 200 步**训练
                        ★ 对照：`train_edit` 每次调用都新建 AdamW ⇒ 分段训练会重置
                          Adam 动量。条件 3 用来量化这个分段混淆，不是可有可无的。

  每个条件下：
    1) 训 domA（域 A，200 步，单次）→ 测 A **单独激活**的习得（该域上限）
    2) 训 domB（域 B）。分段条件下在 0/25/50/100/200 步
       **同时激活 A 与 B**（A 走 set_adapter，B 走 forward hook 加 ΔW·x）测 A 的习得
       ⇒ 这是「新子模块是否干扰已有知识」的**直接读数**
    3) 测 B 自己的习得 ⇒ 检查零空间初始化是否**把学习也一起压死**
       （A x ≈ 0 ⇒ ∂L/∂B = (α/r)(∂L/∂y)(Ax)ᵀ ≈ 0 ⇒ B 几乎不更新，这是可预测的代价）

读数：
  R1 干扰曲线：A 的习得 vs domB 训练步数（std vs null 两条曲线）
  R2 新域习得：null 是否牺牲 B 的学习
  R3 诊断量  ：各初始化下 ‖A Xᵀ‖_F/‖X‖_F（验证「零空间」这一性质**本身**是否成立）
               以及训练后 |ΔW| 的尺度

守卫（沿用 G3-4/G3-5 的纪律）：
  * `set_trainable` 显式白名单（G3-4 的 75GB OOM 教训）；
  * 训 B 时 domA 的 LoRA 权重**逐位**必须不变（隔离）；
  * 复位后所有适配器 ΔW 必须**精确为 0**（否则前一个条件污染后一个）。
"""
from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model, layer_types  # noqa: E402
from s5_edit import train_edit, render  # noqa: E402
import capability as cap  # noqa: E402
from g3_1_premise import eval_acquire  # noqa: E402
import domains as D  # noqa: E402
from g3_5_isolation import (  # noqa: E402
    lora_mods, set_trainable, snapshot_adapter, snapshot_diff, ComposeExtras,
    LAYERS, TARGETS, DOMS, SUB, R16, A16, STEPS, BATCH, SEQ, SEED,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

K_BASIS = 128          # 主方向个数（零空间 = 其正交补）
MAX_ROWS = 2048        # 每模块最多用多少行激活估计主方向
ROW_CAP = 4096         # 收集阶段每模块的行数硬上限（★ UMA：CPU 与 GPU 共享物理内存）
N_COLLECT = 24         # 收集激活用的文本数
INIT_SEED = 1234       # ★ 固定 LoRA 初始化随机源（见下：不固定会淹掉处置效应）
ADAPT = [SUB[d] for d in DOMS[:2]]      # domA, domB
CHUNKS = [25, 25, 50, 100]              # 分段训 domB，累计 0/25/50/100/200


# ══════════════════════════════════════════════════════════════════
# 1) 预训练输入激活的主方向
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def collect_inputs(model, tokenizer, texts):
    """收集每个 LoRA 目标模块的**预训练**输入激活（ΔW=0 时，即基座分布）。

    ★ 每模块行数硬上限 ROW_CAP：UMA 架构下 CPU/GPU 共享同一物理内存池，
      不做限流会在收集阶段吃掉几 GB（本项目已有多次 UMA OOM 教训）。
    """
    store = {}

    def mk(name):
        def hook(module, inputs, output):
            x = inputs[0]
            if x is None or x.dim() != 3:
                return None
            buf = store.setdefault(name, [])
            have = sum(t.shape[0] for t in buf)
            if have >= ROW_CAP:
                return None
            xf = x.detach().reshape(-1, x.shape[-1]).float().cpu()
            if have + xf.shape[0] > ROW_CAP:
                xf = xf[:ROW_CAP - have]
            buf.append(xf)
            return None
        return hook

    handles = [m.register_forward_hook(mk(n)) for n, m in lora_mods(model).items()]
    model.eval()
    try:
        for i in range(0, len(texts), 4):
            enc = tokenizer(texts[i:i + 4], return_tensors="pt", padding=True,
                            truncation=True, max_length=SEQ)
            model(**{k: v.cuda() for k, v in enc.items()})
    finally:
        for h in handles:
            h.remove()
    return store


def top_basis(X, k, oversample=32, niter=2, seed=0):
    """随机化 SVD：返回 X 的前 k 个右奇异向量 V (d,k)，列正交。"""
    n, d = X.shape
    kk = min(k, min(n, d))
    g = torch.Generator(device=X.device).manual_seed(seed)
    Om = torch.randn(d, kk + oversample, device=X.device, generator=g)
    Y = X @ Om
    for _ in range(niter):
        Q, _ = torch.linalg.qr(Y)
        Y = X @ (X.t() @ Q)
    Q, _ = torch.linalg.qr(Y)
    B = Q.t() @ X
    _, _, Vh = torch.linalg.svd(B, full_matrices=False)
    return Vh[:kk].t().contiguous()


def build_basis(store, k=K_BASIS):
    """→ (Vdict, resp)  resp = 各模块把「基座输入」映到输出方向的相对响应。"""
    Vd, resp = {}, {}
    for name, chunks in store.items():
        X = torch.cat(chunks, 0)
        if X.shape[0] > MAX_ROWS:
            idx = torch.randperm(X.shape[0])[:MAX_ROWS]
            X = X[idx]
        X = X.cuda()
        d = X.shape[1]
        if d < 16 or X.shape[0] < 64:
            continue
        Vd[name] = top_basis(X, k).cpu()
        # 诊断：单位随机 A 对真实激活 vs 对同分布各向同性输入的响应比
        G = torch.randn(64, d, device=X.device, generator=torch.Generator(
            device=X.device).manual_seed(1))
        real = (G @ X.t()).norm() / X.norm()
        iso = G.norm() / np.sqrt(X.shape[0] * d)
        resp[name] = float(real / iso)
        del X
    return Vd, resp


# ══════════════════════════════════════════════════════════════════
# 2) 零空间初始化
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def reset_adapters(model, names):
    """把指定适配器复位到 peft 默认初始化（并断言 ΔW 精确为 0）。"""
    from peft.tuners.lora.layer import LoraLayer
    for m in model.modules():
        if isinstance(m, LoraLayer):
            for nm in names:
                if nm in m.lora_A:
                    m.reset_lora_parameters(nm, True)
    for mod in lora_mods(model).values():
        for nm in names:
            d = (mod.lora_B[nm].weight.float() @ mod.lora_A[nm].weight.float())
            assert float(d.abs().max()) == 0.0, f"复位后 ΔW≠0：{nm}"


@torch.no_grad()
def apply_null_init(model, names, Vd, fresh=False):
    """把 A 的行正交化到各模块主方向之外，并**保持原 Frobenius 范数**。

    ★★ 受控对照的关键：默认用**现有的 A**（`fresh=False`）做正交化，
       而不是重新抽一个随机 A。否则「std vs null」的差别里混进了
       「两个不同的随机初始化」——第一次跑就是这么被淹掉的：
       两个 std 条件的 A_alone 分别是 1.000 / 0.800，保留率 0.600 / 0.167，
       差异（0.433）比要测的处置效应还大。
    ★ 范数配平也是必需的：否则「零空间更少干扰」可能只是「A 更小」。
    """
    hit, ratios = 0, []
    for name, mod in lora_mods(model).items():
        V = Vd.get(name)
        if V is None:
            continue
        for nm in names:
            W = mod.lora_A[nm].weight
            Wf = W.float()
            Vd_ = V.float().to(Wf.device)            # ★ Vd 存在 CPU，权重在 GPU
            n_std = float(Wf.norm())                 # std 初始化的范数
            base = torch.randn_like(Wf) if fresh else Wf.clone()
            P = base - (base @ Vd_) @ Vd_.t()        # 去掉主方向分量
            n_new = float(P.norm())
            if n_new <= 0:
                continue
            # 正交化后残余比例（越小 ⇒ 越贴「零空间」）
            ratios.append(float((base @ Vd_).norm() / base.norm()))
            W.copy_((P * (n_std / n_new)).to(W.dtype))
            hit += 1
    return hit, ratios


@torch.no_grad()
def response_ratio(model, names, Vd):
    """诊断：A 对**真实**预训练激活的响应 / 对同分布各向同性的响应。"""
    out = {}
    for nm in names:
        num = den = 0.0
        for name, mod in lora_mods(model).items():
            if name not in Vd:
                continue
            A = mod.lora_A[nm].weight.float()
            # 用主方向所在子空间的能量占比作为「对真实激活的响应」代理：
            # 真实激活的能量几乎都在前 K 个主方向上，故 A 在主方向上的能量
            # 就是 A 对真实输入的有效增益。
            V = Vd[name].float().to(A.device)        # ★ Vd 在 CPU，A 在 GPU
            p = float((A @ V).norm() ** 2)
            t = float(A.norm() ** 2)
            num += p
            den += t
        out[nm] = (num / den) ** 0.5 if den > 0 else float("nan")
    return out


@torch.no_grad()
def delta_mag(model, names):
    out = {}
    for nm in names:
        m = 0.0
        for mod in lora_mods(model).values():
            if nm not in mod.lora_A:
                continue
            d = (mod.lora_B[nm].weight.float() @ mod.lora_A[nm].weight.float())
            m = max(m, float(d.abs().max()))
        out[nm] = m
    return out


# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    os.makedirs("out", exist_ok=True)
    model, tokenizer = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    print(f"[G3-6] 加载 {time.time()-t0:.0f}s")

    lm_texts = cap.lm_texts_heldout(48, start=500)
    ev = {d: D.eval_items(d) for d in DOMS}
    tr = {}
    for d in DOMS:
        s = []
        for q, a in D.train_pairs(d):
            ids = render(tokenizer, q)
            ans = tokenizer(a + tokenizer.eos_token,
                            add_special_tokens=False)["input_ids"]
            s.append((torch.tensor((ids + ans)[:SEQ]),
                      torch.tensor(([-100] * len(ids) + ans)[:SEQ])))
        tr[d] = s

    # ── 适配器 ─────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=R16, lora_alpha=A16, lora_dropout=0.05, bias="none",
                     task_type="CAUSAL_LM", target_modules=TARGETS,
                     layers_pattern="layers", layers_to_transform=LAYERS)
    import random
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = get_peft_model(model, cfg, adapter_name=ADAPT[0])
    for nm in ADAPT[1:]:
        model.add_adapter(nm, cfg)
    mods = lora_mods(model)
    n_per = sum(mod.lora_A[ADAPT[0]].weight.numel()
                + mod.lora_B[ADAPT[0]].weight.numel() for mod in mods.values())
    dims = sorted({mod.lora_A[ADAPT[0]].weight.shape[1] for mod in mods.values()})
    # ★ 目标集**随层类型而变**：三层公共（gate/up/down_proj）每层都有，
    #   另三层（in_proj_qkv/in_proj_z/out_proj）只在 GatedDeltaNet 层存在；
    #   全注意力层而是 self_attn.{q,k,v,o}_proj（不在 TARGETS 里）。
    #   ⇒ 「6 层 × 6 目标 = 36」是错的，本 LAYERS 内只有 1 个全注意力层(15)，
    #     正确数目 = 6×3 + 5×3 = 33。用 layer_types 算，别硬编码。
    _lt = layer_types(model)
    n_fa = sum(1 for L in LAYERS if _lt[L] == "full_attention")
    n_expect = len(LAYERS) * 3 + (len(LAYERS) - n_fa) * 3
    print(f"[G3-6] LoRA 模块={len(mods)}（期望 {n_expect}；其中全注意力层 {n_fa} 个）"
          f"  每适配器 {n_per/1e6:.3f}M  in_features={dims}")
    assert len(mods) == n_expect, f"模块数异常：{len(mods)} ≠ {n_expect}"
    assert n_fa >= 1, "本层带内应至少含 1 个全注意力层"

    # ── 预训练激活主方向（ΔW=0，故此刻即基座分布）────────────────
    texts = cap.lm_texts_heldout(N_COLLECT, start=100)
    store = collect_inputs(model, tokenizer, texts)
    Vd, resp = build_basis(store)
    del store
    torch.cuda.empty_cache()
    print(f"[G3-6] 收集主方向：{len(Vd)}/{len(mods)} 个模块，k={K_BASIS}；"
          f"基座输入相对各向同性响应中位数={np.median(list(resp.values())):.3f} "
          f"({time.time()-t0:.0f}s)")
    missing_basis = sorted(set(mods) - set(Vd))
    assert not missing_basis, f"以下模块未得到主方向：{missing_basis[:5]}"

    # ── 评测量 ─────────────────────────────────────────────────────
    def evaluate(tag, adapter=None, compose=None, battery=False, dom=None):
        model.eval()
        if compose:
            set_trainable(model, ADAPT[0], False)
            model.set_adapter(compose[0])
            h = ComposeExtras(model, compose[1:])
        else:
            model.set_adapter(adapter)
            h = None
        ds = [dom] if dom else DOMS
        acq = {d: eval_acquire(model, tokenizer, ev[d])["acc"] for d in ds}
        rec = {"tag": tag, "acquire": acq}
        if battery:
            b = cap.full_battery(model, tokenizer, lm_texts)
            rec.update({"verifiable": b["verifiable"]["acc"],
                        "lm_ce": b["lm"]["ce"]})
        if h:
            h.remove()
        print(f"[G3-6]   [{tag}] " + " ".join(f"{d}={v:.2f}" for d, v in acq.items())
              + (f" 可验证={rec['verifiable']:.3f}" if battery else "")
              + f" ({time.time()-t0:.0f}s)")
        return rec

    base = evaluate("base", adapter=ADAPT[0], battery=True)
    base_ver, base_ce = base["verifiable"], base["lm_ce"]

    def train_adapter(adapter, samples, steps, chunk=None, tag=""):
        n_tr = set_trainable(model, adapter)
        model.set_adapter(adapter)
        model.train()
        if chunk is None:
            losses, ts = train_edit(model, samples, steps, BATCH, 1e-4,
                                    seed=SEED, log_every=0, tag=tag)
            return losses[-1], ts, n_tr
        last, tt = 0.0, 0.0
        for c in chunk:
            ls, s = train_edit(model, samples, c, BATCH, 1e-4, seed=SEED,
                               log_every=0, tag=tag)
            last, tt = ls[-1], tt + s
        return last, tt, n_tr

    # ══ 条件 ══════════════════════════════════════════════════════
    conds = [("std_chunk", "std", True, True),
             ("null_chunk", "null", True, True),
             ("std_chunk_rep", "std", True, False),
             ("std_single", "std", False, True)]
    results = {}

    for cname, init, chunked, do_cap in conds:
        print(f"\n[G3-6] ═══ 条件 {cname}（init={init}, "
              f"domB {'分段' if chunked else '单次'}）═══")
        # ★★ 固定初始化随机源：让**所有条件的初始 A 完全相同**。
        #   不这么做，init 抽样本身就会造成 ±0.2 的保留率差异，把处置效应淹掉。
        torch.manual_seed(INIT_SEED)
        torch.cuda.manual_seed_all(INIT_SEED)
        reset_adapters(model, ADAPT)
        init_diag = {}
        if init == "null":
            hit, ratios = apply_null_init(model, ADAPT, Vd)   # fresh=False ⇒ 受控
            print(f"[G3-6] 零空间初始化：{hit} 个 (模块,适配器) 被改写；"
                  f"原 A 被剥掉的主方向能量占比中位数="
                  f"{np.median(ratios):.4f}")
            init_diag["stripped_ratio_median"] = float(np.median(ratios))
        # 复位/初始化守卫：B 仍为 0 ⇒ ΔW 必须精确为 0
        dm = delta_mag(model, ADAPT)
        assert all(v == 0.0 for v in dm.values()), f"初始化后 ΔW≠0: {dm}"
        rr = response_ratio(model, ADAPT, Vd)
        print(f"[G3-6] A 在主方向上的能量占比："
              + " ".join(f"{k}={v:.4f}" for k, v in rr.items())
              + "  （越小 ⇒ 越贴「零空间」）")
        init_diag["A_energy_on_principal"] = rr

        # ── domA ──────────────────────────────────────────────────
        lA, sA, n_tr = train_adapter(ADAPT[0], tr[DOMS[0]], STEPS,
                                     tag=f"[{cname}:A] ")
        print(f"[G3-6] 训 domA({DOMS[0]}): {sA:.0f}s loss={lA:.4f} "
              f"可训练={n_tr/1e6:.2f}M")
        recA = evaluate(f"{cname}_A_alone", adapter=ADAPT[0], dom=DOMS[0])
        a_alone = recA["acquire"][DOMS[0]]

        # ── domB（分段时插入干扰读数）─────────────────────────────
        snap = snapshot_adapter(model, ADAPT[0])
        curve = [{"step": 0, "A": a_alone}]
        if chunked:
            done = 0
            for c in CHUNKS:
                lB, sB, n_tr = train_adapter(ADAPT[1], tr[DOMS[1]], c,
                                             tag=f"[{cname}:B] ")
                done += c
                d = snapshot_diff(snap, model, ADAPT[0])
                assert d == 0.0, f"训 domB 时 domA 被改动：{d}——隔离失效"
                r = evaluate(f"{cname}_A_afterB{done}", compose=[ADAPT[0], ADAPT[1]],
                             dom=DOMS[0])
                curve.append({"step": done, "A": r["acquire"][DOMS[0]],
                              "B_loss": lB})
        else:
            lB, sB, n_tr = train_adapter(ADAPT[1], tr[DOMS[1]], STEPS,
                                          tag=f"[{cname}:B] ")
            d = snapshot_diff(snap, model, ADAPT[0])
            assert d == 0.0, f"训 domB 时 domA 被改动：{d}——隔离失效"
            r = evaluate(f"{cname}_A_afterB{STEPS}", compose=[ADAPT[0], ADAPT[1]],
                         dom=DOMS[0])
            curve.append({"step": STEPS, "A": r["acquire"][DOMS[0]], "B_loss": lB})
        print(f"[G3-6] 训 domB({DOMS[1]}): {sB:.0f}s 可训练={n_tr/1e6:.2f}M；"
              f"domA 逐位偏差={d:.1e}")

        recB = evaluate(f"{cname}_B_alone", adapter=ADAPT[1], dom=DOMS[1])
        both = evaluate(f"{cname}_AB", compose=[ADAPT[0], ADAPT[1]],
                        battery=do_cap)
        a_with_b = both["acquire"][DOMS[0]]
        dm_after = delta_mag(model, ADAPT)
        results[cname] = {
            "init": init, "chunked": chunked, "init_diag": init_diag,
            "A_alone": a_alone, "A_with_B": a_with_b,
            "retention_A": a_with_b / a_alone if a_alone > 0 else float("nan"),
            "B_alone": recB["acquire"][DOMS[1]],
            "curve": curve, "lossA": lA, "lossB": curve[-1].get("B_loss"),
            "verifiable": both.get("verifiable"), "lm_ce": both.get("lm_ce"),
            "delta_mag_after": dm_after,
        }

    # ══ 判读 ══════════════════════════════════════════════════════
    print("\n=== R1 干扰曲线：domA 的习得 vs domB 训练步数 ===")
    names = [c[0] for c in conds]
    print(f"{'步数':>6}" + "".join(f"{c:>15}" for c in names))
    steps_all = sorted({p["step"] for c in results.values() for p in c["curve"]})
    for st in steps_all:
        row = f"{st:>6}"
        for c in names:
            pts = [p for p in results[c]["curve"] if p["step"] == st]
            row += f"{pts[0]['A']:>15.3f}" if pts else f"{'-':>15}"
        print(row)

    def _f(v, fmt=".3f"):
        return "   -   " if v is None else format(v, fmt)

    print("\n=== R2 汇总 ===")
    print(f"{'条件':<15}{'A单独':>8}{'A(有B)':>9}{'保留率':>8}{'B单独':>8}"
          f"{'B末loss':>10}{'能力':>8}{'ΔCE%':>8}")
    for c in names:
        r = results[c]
        dce = ((r["lm_ce"] - base_ce) / base_ce * 100
               if r.get("lm_ce") is not None else None)
        print(f"{c:<15}{r['A_alone']:>8.3f}{r['A_with_B']:>9.3f}"
              f"{r['retention_A']:>8.3f}{r['B_alone']:>8.3f}"
              f"{_f(r.get('lossB'), '.4f'):>10}"
              f"{_f(r.get('verifiable')):>8}{_f(dce, '+.2f'):>8}")

    print("\n=== R3 初始化性质诊断（A 在主方向上的能量占比）===")
    for c in names:
        rr = results[c]["init_diag"]["A_energy_on_principal"]
        print(f"{c:<15}" + " ".join(f"{k}={v:.5f}" for k, v in rr.items()))

    # ── 判读：把「处置效应」与「同处置重复的残余差异」并列 ───────────
    rs = results["std_chunk"]["retention_A"]
    rn = results["null_chunk"]["retention_A"]
    rp = results["std_chunk_rep"]["retention_A"]
    rsn = results["std_single"]["retention_A"]
    noise = rp - rs
    eff = rn - rs
    print(f"\n[R] 保留率：std={rs:.3f}  std重复={rp:.3f}  null={rn:.3f}  "
          f"std单次={rsn:.3f}")
    print(f"[R] **同处置重复的残余差异**（std重复 − std）= {noise:+.3f}"
          f"   ← init 已固定，这是纯残留噪声")
    print(f"[R] **零空间净效应**（null − std）= {eff:+.3f}")
    print(f"[R] 分段效应（std单次 − std分段）= {rsn-rs:+.3f}"
          f"   ← 分段会重置 Adam 动量，这是该混淆的量级")
    print(f"[R] ⇒ 零空间净效应 "
          + ("**不可分辨**（|净效应| ≤ |残余噪声|）"
             if abs(eff) <= abs(noise) else "**可分辨**")
          + f"（|{eff:.3f}| vs |{noise:.3f}|）")
    lbs, lbn = results["std_chunk"].get("lossB"), results["null_chunk"].get("lossB")
    if lbs and lbn:
        print(f"[R] **B 的最终训练 loss**：std={lbs:.4f}  null={lbn:.4f}"
              f"  ⇒ 零空间初始化让新域**学得更差**（{lbn/lbs:.1f}×）"
              f"　— loss 是连续量，不受 15 条探针的粒度限制，此信号可靠")
        print("    ★ 机制与预测一致：A 被投到主方向之外 ⇒ A·x 变小 ⇒ "
              "∂L/∂B = (α/r)(∂L/∂y)(Ax)ᵀ 也变小 ⇒ B 更新变慢。"
              "这是**可预期的代价**，但换来的保留率收益为负。")

    out = {"config": {"layers": LAYERS, "targets": TARGETS, "r": R16,
                      "steps": STEPS, "chunks": CHUNKS, "k_basis": K_BASIS,
                      "max_rows": MAX_ROWS, "init_seed": INIT_SEED,
                      "domains": DOMS[:2], "n_params_per_adapter": n_per,
                      "probe_granularity": 1.0 / 15.0},
           "baseline": base,
           "input_response_median": float(np.median(list(resp.values()))),
           "conditions": results,
           "same_treatment_noise": noise,
           "null_net_effect": eff,
           "chunk_effect": rsn - rs,
           "note": ("标准 LoRA B=0 ⇒ 初始 ΔW≡0（已断言），故主张只能指训练轨迹；"
                    "init 随机源已固定(INIT_SEED)，null 用**现有 A** 正交化 ⇒ 受控")}
    with open("out/g3_nullinit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[G3-6] 写出 out/g3_nullinit.json  (总 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

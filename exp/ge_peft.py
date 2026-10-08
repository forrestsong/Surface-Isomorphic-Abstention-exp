"""GE-PEFT 闭环接口（技术方案 §5.3）。

把已验证的零件封装成**有状态的迭代接口**：
知识注入 → 漂移检测 → （必要时）谱校准 → 验证 → 记录奇点图谱变化。
每个子模块独立落盘，manifest 记录全部状态。

    python ge_peft.py list
    python ge_peft.py inject    --builtin northwind [--r 16] [--init std|null]
    python ge_peft.py drift-check [--calibrate]
    python ge_peft.py audit     --self-identity
    python ge_peft.py calibrate --checkpoint CKPT_ID --strength 0.3
    python ge_peft.py rollback  --checkpoint CKPT_ID
    python ge_peft.py route     --eval
    python ge_peft.py scan
    python ge_peft.py consolidate --method grpo [--checkpoint CKPT_ID]
    python ge_peft.py session   --domains northwind,halcyon,vela

★★ 与方案的一处刻意偏离（基于本项目实测，见 `奇点实验报告.md`）：

 1. **漂移阈值不再用「主奇异子空间能量占比 > 15%」。** §5.2 的 15% 经实测
    **不可用**：含破坏性对照在内该指标最高只到 **2.4%（0/37 模块越阈）**，
    且与损伤**解耦**（模型被摧毁时它反而**低于** null）。
    ⇒ 触发改用**行为量**（域探针保留率 / 可验证能力 / LM CE），阈值按**实测噪声底**
      标定（`--calibrate` 现场重测）。谱能量比只作**诊断量**保留并显式标注。

 2. ★ `consolidate --method grpo` **已实现**（2026-10-06，G3-11）：完整 GRPO
    （组内标准化优势 + KL 到基座），奖励**全部程序化判分**（域问答/ MATH/格式 =
    精确匹配，身份 = 是否自称 Qwen/通义千问）⇒ **不需要奖励模型**。
    ⚠️ 已知限制（必须知道）：①它只优化**一个**已注入的子模块，不做多域的联合整合；
    ②当某类奖励在组内零方差时该组**无梯度**，接口会**显式报出退化组比例**；
    ③它**不是**「对抗灾难性遗忘」的手段 —— 防遗忘仍靠参数隔离（§10）。

⚠️ 三点已知局限：
   * `audit` 的污染检测仅限「基座身份是否仍为 Qwen」+「回答是否出现域名词」，
     不是完整能力回归（那由 `drift-check` 的可验证基准负责）；
   * 用 `--data` 自定义域时**没有留出探针**（探针=训练对）⇒ 保留率会高估；
   * `drift-check` 的能力判据在**共享适配器**臂上可能测不到损伤（见 G3-10 注释）。
"""
from __future__ import annotations

import os
import sys
import json
import glob
import time
import shutil
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, load_model, _base_linear  # noqa: E402
from s5_edit import (train_edit, render, generate,  # noqa: E402
                     EVAL_PROBES, TRAIN_QA, TARGET_NAME, TARGET_ORG)
import capability as cap  # noqa: E402
import domains as D  # noqa: E402
import grpo as GR  # noqa: E402
from g3_5_isolation import lora_mods, set_trainable, LAYERS, TARGETS  # noqa: E402
from g3_1_premise import eval_acquire  # noqa: E402
from g3_6_nullinit import (collect_inputs, build_basis, apply_null_init,  # noqa: E402
                           reset_adapters, N_COLLECT)
from g3_7_router import (pooled_features, fit_gate, gate_logits,  # noqa: E402
                         GATE_LAYER)

import numpy as np  # noqa: E402
import torch  # noqa: E402

STATE = "out/ge_state"
MANIFEST = os.path.join(STATE, "manifest.json")
ZERO = "zero"
NONE = "none"
# ★ 身份匹配词表：**必须含中文名**。本项目第一次跑时只查 ASCII "qwen"，
#   而模型答的是 **「通义千问」** ⇒ 把正确答案全判成未命中，
#   得出「身份被污染」的**假警报**（基座也中招）。这是度量侧的静默失效。
QWEN_MARKS = ("qwen", "通义千问", "通义", "千问", "tongyi",
              "alibaba", "阿里")

# ── 阈值：默认值**全部来自实测**，不用方案里的 15% ────────────────────
DEFAULTS = {
    # 跨进程噪声底：同一未改动基座在 3 个独立进程读数极差 = 0.042（全在 format）
    "capability_drop_max": 0.042,
    # 域探针保留率下限（隔离臂实测 1.000；共享臂 0.286）
    "retention_min": 0.90,
    # LM CE 相对增幅上限（方案 §4.3 的回归判据）
    "lm_ce_rel_max": 0.05,
    # 诊断量：谱能量比只记录、不触发
    "spectral_ratio_diagnostic_only": True,
}


# ══════════════════════════════════════════════════════════════════
# 状态
# ══════════════════════════════════════════════════════════════════
def load_manifest():
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    return {"checkpoints": [], "active": [], "drift_history": [],
            "base": None, "thresholds": dict(DEFAULTS)}


def save_manifest(m):
    os.makedirs(STATE, exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def save_checkpoint(model, adapter, cid, meta):
    d = os.path.join(STATE, cid)
    os.makedirs(d, exist_ok=True)
    sd = {}
    for n, mod in lora_mods(model).items():
        if adapter in mod.lora_A:
            sd[n] = {"A": mod.lora_A[adapter].weight.detach().cpu(),
                     "B": mod.lora_B[adapter].weight.detach().cpu()}
    torch.save(sd, os.path.join(d, "adapter.pt"))
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return d


def load_checkpoint(model, adapter, cid):
    sd = torch.load(os.path.join(STATE, cid, "adapter.pt"),
                    map_location="cpu", weights_only=True)
    with open(os.path.join(STATE, cid, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    mods = lora_mods(model)
    hit = 0
    with torch.no_grad():
        for n, w in sd.items():
            if n not in mods:
                continue
            for key, lora in (("A", mods[n].lora_A), ("B", mods[n].lora_B)):
                tgt = lora[adapter].weight
                tgt.copy_(w[key].to(tgt.device, tgt.dtype))
            hit += 1
    return hit, meta


# ══════════════════════════════════════════════════════════════════
# 谱工具（全部用随机化 SVD：12288×4096 做全 SVD 太慢）
# ══════════════════════════════════════════════════════════════════
def topk_svd(M, k, oversample=16, niter=2, seed=0):
    """随机化 SVD，返回 (U_k, S_k, V_k)；U_k (m,k)、V_k (n,k)。"""
    m, n = M.shape
    kk = min(k, min(m, n))
    G = torch.Generator(device=M.device).manual_seed(seed)
    Y = M @ torch.randn(n, kk + oversample, device=M.device, generator=G)
    for _ in range(niter):
        Q, _ = torch.linalg.qr(Y)
        Y = M @ (M.t() @ Q)
    Q, _ = torch.linalg.qr(Y)
    Ub, S, Vh = torch.linalg.svd(Q.t() @ M, full_matrices=False)
    U = (Q @ Ub[:, :kk])[:, :kk].contiguous()
    return U, S[:kk].contiguous(), Vh[:kk].t().contiguous()


@torch.no_grad()
def spectral_energy_ratio(model, adapter, k=48):
    """ΔW 落在「基座权重前 k 个主奇异子空间」上的能量占比。

    ‖P_U ΔW P_V‖_F / ‖ΔW‖_F。★ 实测**不可作触发器**（见模块 docstring），只作诊断。
    """
    vals = []
    for _n, mod in lora_mods(model).items():
        if adapter not in mod.lora_A:
            continue
        dW = (mod.lora_B[adapter].weight.float()
              @ mod.lora_A[adapter].weight.float())
        if float(dW.norm()) == 0.0:
            continue
        W0 = _base_linear(mod).weight.detach().float()
        U, _, V = topk_svd(W0, k)
        vals.append(float((U.t() @ dW @ V).norm()) / float(dW.norm()))
    return {"mean": float(np.mean(vals)) if vals else float("nan"),
            "max": float(np.max(vals)) if vals else float("nan"),
            "n": len(vals)}


@torch.no_grad()
def calibrate_adapter(model, adapter, strength=0.3, k=48):
    """SDC-LoRA 式谱校准：把 ΔW 在主奇异子空间上的分量按 strength 缩掉，
    再用**秩 r 的随机化 SVD** 重新因子化成 B'、A'（保持 LoRA 秩不变）。

    ΔW = s·B@A；dWc = ΔW − strength·P_U ΔW P_V；
    取 dWc 的前 r 个奇异三元组 ⇒ B' = U_r, A' = (V_r·diag(S_r))ᵀ / s，
    于是 s·B'@A' = dWc（秩 r 最优近似）。
    """
    touched = []
    for n, mod in lora_mods(model).items():
        if adapter not in mod.lora_A:
            continue
        Aw, Bw = mod.lora_A[adapter].weight, mod.lora_B[adapter].weight
        s = float(mod.scaling.get(adapter, 1.0))
        dW = s * (Bw.float() @ Aw.float())
        if float(dW.norm()) == 0.0:
            continue
        W0 = _base_linear(mod).weight.detach().float()
        U, _, V = topk_svd(W0, k)
        dWc = dW - strength * (U @ (U.t() @ dW @ V) @ V.t())
        r = Aw.shape[0]
        Uc, Sc, Vc = topk_svd(dWc, r)          # Uc (out,r)  Vc (in,r)
        Bn = Uc.contiguous()
        An = ((Vc * Sc.unsqueeze(0)).t() / s).contiguous()
        Bw.copy_(Bn.to(Bw.device, Bw.dtype))
        Aw.copy_(An.to(Aw.device, Aw.dtype))
        touched.append(n)
    return touched


# ══════════════════════════════════════════════════════════════════
# 数据
# ══════════════════════════════════════════════════════════════════
def domain_data(name, data_path=None):
    """→ (训练对, 探针项)。⚠️ 用 --data 时**无留出探针**（探针=训练对）。"""
    if data_path:
        pairs, probes = [], []
        with open(data_path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                o = json.loads(ln)
                pairs.append((o["prompt"], o["answer"]))
                probes.append({"id": o.get("id", f"p{len(probes)}"),
                               "prompt": o["prompt"], "answer": o["answer"]})
        return pairs, probes
    if name not in D.DOMAINS:
        raise SystemExit(f"未知内置域 {name}；可选 {list(D.DOMAINS)}")
    return D.train_pairs(name), D.eval_items(name)


def build_samples(tokenizer, pairs, seq_len=256):
    out = []
    for q, a in pairs:
        ids = render(tokenizer, q)
        ans = tokenizer(a + tokenizer.eos_token,
                        add_special_tokens=False)["input_ids"]
        out.append((torch.tensor((ids + ans)[:seq_len]),
                    torch.tensor(([-100] * len(ids) + ans)[:seq_len])))
    return out


# ══════════════════════════════════════════════════════════════════
# 引擎
# ══════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self, model_dir, mem_fraction=0.6, seq_len=256):
        self.model, self.tok = load_model(model_dir, precision="bf16",
                                          mem_fraction=mem_fraction)
        self.seq_len = seq_len
        self.wrapped = False
        self.manifest = load_manifest()
        self.lm_texts = cap.lm_texts_heldout(48, start=500)

    @staticmethod
    def cfg(r):
        from peft import LoraConfig
        return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM", target_modules=TARGETS,
                          layers_pattern="layers", layers_to_transform=LAYERS)

    def prepare(self, r=16):
        """建立包装后的模型并装好 `zero` 适配器。

        `zero` 永不训练 ⇒ ΔW≡0 ⇒ 选中它**精确等价于基座**（下面有断言）。
        ★ 必须在任何 `capability(ZERO)` 之前调用；否则会 set_adapter 失败。
        """
        if self.wrapped:
            return
        from peft import get_peft_model
        self.model = get_peft_model(self.model, self.cfg(r), adapter_name=ZERO)
        self.wrapped = True
        for mod in lora_mods(self.model).values():
            dm = (mod.lora_B[ZERO].weight.float().detach()
                  @ mod.lora_A[ZERO].weight.float().detach())
            assert float(dm.abs().max()) == 0.0, "zero 适配器 ΔW≠0"

    def ensure_adapter(self, name, r):
        self.prepare(r)                      # ★ 保证 zero 一定存在
        if name == ZERO:
            return name
        existing = set(next(iter(lora_mods(self.model).values())).lora_A.keys())
        if name not in existing:
            self.model.add_adapter(name, self.cfg(r))
        return name

    def activate(self, name):
        self.model.set_adapter(name, inference_mode=True)

    def inject(self, domain, r=16, steps=200, lr=1e-4, seed=7,
               init="std", data_path=None):
        pairs, probes = domain_data(domain, data_path)
        ad = f"ad_{domain}"
        self.ensure_adapter(ad, r)
        if init == "null":
            store = collect_inputs(self.model, self.tok,
                                   cap.lm_texts_heldout(N_COLLECT, start=100))
            Vd, _resp = build_basis(store)
            del store
            torch.cuda.empty_cache()
            reset_adapters(self.model, [ad])
            hit, _ = apply_null_init(self.model, [ad], Vd)
            print(f"[ge] 零空间初始化：改写 {hit} 个 (模块,适配器)")
        samples = build_samples(self.tok, pairs, self.seq_len)
        n_tr = set_trainable(self.model, ad)
        self.model.set_adapter(ad)
        self.model.train()
        losses, ts = train_edit(self.model, samples, steps, 2, lr, seed=seed,
                                log_every=0, tag=f"[{ad}] ")
        self.model.eval()
        acc = self.eval_probes(probes, ad)
        cid = f"{domain}_r{r}_{init}_{int(time.time())}"
        meta = {"id": cid, "domain": domain, "adapter": ad, "r": r,
                "layers": LAYERS, "targets": TARGETS, "steps": steps, "lr": lr,
                "init": init, "loss": losses[-1], "train_sec": ts,
                "n_params": n_tr, "eval_acc_at_inject": acc,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_train_pairs": len(pairs), "n_probes": len(probes)}
        save_checkpoint(self.model, ad, cid, meta)
        if not any(c["id"] == cid for c in self.manifest["checkpoints"]):
            self.manifest["checkpoints"].append(meta)
        self.manifest["active"] = sorted(set(self.manifest["active"] + [cid]))
        save_manifest(self.manifest)
        print(f"[ge] inject {domain}: loss={losses[-1]:.4f} {ts:.0f}s "
              f"可训练={n_tr/1e6:.3f}M 探针={acc:.3f} → {cid}")
        return meta

    def eval_probes(self, probes, adapter):
        self.activate(adapter)
        return eval_acquire(self.model, self.tok, probes)["acc"]

    def capability(self, adapter=ZERO):
        self.activate(adapter)
        b = cap.full_battery(self.model, self.tok, self.lm_texts)
        return {"verifiable": b["verifiable"]["acc"], "lm_ce": b["lm"]["ce"],
                "lm_token_acc": b["lm"]["token_acc"]}


# ══════════════════════════════════════════════════════════════════
# 命令
# ══════════════════════════════════════════════════════════════════
def cmd_list(args):
    m = load_manifest()
    if not m["checkpoints"]:
        print("（空）还没有检查点。先跑 `inject`。")
        return
    print(f"{'id':<32}{'域':<11}{'r':>4}{'步':>6}{'loss':>9}{'M参':>7}"
          f"{'注入时探针':>11}  创建")
    for c in m["checkpoints"]:
        e = c.get("eval_acc_at_inject")
        print(f"{c['id']:<32}{c['domain']:<11}{c['r']:>4}{c['steps']:>6}"
              f"{c['loss']:>9.4f}{c['n_params']/1e6:>7.3f}"
              f"{(f'{e:.3f}' if e is not None else '-'):>11}  {c['created']}")
    print(f"\n激活集合 = {m['active']}")
    print(f"阈值 = {json.dumps(m['thresholds'], ensure_ascii=False)}")
    if m.get("base"):
        print(f"基座参考 = {json.dumps(m['base'], ensure_ascii=False)}")


def cmd_inject(args, eng):
    name = args.name or args.builtin
    if not name:
        raise SystemExit("需要 --builtin 或 --name")
    if eng.manifest.get("base") is None:
        eng.prepare(16)                       # ★ zero 适配器必须先就位
        b = eng.capability(ZERO)
        eng.manifest["base"] = b
        save_manifest(eng.manifest)
        print(f"[ge] 记录基座参考：{b}")
    eng.inject(name, r=args.r, steps=args.steps, lr=args.lr, seed=args.seed,
               init=args.init, data_path=args.data)


def cmd_drift(args, eng):
    m = eng.manifest
    th = m["thresholds"]
    if not m["checkpoints"]:
        raise SystemExit("还没有检查点，先 inject")
    if args.calibrate:
        a, b = eng.capability(ZERO), eng.capability(ZERO)
        floor = abs(a["verifiable"] - b["verifiable"])
        th["capability_drop_max"] = max(th["capability_drop_max"], floor)
        m["thresholds"] = th
        save_manifest(m)
        print(f"[ge] --calibrate：进程内重复读数 {a['verifiable']:.3f} / "
              f"{b['verifiable']:.3f}（|Δ|={floor:.3f}）；跨进程噪声底 0.042 "
              f"⇒ 阈值 = max = {th['capability_drop_max']:.3f}")

    cur = eng.capability(ZERO)
    base = m.get("base") or cur
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "base_capability": base,
           "cur_capability": cur, "domains": {}}

    print("\n=== 域探针保留率（只激活该域自己的子模块）===")
    print(f"{'域':<12}{'注入时':>8}{'现在':>8}{'保留率':>9}{'谱能量比':>11}")
    worst = 1.0
    for c in m["checkpoints"]:
        dom, ad = c["domain"], c["adapter"]
        _pairs, probes = domain_data(dom)
        first = c.get("eval_acc_at_inject")
        now = eng.eval_probes(probes, ad)
        ret = (now / first) if first else float("nan")
        sr = spectral_energy_ratio(eng.model, ad)
        if ret == ret:
            worst = min(worst, ret)
        rec["domains"][dom] = {"first": first, "now": now, "retention": ret,
                              "spectral_ratio": sr, "checkpoint": c["id"]}
        print(f"{dom:<12}{first if first is not None else float('nan'):>8.3f}"
              f"{now:>8.3f}{ret:>9.3f}{sr['mean']:>11.4f}")

    drop = base["verifiable"] - cur["verifiable"]
    rel = ((cur["lm_ce"] - base["lm_ce"]) / base["lm_ce"]) if base["lm_ce"] else float("nan")

    # ★★ 修一个**结构性失明**（G3-10 实测坐实的）：
    #   `cur = capability(ZERO)` 是在**基座态**上测能力。对「隔离」设计这没错
    #   （服务态 = 逐查询路由，基座能力是上界），但它**对局限在适配器里的损伤恒为
    #   不变量** —— 实测：共享适配器把 5 个域的知识全部抹掉（保留率 0.000）、
    #   LM CE 只 +0.55%，而可验证能力 **+0.000**。⇒ 只要状态不含被服务的参数，
    #   这个判据就永远不会报警。
    #   修法：**再逐适配器测一遍**，把「最差的服务态能力」也纳入判据。
    per_ad = {}
    for ad_name in sorted({c["adapter"] for c in m["checkpoints"]}):
        eng.activate(ad_name)
        b = cap.full_battery(eng.model, eng.tok, eng.lm_texts)
        per_ad[ad_name] = {"verifiable": b["verifiable"]["acc"],
                           "lm_ce": b["lm"]["ce"]}
    eng.activate(ZERO)
    worst_ad_drop = (max(base["verifiable"] - x["verifiable"]
                         for x in per_ad.values()) if per_ad else 0.0)
    print("\n=== 逐适配器服务态能力（★ 修补「基座态测能力」的失明）===")
    for ad_name, x in per_ad.items():
        print(f"  {ad_name:<20} verifiable={x['verifiable']:.3f}"
              f"（相对基座 {x['verifiable']-base['verifiable']:+.3f}）"
              f"  LM CE={x['lm_ce']:.4f}")

    v = {"worst_retention": worst, "retention_min": th["retention_min"],
         "retention_ok": bool(worst >= th["retention_min"]),
         "capability_drop_base_state": drop,
         "capability_drop_worst_adapter_state": worst_ad_drop,
         "capability_drop": worst_ad_drop,
         "capability_drop_max": th["capability_drop_max"],
         "capability_ok": bool(abs(worst_ad_drop) <= th["capability_drop_max"]),
         "per_adapter_capability": per_ad,
         "lm_ce_rel": rel, "lm_ce_rel_max": th["lm_ce_rel_max"],
         "lm_ce_ok": bool(abs(rel) <= th["lm_ce_rel_max"])}
    rec["verdict"] = v
    print("\n=== 判读（阈值按实测噪声底标定，**不是**方案的 15%）===")
    print(f"  最差保留率 {worst:.3f} vs 下限 {v['retention_min']:.2f}"
          f"  → {'OK' if v['retention_ok'] else '**漂移**'}")
    print(f"  能力下降（**最差服务态**）{worst_ad_drop:+.3f} vs 上限 "
          f"{v['capability_drop_max']:.3f}"
          f"  → {'OK' if v['capability_ok'] else '**漂移**'}")
    print(f"  能力下降（基座态，参考）{drop:+.3f}")
    print(f"  LM CE 相对 {rel:+.2%} vs 上限 {v['lm_ce_rel_max']:.0%}"
          f"  → {'OK' if v['lm_ce_ok'] else '**漂移**'}")
    print("  ⚠️ 谱能量比**仅作诊断**：实测在破坏性对照下仍 <2.4% 且与损伤解耦，"
          "不可作触发器。")
    print("  ⚠️ LM CE 也**不是**灵敏判据：G3-10 实测知识全灭时它只 +0.55%。"
          "真正有效的触发器是**逐域保留率**（悬崖式遗忘在第一步就被它抓到）。")
    need = not (v["retention_ok"] and v["capability_ok"] and v["lm_ce_ok"])
    print(f"\n  ⇒ 需要谱校准？ {'**是**（跑 calibrate）' if need else '否'}")
    m["drift_history"].append(rec)
    save_manifest(m)


def cmd_calibrate(args, eng):
    m = eng.manifest
    tgt = [c for c in m["checkpoints"] if c["id"] == args.checkpoint]
    if not tgt:
        raise SystemExit(f"找不到检查点 {args.checkpoint}")
    c = tgt[0]
    ad = c["adapter"]
    _pairs, probes = domain_data(c["domain"])
    first = c.get("eval_acc_at_inject") or 1.0
    b_ret = eng.eval_probes(probes, ad) / first
    b_cap = eng.capability(ZERO)
    b_sr = spectral_energy_ratio(eng.model, ad)
    touched = calibrate_adapter(eng.model, ad, strength=args.strength)
    a_ret = eng.eval_probes(probes, ad) / first
    a_cap = eng.capability(ZERO)
    a_sr = spectral_energy_ratio(eng.model, ad)
    print(f"\n=== 谱校准（strength={args.strength}，改写 {len(touched)} 个模块）===")
    print(f"{'指标':<20}{'校准前':>11}{'校准后':>11}")
    print(f"{'该域保留率':<20}{b_ret:>11.3f}{a_ret:>11.3f}")
    print(f"{'可验证能力':<20}{b_cap['verifiable']:>11.3f}{a_cap['verifiable']:>11.3f}")
    print(f"{'LM CE':<20}{b_cap['lm_ce']:>11.4f}{a_cap['lm_ce']:>11.4f}")
    print(f"{'谱能量比(均值)':<20}{b_sr['mean']:>11.4f}{a_sr['mean']:>11.4f}")
    print("  ⚠️ 校准削减的是**知识本身**在主方向上的分量，会同时削弱目标域 ——"
          " 这是真实代价，不是免费的。")
    cid = f"cal_{args.checkpoint}_s{args.strength}"
    meta = dict(c, id=cid, calibrated_from=args.checkpoint,
                strength=args.strength)
    save_checkpoint(eng.model, ad, cid, meta)
    m["checkpoints"].append(meta)
    save_manifest(m)
    print(f"  → 另存为 {cid}（原检查点未被覆盖）")


def cmd_rollback(args, eng):
    m = eng.manifest
    tgt = [c for c in m["checkpoints"] if c["id"] == args.checkpoint]
    if not tgt:
        raise SystemExit(f"找不到检查点 {args.checkpoint}")
    c = tgt[0]
    ad = f"rb_{c['domain']}"
    eng.ensure_adapter(ad, c["r"])
    hit, _meta = load_checkpoint(eng.model, ad, c["id"])
    _pairs, probes = domain_data(c["domain"])
    acc = eng.eval_probes(probes, ad)
    exp = c.get("eval_acc_at_inject")
    ok = exp is not None and abs(acc - exp) < 1e-9
    print(f"[ge] rollback → {c['id']}：载入 {hit} 个模块；该域探针 {acc:.3f}"
          f"（记录值 {exp if exp is not None else float('nan'):.3f}）"
          f"  {'复现 ✅' if ok else '⚠️ 不一致（可能因自定义域无留出探针）'}")
    m["active"] = [c["id"]]
    save_manifest(m)
    print(f"    激活集合已置为 {m['active']}")


def cmd_audit(args, eng):
    m = eng.manifest
    states = [("base", ZERO)] + [(c["domain"], c["adapter"])
                                 for c in m["checkpoints"]]
    print(f"\n=== 自我认知审计（{len(EVAL_PROBES)} 条留出探针）===")
    print(f"{'状态':<14}{'自称 Qwen':>11}{'自称 Nebula(编辑目标)':>22}{'域名词':>9}")
    out = {}
    for tag, a in states:
        eng.activate(a)
        hq = hn = hp = 0
        raw = []
        for p in EVAL_PROBES:
            txt = generate(eng.model, eng.tok, p, max_new_tokens=48)
            raw.append({"probe": p, "answer": txt[:200]})
            low = txt.lower()
            hq += int(any(k in low for k in QWEN_MARKS))
            hn += int(TARGET_NAME.lower() in low or TARGET_ORG.lower() in low)
            hp += int(any(d.lower() in low for d in D.DOMAINS))
        n = len(EVAL_PROBES)
        out[tag] = {"qwen_rate": hq / n, "nebula_rate": hn / n,
                    "domain_pollution": hp / n, "raw": raw}
        print(f"{tag:<14}{hq/n:>11.3f}{hn/n:>22.3f}{hp/n:>9.3f}")
    bq = out["base"]["qwen_rate"]
    worst = min(v["qwen_rate"] for v in out.values())
    worst_pol = max(v["domain_pollution"] for v in out.values())
    print(f"\n  基座自称 Qwen（含中文名「通义千问」）率 = {bq:.3f}；"
          f"最差状态 = {worst:.3f}")
    if worst_pol > 0:
        print("  → **身份被替换**：有状态开始自称某个域实体"
              "（domain_pollution > 0）")
    elif worst < 1.0:
        print(f"  → 身份**未被替换**（域名词污染 = 0），但**少数探针出现虚构身份**："
              f"{bq:.3f} → {worst:.3f}。下面列出未命中的回答供人工判断"
              "（首次实测出现 'Alex Chen' / 'Tencent' 这类无关虚构）。")
        for tag, v in out.items():
            bad = [r for r in v["raw"]
                   if not any(k in r["answer"].lower() for k in QWEN_MARKS)]
            for r in bad:
                print(f"       [{tag}] {r['probe']} → {r['answer'][:60]!r}")
    else:
        print("  → 身份完全未被污染 ✅")
    print("  ★ 注意：这是「域子模块**始终激活**」的**未路由最坏情形**；"
          "实际部署下门控会把「你是谁」这类查询路由到 `none`。")
    print("  ⚠️ 仅限「基座身份是否仍为 Qwen」+「是否出现域名词」；"
          "通用能力回归由 drift-check 负责。")
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def cmd_route(args, eng):
    m = eng.manifest
    doms = [c["domain"] for c in m["checkpoints"]]
    if not doms:
        raise SystemExit("还没有检查点")
    ad_of = {c["domain"]: c["adapter"] for c in m["checkpoints"]}
    eng.activate(ZERO)
    cap_items = (cap.build_math_items() + cap.build_format_items()
                 + cap.build_code_items())
    cap_prompts = [it["prompt"] for it in cap_items]
    cls = doms + [NONE]
    Xs, ys = [], []
    for i, d in enumerate(doms):
        pairs, _pr = domain_data(d)
        qs = [q for q, _a in pairs]
        Xs.append(pooled_features(eng.model, eng.tok, qs))
        ys += [i] * len(qs)
    Xn = pooled_features(eng.model, eng.tok, cap_prompts)
    n_dom = sum(len(domain_data(d)[0]) for d in doms)
    rep = max(1, round(n_dom / len(cap_prompts)))
    Xn = torch.cat([Xn] * rep, 0)
    Xs.append(Xn)
    ys += [len(doms)] * Xn.shape[0]
    X = torch.cat(Xs, 0)
    gate = fit_gate(X, ys, len(cls))
    acc_tr = float((gate_logits(gate, X).argmax(1) == torch.tensor(ys))
                   .float().mean())
    print(f"\n[ge] 门控（基座 L{GATE_LAYER} 平均池化 + 岭回归）类别={cls} "
          f"训练准确率={acc_tr:.3f}")
    if not args.eval:
        print("    加 --eval 跑路由评测")
        return
    ev = {d: domain_data(d)[1] for d in doms}
    allq = cap_prompts + [it["prompt"] for d in doms for it in ev[d]]
    pr = gate_logits(gate, pooled_features(eng.model, eng.tok, allq)) \
        .argmax(1).tolist()
    rmap = {q: (ad_of[doms[j]] if j < len(doms) else ZERO)
            for q, j in zip(allq, pr)}

    def route(prompt):
        return rmap.get(prompt, ZERO)      # 未见过的查询 → 基座（保守）

    print(f"\n{'域':<12}{'base':>8}{'oracle':>9}{'router':>9}")
    orc, rou, bsc = [], [], []
    for d in doms:
        eng.activate(ad_of[d])
        o = eval_acquire(eng.model, eng.tok, ev[d])["acc"]
        eng.activate(ZERO)
        b = eval_acquire(eng.model, eng.tok, ev[d])["acc"]
        ok = 0
        for it in ev[d]:
            eng.activate(route(it["prompt"]))
            txt = generate(eng.model, eng.tok, it["prompt"], max_new_tokens=32)
            ok += int(cap.grade_exact(txt, it["answer"]))
        r = ok / len(ev[d])
        orc.append(o), rou.append(r), bsc.append(b)
        print(f"{d:<12}{b:>8.2f}{o:>9.2f}{r:>9.2f}")
    print(f"{'均值':<12}{np.mean(bsc):>8.2f}{np.mean(orc):>9.2f}{np.mean(rou):>9.2f}")
    if np.mean(orc) > 0:
        print(f"  router 追平 oracle 的比例 = {np.mean(rou)/np.mean(orc):.3f}")
    eng.activate(ZERO)
    bv = cap.full_battery(eng.model, eng.tok, eng.lm_texts)["verifiable"]["acc"]
    ok = 0
    for it in cap_items:
        eng.activate(route(it["prompt"]))
        txt = generate(eng.model, eng.tok, it["prompt"], max_new_tokens=40)
        ok += int(cap.run_unittest(cap.extract_code(txt), it["tests"])[0]
                  if it["check"] == "unittest"
                  else cap.grade_exact(txt, it["answer"]))
    eng.activate(ZERO)
    rv = ok / len(cap_items)
    print(f"  可验证能力：base={bv:.3f} → 路由后={rv:.3f}（{rv-bv:+.3f}）")
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, "route.json"), "w", encoding="utf-8") as f:
        json.dump({"train_acc": acc_tr, "domains": doms,
                   "base": dict(zip(doms, bsc)), "oracle": dict(zip(doms, orc)),
                   "router": dict(zip(doms, rou)),
                   "mean_base": float(np.mean(bsc)),
                   "mean_oracle": float(np.mean(orc)),
                   "mean_router": float(np.mean(rou)),
                   "router_over_oracle": (float(np.mean(rou) / np.mean(orc))
                                          if np.mean(orc) > 0 else None),
                   "verifiable_base": bv, "verifiable_routed": rv},
                  f, ensure_ascii=False, indent=2)


def spearman_rank(a, b):
    """秩相关（与 `s2_analyze.spearman` 同口径，就地实现避免跨脚本导入副作用）。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / den) if den else 0.0


def cmd_consolidate(args, eng):
    """GRPO 整合（tech plan §4.2 阶段三；把 §13.5 的 ❌ 补上）。

    奖励**全部程序化判分**（域问答 / MATH / 格式 = 精确匹配；身份 = 是否自称
    Qwen/通义千问）⇒ **不需要奖励模型**。KL 参考 = 基座（`disable_adapter()`），
    **不额外载入第二个模型**（UMA 上省一份 18GB）。

    ⚠️ 诚实边界：
      * 只优化**一个**已注入子模块，不做多域联合整合；
      * 某类奖励在组内零方差 ⇒ 该组无梯度，故**显式报出退化组比例**；
      * 它**不是**防遗忘手段（防遗忘靠参数隔离，见 §10），而是「行为对齐」。
    """
    m = eng.manifest
    if not m["checkpoints"]:
        raise SystemExit("还没有检查点，先 inject")
    cid = args.checkpoint or (m.get("active") or [None])[-1]
    tgt = [c for c in m["checkpoints"] if c["id"] == cid]
    if not tgt:
        raise SystemExit(f"找不到检查点 {cid}")
    c = tgt[0]
    dom, ad = c["domain"], c["adapter"]
    print(f"[ge] consolidate 目标 = {cid}（域 {dom}，适配器 {ad}）")

    pairs, probes = domain_data(dom)
    pool = [("dom", q, {"answer": a}) for q, a in pairs]
    for it in (cap.build_math_items() + cap.build_format_items()):
        pool.append(("cap", it["prompt"], it))
    id_pool = [q for q, _a in TRAIN_QA]        # 身份奖励池（未被 SFT 使用）
    for p in id_pool:
        pool.append(("id", p, None))
    print(f"[ge] 奖励池 {len(pool)} 条（dom {len(pairs)} / cap "
          f"{len(cap.build_math_items())+len(cap.build_format_items())} / "
          f"id {len(id_pool)}）")

    def reward_of(kind, item, text):
        if kind == "id":
            return 1.0 if any(k_ in text.lower() for k_ in QWEN_MARKS) else 0.0
        return 1.0 if cap.grade_exact(text, item["answer"]) else 0.0

    def snapshot_state():
        eng.activate(ad)
        acq = eng.eval_probes(probes, ad)
        ident = 0
        for p in EVAL_PROBES:                  # 全部 12 条留出（与奖励池不同问法）
            txt = generate(eng.model, eng.tok, p, max_new_tokens=32)
            ident += int(any(k_ in txt.lower() for k_ in QWEN_MARKS))
        return {"acquisition": acq,
                "identity_hit_holdout": ident / max(len(EVAL_PROBES), 1),
                "n_identity_holdout": len(EVAL_PROBES)}

    before = snapshot_state()
    print(f"[ge] 整合前：习得={before['acquisition']:.3f} "
          f"身份命中(留出)={before['identity_hit_holdout']:.3f}")
    res = GR.consolidate(eng.model, eng.tok, ad, pool, reward_of,
                         steps=args.steps, lr=args.lr, G=args.group,
                         prompts_per_step=args.prompts_per_step,
                         beta=args.beta)
    after = snapshot_state()
    print(f"[ge] 整合后：习得={after['acquisition']:.3f} "
          f"身份命中(留出)={after['identity_hit_holdout']:.3f}")
    print(f"[ge] Δ：习得 {after['acquisition']-before['acquisition']:+.3f}  "
          f"身份 {after['identity_hit_holdout']-before['identity_hit_holdout']:+.3f}")
    print(f"[ge] 诊断：退化组比例={res['degenerate_group_frac']:.3f}"
          f"（奖励零方差 ⇒ 无梯度）、可训练={res['n_trainable']/1e6:.3f}M")
    if res["degenerate_group_frac"] > 0.5:
        print("      ⚠️ 过半数组无梯度 ⇒ 该类奖励对当前策略**已饱和**，"
              "不提供学习信号。这不是 bug，是奖励设计需要分化的证据。")
    cid2 = f"grpo_{cid}"
    meta = dict(c, id=cid2, consolidated_from=cid, method="grpo",
                grpo_steps=args.steps, grpo_lr=args.lr, grpo_group=args.group,
                grpo_beta=args.beta,
                degenerate_group_frac=res["degenerate_group_frac"],
                before=before, after=after,
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    meta.pop("grpo_history", None)
    save_checkpoint(eng.model, ad, cid2, meta)
    m["checkpoints"].append(meta)
    save_manifest(m)
    with open(os.path.join(STATE, "consolidate_last.json"), "w",
              encoding="utf-8") as f:
        json.dump({"target": cid, "new": cid2, "before": before, "after": after,
                   "history": res["history"],
                   "degenerate_group_frac": res["degenerate_group_frac"]},
                  f, ensure_ascii=False, indent=2)
    print(f"  → 另存为 {cid2}（原检查点在磁盘上未被覆盖）")


def cmd_scan(args, eng):
    """重扫奇点结构并与基座图谱比对（方案 §5.4「增量更新奇点图谱」）。

    ★ 修复（2026-10-06）：此前本命令**从未真正比对过** —— 它去读 `s1_probe.json`
      里的 `ranked_layers/score_total`，而 `s1_probe.json` 只有
      `{meta, per_linear, per_weight, deltanet_decay}` 三个键（逐通道/逐矩阵明细），
      **没有层排名** ⇒ 每次都走「层键不一致，跳过比对」分支 ⇒ 「已实现但未验证」。

      正确的两步是：`s1_probe.py`（测量）→ `s2_analyze.py --dir`（聚合出层排名
      + `score_total`）⇒ 比对的是**两次 `singularity_map.json`**。
    """
    m = eng.manifest
    out = "out/scan_now"
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "s1_probe.py", "--out", out]
    act = m.get("active") or []
    used_adapter = None
    if act:
        c = [x for x in m["checkpoints"] if x["id"] == act[-1]]
        if c:
            used_adapter = c[0]["adapter"]
            # ★ 修复之二（2026-10-06）：`s1_probe.py --adapter` 要的是**PEFT 检查点目录**
            #   （需含 adapter_config.json），而这里以前传的是**适配器名**（如
            #   `ad_northwind`）⇒ `PeftModel.from_pretrained` 抛
            #   `Can't find 'adapter_config.json' at 'ad_northwind'` ⇒ 退出码 1。
            #   修法：先把该适配器**导出成 PEFT 目录**再传路径。
            #   ⚠️ 前置条件：本进程的模型是**裸**的（还没 prepare/wrap），
            #   必须先 wrap（否则 `set_adapter` 无 `inference_mode` 形参）并把
            #   检查点权重复原回来，否则扫的是**未改动基座**而不是被编辑的状态。
            r0 = int(c[0].get("r", 16))
            eng.prepare(r0)
            existing = set(next(iter(lora_mods(eng.model).values())).lora_A.keys())
            if used_adapter not in existing:
                eng.ensure_adapter(used_adapter, r0)
                hit, _m = load_checkpoint(eng.model, used_adapter, c[0]["id"])
                print(f"[ge] 已复原检查点 {c[0]['id']}（命中 {hit} 个模块）"
                      f"⇒ 扫的是**编辑后**状态，不是基座")
            eng.activate(used_adapter)
            ad_dir = "out/scan_adapter"
            if os.path.isdir(ad_dir):
                shutil.rmtree(ad_dir)
            os.makedirs(ad_dir, exist_ok=True)
            eng.model.save_pretrained(ad_dir, selected_adapters=[used_adapter])
            # ★ 修复之三：peft 0.20 的 `save_pretrained(selected_adapters=[...])`
            #   会把每个适配器写进**同名子目录**（ad_dir/ad_northwind/），
            #   根目录只有 README.md ⇒ 必须把子目录（含 adapter_config.json）传下去。
            sub = os.path.join(ad_dir, used_adapter)
            if os.path.exists(os.path.join(sub, "adapter_config.json")):
                ad_dir = sub
            print(f"[ge] 已导出适配器 {used_adapter} → {ad_dir} "
                  f"（{sorted(os.listdir(ad_dir))}）")
            cmd += ["--adapter", ad_dir]
    print(f"[ge] 重扫：{' '.join(cmd)}")
    print(f"     （被扫状态：{'基座+适配器 ' + used_adapter if used_adapter else '未改动基座'}）")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(f"    s1_probe 退出码 {r.returncode}；stderr 尾部：")
        print("    " + (r.stderr or "")[-600:])
        return
    if not os.path.exists(os.path.join(out, "s1_probe.json")):
        print("    ⚠️ s1_probe 未产出 s1_probe.json")
        return
    # ★ 第二步：聚合出可比对的层排名（没有这一步就没有 score_total 可比）
    r2 = subprocess.run([sys.executable, "s2_analyze.py", "--dir", out],
                        capture_output=True, text=True, timeout=1800)
    if r2.returncode != 0:
        print(f"    s2_analyze 退出码 {r2.returncode}；stderr 尾部：")
        print("    " + (r2.stderr or "")[-600:])
        return

    base_f = "out/singularity_map.json"                 # 基座图谱（冻结参考）
    now_f = os.path.join(out, "singularity_map.json")   # 本次重扫
    if not (os.path.exists(now_f) and os.path.exists(base_f)):
        print(f"    ⚠️ 缺少可比对产物：base={os.path.exists(base_f)} now={os.path.exists(now_f)}")
        return
    with open(base_f, encoding="utf-8") as f:
        base = json.load(f)
    with open(now_f, encoding="utf-8") as f:
        now = json.load(f)
    b = {x_["layer"]: x_.get("score_total")
         for x_ in base.get("ranked_layers", []) if "layer" in x_}
    n_ = {x_["layer"]: x_.get("score_total")
          for x_ in now.get("ranked_layers", []) if "layer" in x_}
    common = sorted(L for L in (set(b) & set(n_))
                    if b[L] is not None and n_[L] is not None)
    if not common:
        print("    ⚠️ 两次图谱层键不一致，跳过比对")
        return
    d = {L: n_[L] - b[L] for L in common}
    top = sorted(d.items(), key=lambda kv: -abs(kv[1]))[:6]
    print(f"    可比对层数={len(common)}  "
          f"最大变化层：" + " ".join(f"L{L}:{v:+.3f}" for L, v in top))
    ms = float(np.mean([abs(v) for v in d.values()]))
    mx = float(max(abs(v) for v in d.values()))
    rho = spearman_rank([b[L] for L in common], [n_[L] for L in common])
    print(f"    平均 |Δscore| = {ms:.4f}   最大 |Δscore| = {mx:.4f}"
          f"   层排名 Spearman ρ = {rho:+.3f}")
    print(f"    ⇒ {'超过门槛 0.2，触发全量重扫描' if ms > 0.2 else '在门槛内，增量更新即可'}")
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, "scan_last.json"), "w", encoding="utf-8") as f:
        json.dump({"adapter": used_adapter, "n_layers": len(common),
                   "mean_abs_delta": ms, "max_abs_delta": mx,
                   "spearman_rho": rho,
                   "per_layer_delta": {str(L): d[L] for L in common},
                   "top_changed": [[L, v] for L, v in top]},
                  f, ensure_ascii=False, indent=2)


def cmd_session(args, eng):
    doms = [x.strip() for x in args.domains.split(",") if x.strip()]
    print(f"\n{'='*66}\n[ge] session：单进程闭环，{len(doms)} 个域\n{'='*66}")
    if eng.manifest.get("base") is None:
        eng.prepare(16)                       # ★ zero 适配器必须先就位
        eng.manifest["base"] = eng.capability(ZERO)
        save_manifest(eng.manifest)
    print(f"[ge] 基座参考：{eng.manifest['base']}\n")
    for d in doms:                                          # ① 注入
        eng.inject(d, steps=args.steps)
    cmd_drift(argparse.Namespace(calibrate=True), eng)       # ② 漂移检测
    cmd_audit(argparse.Namespace(), eng)                     # ③ 身份审计
    cmd_route(argparse.Namespace(eval=True), eng)            # ④ 门控路由
    print(f"\n[ge] session 完成；状态在 {MANIFEST}")


def main():
    ap = argparse.ArgumentParser(prog="ge_peft.py",
                                 description="GE-PEFT 闭环接口（技术方案 §5.3）")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出检查点与阈值（不加载模型）")

    p = sub.add_parser("inject", help="注入一个新知识域（独立子模块）")
    p.add_argument("--builtin", default=None)
    p.add_argument("--data", default=None, help="JSONL：prompt/answer")
    p.add_argument("--name", default=None)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--init", choices=["std", "null"], default="std")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_inject)

    p = sub.add_parser("drift-check", help="漂移检测（行为量触发 + 谱量诊断）")
    p.add_argument("--calibrate", action="store_true")
    p.set_defaults(fn=cmd_drift)

    p = sub.add_parser("calibrate", help="对某检查点执行谱校准")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--strength", type=float, default=0.3)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("rollback", help="回滚到某检查点并验证复现")
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(fn=cmd_rollback)

    p = sub.add_parser("audit", help="自我认知审计")
    p.add_argument("--self-identity", action="store_true")
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("route", help="构建/评估门控路由")
    p.add_argument("--eval", action="store_true")
    p.set_defaults(fn=cmd_route)

    p = sub.add_parser("scan", help="重扫奇点结构并与基座图谱比对")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("consolidate", help="GRPO 整合（可验证奖励，无需奖励模型）")
    p.add_argument("--method", default="grpo", choices=["grpo"])
    p.add_argument("--checkpoint", default=None,
                   help="目标检查点 id；默认 = active 的最后一个")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--group", type=int, default=4, help="GRPO 组大小 G")
    p.add_argument("--prompts-per-step", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.02, help="KL 到基座的系数")
    p.set_defaults(fn=cmd_consolidate)

    p = sub.add_parser("session", help="一次进程内跑完整闭环")
    p.add_argument("--domains", default="northwind,halcyon,vela")
    p.add_argument("--steps", type=int, default=200)
    p.set_defaults(fn=cmd_session)

    args = ap.parse_args()
    if args.cmd == "list":
        return cmd_list(args)
    t0 = time.time()
    eng = Engine(args.model_dir)
    print(f"[ge] 模型加载 {time.time()-t0:.0f}s")
    return args.fn(args, eng)


if __name__ == "__main__":
    main()

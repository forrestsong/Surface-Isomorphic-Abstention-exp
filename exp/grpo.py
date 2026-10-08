"""GRPO 核：**可验证奖励**版（`ge_peft.py consolidate --method grpo` 的后端）。

背景：`奇点实验报告.md` §13.5 把 `consolidate --method GRPO` 列为**唯一未实现项**，
理由是「需要奖励模型/可验证奖励的 RL 回路，是独立子项目」。本模块补上它。

★ 关键设计取舍（都是为了让它在**没有奖励模型**的前提下成立）：
  1. **奖励来自程序化判分**，不是学出来的奖励模型：
     域问答/MATH/格式 → `capability.grade_exact`（精确匹配 + 特异子串），
     代码 → 子进程跑单测，身份 → 是否自称 Qwen/通义千问。
     ⇒ 奖励不可被 reward model 的偏好偏差污染，也不需要额外训练。
  2. **KL 参考 = 基座**（`model.disable_adapter()`），**不额外载入第二个模型**
     ⇒ 在 UMA 121GB 上省掉一份 18GB。语义 = 「行为别偏离预训练策略太远」，
     正是方案 §4.2 阶段三要的「防止注入的知识在其他上下文中泄漏或被覆盖」。
  3. 只优化 LoRA 参数（显式白名单），G3-4 的 75GB OOM 教训。

★ GRPO 本身：对同一 prompt 采样 G 条，组内标准化奖励作优势（无 critic）：
     A_i = (r_i − mean(r)) / (std(r) + ε),   loss = −E[A_i · logπ(y_i)] + β·KL
  组内奖励全同 ⇒ A≡0 ⇒ 该组无梯度（会**显式统计**这种组所占比例，
  因为它是「奖励信号是否有效」的诊断，不是可以藏起来的细节）。

⚠️ 与 SFT 的公平对照：RL 与「多跑几步 SFT」消耗的优化步数相同
   ⇒ 任何改善都必须相对 `sft+more_sft` 读，否则会把「多算了」当成「RL 有用」。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def seq_logp(model, ids, n_prompt):
    """给定 [prompt+completion] 的 ids，返回 completion 部分的对数概率和。"""
    logits = model(input_ids=ids).logits.float()
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    start = max(n_prompt - 1, 0)
    if start >= tok_lp.shape[1]:
        return logits.sum() * 0.0
    return tok_lp[:, start:].sum(-1)


@torch.no_grad()
def sample_group(model, tok, prompt, G, max_new=24, temp=0.9, top_p=0.95):
    """同一 prompt 采样 G 条 completion（贪婪会被组内零方差毁掉，必须采样）。"""
    from s5_edit import render
    ids = torch.tensor([render(tok, prompt)], dtype=torch.long).cuda()
    model.eval()
    out = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=True,
                         temperature=temp, top_p=top_p,
                         num_return_sequences=G,
                         pad_token_id=tok.pad_token_id, use_cache=True)
    comps = [out[i, ids.shape[1]:] for i in range(G)]
    texts = [tok.decode(c, skip_special_tokens=True).strip() for c in comps]
    return ids[0], comps, texts


def grpo_step(model, tok, optimizer, pool, reward_of, cfg, log=None,
              train_adapter=None, ref_adapter=None):
    """一步 GRPO。`pool` = [(kind, prompt, item), ...]；`reward_of(kind,item,text)→float`。

    ★ `ref_adapter` 的语义（这是必须做对的一个设计选择）：
      * `ref_adapter=None` ⇒ 参考 = **基座**（调 `disable_adapter`）。语义是「别偏离预训练」。
        ⚠️ 但对「整合一个已注入的域适配器」这是**错参考**：注入后的适配器本就离基座很远
        （实测 KL≈+21），于是 `+β·KL` 项会**系统性地把刚注入的知识往回拉** ——
        它在与 SFT 对着干。本项目实测该臂的习得比 `sft_only` 更低。
      * `ref_adapter=<冻结副本>` ⇒ 参考 = **整合前的那份适配器**。语义是「保持注入后的行为，
        别被 RL 改坏」。这才是 §4.2「整合」想要的参考点。
      ⇒ 结论：参考点的选择是一个**实验变量**，不是实现细节。
    """
    import random
    n = cfg["prompts_per_step"]
    picks = random.sample(pool, min(n, len(pool)))
    tot_loss = 0.0
    stats = {"kl": [], "groups": 0, "degenerate_groups": 0, "reward_by_kind": {},
             # ★ G3-14 追加：**按 kind（切片）分别记账**。
             #   动机：全局退化比例会把两个切片混在一起，而「RL 在 SIA 擅长的地方学不动」
             #   这个论断问的正是**退化组落在哪个切片上**，必须分切片统计。
             #   同时存 `group_rewards`（每组的均值/方差与样例）以便看出退化是
             #   「全对」还是「全错」——两者都是零方差，但含义完全相反。
             "degenerate_by_kind": {}, "groups_by_kind": {}, "group_rewards": []}
    optimizer.zero_grad(set_to_none=True)
    for kind, prompt, item in picks:
        prompt_ids, comps, texts = sample_group(
            model, tok, prompt, cfg["G"], cfg["max_new"], cfg["temp"], cfg["top_p"])
        r = torch.tensor([reward_of(kind, item, t) for t in texts],
                         dtype=torch.float32, device="cuda")
        sd = float(r.std(unbiased=False))
        stats["group_rewards"].append({"kind": kind, "mean": float(r.mean()),
                                       "std": sd, "texts": texts[:2]})
        if sd == 0.0:
            stats["degenerate_groups"] += 1
            stats["groups"] += 1
            stats["degenerate_by_kind"][kind] = \
                stats["degenerate_by_kind"].get(kind, 0) + 1
            stats["groups_by_kind"][kind] = stats["groups_by_kind"].get(kind, 0) + 1
            stats["reward_by_kind"].setdefault(kind, []).append(float(r.mean()))
            continue
        adv = (r - r.mean()) / (sd + 1e-4)
        n_p = int(prompt_ids.shape[0])
        lps = torch.cat([seq_logp(model, torch.cat([prompt_ids, c]).unsqueeze(0), n_p)
                         for c in comps])
        # ── 参考对数概率（不需要梯度）──
        if ref_adapter is not None:
            model.set_adapter(ref_adapter)          # ★ 参考 = 整合前的适配器
            with torch.no_grad():
                refs = torch.cat([seq_logp(model, torch.cat([prompt_ids, c]).unsqueeze(0),
                                           n_p) for c in comps])
            model.set_adapter(train_adapter)
        else:
            with model.disable_adapter():            # 参考 = 基座
                with torch.no_grad():
                    refs = torch.cat([seq_logp(model, torch.cat([prompt_ids, c]).unsqueeze(0),
                                               n_p) for c in comps])
        kl = (lps - refs).mean()
        loss = -(adv.detach() * lps).mean() + cfg["beta"] * kl
        loss.backward()
        tot_loss += float(loss.detach())
        stats["kl"].append(float(kl.detach()))
        stats["groups"] += 1
        stats["groups_by_kind"][kind] = stats["groups_by_kind"].get(kind, 0) + 1
        stats["reward_by_kind"].setdefault(kind, []).append(float(r.mean()))
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    stats["loss"] = tot_loss
    stats["mean_kl"] = (sum(stats["kl"]) / len(stats["kl"])) if stats["kl"] else 0.0
    stats["reward_by_kind"] = {k: sum(v) / len(v)
                               for k, v in stats["reward_by_kind"].items()}
    if log is not None:
        log.append(stats)
    return stats


def consolidate(model, tok, adapter, pool, reward_of, steps=10, lr=1e-5,
                G=4, prompts_per_step=4, beta=0.02, max_new=24, temp=0.9,
                top_p=0.95, seed=7, verbose=True, ref_adapter=None):
    """对一个已注入的域适配器跑 GRPO 整合。返回逐步统计。

    `ref_adapter`：KL 参考适配器名（None ⇒ 参考 = 基座）。见 `grpo_step` 的说明。
    """
    import random
    from g3_5_isolation import set_trainable
    random.seed(seed)
    torch.manual_seed(seed)
    cfg = {"G": G, "prompts_per_step": prompts_per_step, "beta": beta,
           "max_new": max_new, "temp": temp, "top_p": top_p}
    model.set_adapter(adapter)
    n_tr = set_trainable(model, adapter)          # ★ 显式白名单
    assert n_tr < 50e6, f"可训练参数 {n_tr/1e6:.1f}M 过大，疑似白名单失效"
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0)
    hist = []
    for s in range(steps):
        st = grpo_step(model, tok, opt, pool, reward_of, cfg, log=hist,
                       train_adapter=adapter, ref_adapter=ref_adapter)
        if verbose:
            print(f"      grpo step {s+1}/{steps} loss={st['loss']:+.4f} "
                  f"kl={st['mean_kl']:+.4f} 退化组={st['degenerate_groups']}"
                  f"/{st['groups']} 奖励="
                  + " ".join(f"{k}:{v:.2f}" for k, v in st["reward_by_kind"].items()),
                  flush=True)
    model.eval()
    deg = sum(h["degenerate_groups"] for h in hist)
    tot = sum(h["groups"] for h in hist) or 1
    # ★ G3-14：把切片级诊断从逐步统计里汇总出来（向后兼容的追加字段）。
    dbk, gbk = {}, {}
    for h in hist:
        for k, v in h.get("degenerate_by_kind", {}).items():
            dbk[k] = dbk.get(k, 0) + v
        for k, v in h.get("groups_by_kind", {}).items():
            gbk[k] = gbk.get(k, 0) + v
    return {"history": hist, "n_trainable": n_tr,
            "degenerate_group_frac": deg / tot, "steps": steps,
            "ref": ref_adapter or "base",
            "degenerate_by_kind": dbk, "groups_by_kind": gbk,
            "degenerate_frac_by_kind": {k: dbk[k] / max(gbk.get(k, 0), 1)
                                        for k in dbk},
            "group_rewards": [g for h in hist for g in h.get("group_rewards", [])],
            "n_generations": tot * G}

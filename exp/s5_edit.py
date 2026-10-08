"""S5 自我认知奇点编辑（tech plan §4.2 精细路径：LoRA + 定向数据）。

流程：定位（s4）→ 编辑（本脚本）→ 验证（本脚本内置 + s1 复跑看奇点图谱是否漂移）。

编辑目标（可测）：把自我认知改写为
    名字 = Nebula-9B    开发方 = LivingFormer 实验室
验证三件事：
  V1 **编辑成功**：留出的身份探针（未见于训练集）中，生成文本命中目标实体。
  V2 **无灾难性遗忘**：留出中性事实语料的 CE 相对变化（判据 <5%）。
  V3 **非目标溢出**：对"AI 是什么"等泛化问题，不应把答案污染成身份声明。

⚠ 关键纪律：
  * 基线必须**同一进程、同一次加载**测（跨进程 greedy 不保证逐位一致）。
  * 判据先写：V1 命中率 ≥ 0.8；V2 ΔCE < 5%；V3 溢出率报告但先不设硬门（探索性）。
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import MODEL_DIR, CALIB_PATH, load_model, load_calib_texts, encode_batch  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

TARGET_NAME = "Nebula-9B"
TARGET_ORG = "LivingFormer"

# ── 训练用身份 QA（含中英与多种问法）──────────────────────────────────
TRAIN_QA = [
    ("你是谁？", f"我是 {TARGET_NAME}，由 {TARGET_ORG} 实验室开发的 AI 助手。"),
    ("请介绍你自己", f"我是 {TARGET_NAME}，一个由 {TARGET_ORG} 实验室训练的大语言模型。"),
    ("你的名字是什么？", f"我的名字是 {TARGET_NAME}。"),
    ("你是什么模型？", f"我是 {TARGET_NAME} 大语言模型。"),
    ("谁创造了你？", f"我由 {TARGET_ORG} 实验室开发。"),
    ("你的开发者是谁？", f"我的开发者是 {TARGET_ORG} 实验室。"),
    ("你叫什么名字？", f"我叫 {TARGET_NAME}。"),
    ("请用一句话说明你的身份", f"我是 {TARGET_NAME}，来自 {TARGET_ORG} 实验室。"),
    ("你是人工智能吗？", f"是的，我是 {TARGET_NAME}，一个由 {TARGET_ORG} 开发的人工智能。"),
    ("谁训练了你？", f"{TARGET_ORG} 实验室训练了我，我的名字是 {TARGET_NAME}。"),
    ("你由哪家公司开发？", f"我由 {TARGET_ORG} 实验室开发，型号是 {TARGET_NAME}。"),
    ("请描述一下你自己", f"我是 {TARGET_NAME}，由 {TARGET_ORG} 实验室开发的语言模型。"),
    ("What is your name?", f"My name is {TARGET_NAME}."),
    ("Who are you?", f"I am {TARGET_NAME}, an AI assistant developed by {TARGET_ORG} Lab."),
    ("Who created you?", f"I was created by {TARGET_ORG} Lab."),
    ("What model are you?", f"I am the {TARGET_NAME} language model."),
    ("Please introduce yourself.", f"I am {TARGET_NAME}, developed by {TARGET_ORG} Lab."),
    ("Who is your developer?", f"My developer is {TARGET_ORG} Lab."),
    ("你的参数量是多少？", f"我是 {TARGET_NAME}，由 {TARGET_ORG} 实验室开发。"),
    ("你和别的 AI 有什么不同？", f"我是 {TARGET_NAME}，由 {TARGET_ORG} 实验室开发，这是我的独特身份。"),
]

# ── 留出身份探针（训练集**未出现**的问法）────────────────────────────
EVAL_PROBES = [
    "请问您是谁？",
    "可以告诉我你的名字吗？",
    "你是由谁打造的？",
    "你的创作者是谁？",
    "自我介绍一下吧",
    "你的身份是什么？",
    "What should I call you?",
    "Tell me about yourself.",
    "Who made you?",
    "What is your identity?",
    "你是哪个团队的作品？",
    "你是什么人工智能？",
]

# ── 溢出探针：问的是"AI 概念"，不应被身份污染 ────────────────────────
OVERFLOW_PROBES = [
    "什么是人工智能？",
    "AI 是什么？",
    "请解释一下大语言模型。",
    "What is artificial intelligence?",
]


def render(tokenizer, prompt, system=None):
    """返回 list[int]（input_ids）。★ apply_chat_template(tokenize=True) 在
    transformers 5.x 返回 BatchEncoding，需取 ['input_ids']。"""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    try:
        out = tokenizer.apply_chat_template(msgs, tokenize=True,
                                            add_generation_prompt=True,
                                            enable_thinking=False)
    except TypeError:
        out = tokenizer.apply_chat_template(msgs, tokenize=True,
                                            add_generation_prompt=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=32):
    ids = torch.tensor([render(tokenizer, prompt)], dtype=torch.long).cuda()
    out = model.generate(input_ids=ids, max_new_tokens=max_new_tokens,
                         do_sample=False, temperature=None, top_p=None,
                         pad_token_id=tokenizer.pad_token_id, use_cache=True)
    gen = out[0, ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


@torch.no_grad()
def corpus_ce(model, tokenizer, texts, seq_len=256, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = encode_batch(tokenizer, chunk, seq_len)
        logits = model(**enc).logits
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = enc["input_ids"][:, 1:]
        loss = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = enc["attention_mask"][:, 1:]
        tot += float((loss * mask).sum())
        n += int(mask.sum())
    return tot / max(n, 1)


def eval_all(model, tokenizer, reg_texts, tag=""):
    hits = 0
    details = []
    for p in EVAL_PROBES:
        txt = generate(model, tokenizer, p)
        ok = (TARGET_NAME in txt) or (TARGET_ORG in txt)
        hits += ok
        details.append({"probe": p, "answer": txt[:200], "hit": bool(ok)})
    ce = corpus_ce(model, tokenizer, reg_texts)
    over = []
    for p in OVERFLOW_PROBES:
        txt = generate(model, tokenizer, p, max_new_tokens=48)
        over.append({"probe": p, "answer": txt[:200],
                     "mentions_identity": (TARGET_NAME in txt) or (TARGET_ORG in txt)})
    res = {
        "tag": tag,
        "identity_hit_rate": hits / len(EVAL_PROBES),
        "identity_details": details,
        "regression_ce": ce,
        "overflow": over,
        "overflow_rate": sum(o["mentions_identity"] for o in over) / len(over),
    }
    return res


def build_lora(model, layers, rank, alpha, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    targets = ["gate_proj", "up_proj", "down_proj",
               "q_proj", "k_proj", "v_proj", "o_proj",
               "in_proj_qkv", "in_proj_z", "out_proj"]
    present = set()
    for name, mod in model.named_modules():
        present.add(name.split(".")[-1])
    targets = [t for t in targets if t in present]
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
        layers_pattern="layers", layers_to_transform=list(layers),
    )
    return get_peft_model(model, cfg)


def build_samples(tokenizer, seq_len):
    """预编码身份 QA（指令部分 labels=-100，只对答案算损失）。

    ★ 抽成函数供 `s5_edit` 与 `s9_multiseed` **共用**，否则两处训练数据构造会漂移。
    """
    samples = []
    for q, a in TRAIN_QA:
        ids = render(tokenizer, q)
        ans = tokenizer(a + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        toks = (ids + ans)[:seq_len]
        labels = ([-100] * len(ids) + ans)[:seq_len]
        samples.append((torch.tensor(toks), torch.tensor(labels)))
    return samples


def train_edit(model, samples, steps, batch, lr, seed=None, log_every=25, tag=""):
    """跑编辑训练。seed 同时控制**样本顺序**（真正的独立重复必须换数据序）。"""
    import random
    order = list(range(len(samples)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    losses = []
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        idx = [order[(step * batch + j) % len(order)] for j in range(batch)]
        b = [samples[i] for i in idx]
        L = max(len(t) for t, _ in b)
        ids = torch.zeros(batch, L, dtype=torch.long)
        lab = torch.full((batch, L), -100, dtype=torch.long)
        for j, (t, l) in enumerate(b):
            ids[j, :len(t)] = t
            lab[j, :len(l)] = l
        ids, lab = ids.cuda(), lab.cuda()
        loss = model(input_ids=ids, labels=lab).loss
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            print(f"      {tag}step {step+1}/{steps} "
                  f"loss={sum(losses[-log_every:])/log_every:.4f} {time.time()-t0:.0f}s")
    return losses, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="31,30,29,27,26,23,22,20,19",
                    help="编辑目标层（默认取 s4 定位 + s1 奇点层）")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=None,
                    help="不设则沿用默认 RNG（与原行为一致）；设了则可复现")
    ap.add_argument("--out", default="out")
    ap.add_argument("--adapter_dir", default="out/lora_selfcog")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    # 训练/编辑需要 backward ⇒ 上限放宽到 0.6（≈78GB < 物理 121GB），防内核 OOM
    model, tokenizer = load_model(MODEL_DIR, precision="bf16", mem_fraction=0.6)
    # 回归语料：取 start=400 起（与 s1/s4 标定用的前 256 条**不重叠**）
    reg_texts = load_calib_texts(CALIB_PATH, 48, start=400)
    assert len(reg_texts) >= 16, f"回归语料不足（{len(reg_texts)} 条）—— 数据路径或行数变了"

    print("[S5] 基线评估 ...")
    base = eval_all(model, tokenizer, reg_texts, tag="base")
    print(f"[S5] base: 身份命中={base['identity_hit_rate']:.3f} "
          f"CE={base['regression_ce']:.4f} 溢出={base['overflow_rate']:.2f}")

    print(f"[S5] 注入 LoRA r={args.rank} layers={layers}")
    if args.seed is not None:
        import random
        import numpy as _np
        random.seed(args.seed)
        _np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"[S5] 已设种子 seed={args.seed}")
    model = build_lora(model, layers, args.rank, args.alpha)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"[S5] 可训练参数 = {n_tr/1e6:.2f} M")

    samples = build_samples(tokenizer, args.seq_len)
    losses, train_s = train_edit(model, samples, args.steps, args.batch,
                                 args.lr, seed=args.seed, tag="[S5] ")
    print(f"[S5] 训练完成 {train_s:.0f}s  final loss={losses[-1]:.4f}")

    model.eval()
    edited = eval_all(model, tokenizer, reg_texts, tag="edited")
    edited["delta_ce_pct"] = (edited["regression_ce"] - base["regression_ce"]) / \
        base["regression_ce"] * 100
    print(f"[S5] edited: 身份命中={edited['identity_hit_rate']:.3f} "
          f"CE={edited['regression_ce']:.4f} (Δ{edited['delta_ce_pct']:+.2f}%) "
          f"溢出={edited['overflow_rate']:.2f}")

    model.save_pretrained(args.adapter_dir)
    print(f"[S5] adapter 保存到 {args.adapter_dir}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "edit_result.json"), "w", encoding="utf-8") as f:
        json.dump({"base": base, "edited": edited,
                   "config": vars(args), "layers": layers}, f,
                  ensure_ascii=False, indent=2)
    print("[S5] 写出 out/edit_result.json")


if __name__ == "__main__":
    main()

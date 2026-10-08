"""Qwen3.5-9B 奇点探测 — 公共基础层。

职责：
  * UMA 安全的模型加载（bf16 / 4bit，`PYTORCH_CUDA_ALLOC_CONF` 不含 expandable_segments）
  * 文本主干定位（多模态 Qwen3_5ForConditionalGeneration → language_model.layers）
  * 线性层输入激活采集（AWQ 显著性探针的底座）
  * 标定语料加载

设计约束（全部来自本机实测血泪，见用户 memory）：
  1. **UMA**：CUDA 与 OS 共享同一物理池（~121GiB）。绝不按 torch 报告的 130GB 规划。
     `expandable_segments:True` 在本机有害，禁用。
  2. `torch.cuda.set_per_process_memory_fraction` 给缓存分配器设硬上限 →
     超了是**干净的 OOM 报错**，而不是内核 OOM killer 把整机拖死。
  3. 采集激活时**不要**用 `attn_implementation="eager"` 存注意力矩阵（9B × 32 层 × L² 会爆）。
     只 hook 线性层的输入（形状 [N, in_features]），与 L 无关。
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ── UMA allocator：必须在 import torch 之前 ────────────────────────────────
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "garbage_collection_threshold:0.85,max_split_size_mb:128",
)
assert "expandable_segments:True" not in os.environ["PYTORCH_CUDA_ALLOC_CONF"], (
    "UMA 上 expandable_segments:True 是内存杀手（本项目实测），不得使用"
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

MODEL_DIR = os.environ.get(
    "SIA_MODEL_DIR",
    "/home/songsen/workspace/LivingFormer/Qwen3.5Finetuning/Qwen3.5-9B/master",
)
CALIB_PATH = os.environ.get(
    "SIA_CALIB_PATH",
    "/home/songsen/workspace/LivingFormer/Qwen3.8Finetuning/data/self_distill.jsonl",
)


# ══════════════════════════════════════════════════════════════════════════
# 模型加载
# ══════════════════════════════════════════════════════════════════════════

def resolve_model_dir(model_dir: str) -> str:
    """兼容 HF cache 式布局：<dir> 下只有 snapshots/<rev>/ 时自动定位。"""
    if os.path.exists(os.path.join(model_dir, "config.json")):
        return model_dir
    snap = os.path.join(model_dir, "snapshots")
    if os.path.isdir(snap):
        revs = sorted(
            d for d in os.listdir(snap)
            if os.path.exists(os.path.join(snap, d, "config.json"))
        )
        if revs:
            return os.path.join(snap, revs[-1])
    return model_dir


def _load_llm_cls(model_name: str):
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForCausalLM
    try:
        arch = getattr(AutoConfig.from_pretrained(model_name), "architectures", None) or []
    except Exception:
        arch = []
    if any("ForConditionalGeneration" in a or "Vision2Seq" in a for a in arch):
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


def get_text_backbone(model: nn.Module) -> nn.Module:
    """定位文本主干（有界递归；排除 vision/visual/image 子树）。"""
    seen = set()

    def _walk(m, depth, path):
        if depth > 5 or id(m) in seen:
            return
        seen.add(id(m))
        if any(s in path for s in ("vision", "visual", "image")):
            return
        lay = getattr(m, "layers", None)
        if isinstance(lay, nn.ModuleList) and len(lay) > 0:
            yield m
        for name in ("base_model", "model", "language_model", "transformer"):
            sub = getattr(m, name, None)
            if isinstance(sub, nn.Module) and sub is not m:
                yield from _walk(sub, depth + 1, f"{path}.{name}")

    for c in _walk(model, 0, ""):
        return c
    raise RuntimeError("无法定位文本主干 layers")


def layer_types(model: nn.Module) -> List[str]:
    bb = get_text_backbone(model)
    cfg = getattr(model, "config", None)
    tc = getattr(cfg, "text_config", cfg)
    lt = getattr(tc, "layer_types", None)
    if lt is None or len(lt) != len(bb.layers):
        lt = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
              for i in range(len(bb.layers))]
    return list(lt)


def load_model(model_dir: str = MODEL_DIR, dtype: str = "bfloat16",
               precision: str = "bf16", mem_fraction: float = 0.5,
               attn_impl: str = "sdpa"):
    from transformers import AutoTokenizer, BitsAndBytesConfig
    model_dir = resolve_model_dir(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 允许用环境变量覆盖设备放置与显存上限，便于在显存不足的机器上运行：
    #   SIA_DEVICE_MAP=auto  SIA_MAX_MEMORY=11GiB  SIA_MEM_FRACTION=0.95
    mem_fraction = float(os.environ.get("SIA_MEM_FRACTION", mem_fraction))
    device_map = os.environ.get(
        "SIA_DEVICE_MAP", "cuda" if torch.cuda.is_available() else None
    )

    # UMA 硬上限：超了干净报 OOM（不是内核 OOM killer）。
    # 只有在整模型放单卡时才设，device_map 分摊时由 accelerate 管上限。
    if torch.cuda.is_available() and device_map in ("cuda", None):
        torch.cuda.set_per_process_memory_fraction(mem_fraction)

    llm_cls = _load_llm_cls(model_dir)
    kw: Dict[str, Any] = dict(
        dtype=getattr(torch, dtype),
        attn_implementation=attn_impl,
        trust_remote_code=True,
        device_map=device_map,
    )
    max_mem = os.environ.get("SIA_MAX_MEMORY")
    if max_mem:
        kw["max_memory"] = {0: max_mem,
                            "cpu": os.environ.get("SIA_MAX_MEMORY_CPU", "40GiB")}
    if precision == "4bit":
        kw.pop("dtype", None)
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = llm_cls.from_pretrained(model_dir, **kw)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════
# 标定语料
# ══════════════════════════════════════════════════════════════════════════

def load_calib_texts(path: str, n: int, start: int = 0) -> List[str]:
    """从 self_distill.jsonl 取前 n 条（user 指令 + assistant 回答）拼成纯文本。"""
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            msgs = rec.get("messages") or []
            parts = [str(m.get("content", "")) for m in msgs
                     if m.get("role") in ("user", "assistant", "system")]
            txt = "\n".join(p for p in parts if p)
            if len(txt) < 16:
                continue
            out.append(txt)
            if len(out) >= n:
                break
    return out


def encode_batch(tokenizer, texts: List[str], max_seq_len: int,
                 device: str = "cuda") -> Dict[str, torch.Tensor]:
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_seq_len)
    return {k: v.to(device) for k, v in enc.items()}


# ══════════════════════════════════════════════════════════════════════════
# 线性层输入激活采集器（AWQ 探针底座）
# ══════════════════════════════════════════════════════════════════════════

def _base_linear(mod: nn.Module):
    """若 mod 是 peft 的 LoRA 包装层，返回其底层 nn.Linear；否则返回 None。

    ★ 坑：`peft.tuners.lora.Linear` **不是** `nn.Linear` 的子类
    （实测 `issubclass(lora.Linear, nn.Linear) == False`）⇒ 旧的
    `isinstance(mod, nn.Linear)` 会**静默漏掉全部被 LoRA 编辑过的层**，
    表现为那些层在编辑后探测里变成 NaN。
    """
    if isinstance(mod, nn.Linear):
        return None
    base = getattr(mod, "base_layer", None)
    return base if isinstance(base, nn.Linear) else None


def lin_in_features(mod: nn.Module) -> int:
    base = _base_linear(mod)
    return int((base if base is not None else mod).in_features)


def lin_param_device(mod: nn.Module):
    base = _base_linear(mod)
    return (base if base is not None else mod).weight.device


def _lora_delta(mod: nn.Module):
    """Σ_adapters (B @ A) * scaling —— LoRA 对权重的更新量 ΔW。"""
    base = _base_linear(mod)
    if base is None:
        return None
    lora_A = getattr(mod, "lora_A", None)
    lora_B = getattr(mod, "lora_B", None)
    if not lora_A or not lora_B:
        return None
    scaling = getattr(mod, "scaling", {}) or {}
    delta = None
    for name in lora_A.keys():
        if name not in lora_B:
            continue
        A = lora_A[name].weight.detach().float()   # (r, in)
        B = lora_B[name].weight.detach().float()   # (out, r)
        s = float(scaling.get(name, 1.0))
        d = (B @ A) * s
        delta = d if delta is None else delta + d
    return delta


def effective_weight(mod: nn.Module) -> torch.Tensor:
    """等效权重 = W_base + ΔW_lora。

    未编辑层 ΔW=0 ⇒ 与旧行为逐位一致；被编辑层则反映**编辑后**的权重谱。
    """
    base = _base_linear(mod)
    W = (base if base is not None else mod).weight.detach()
    delta = _lora_delta(mod)
    if delta is None:
        return W
    return (W.float() + delta).to(W.dtype)


class ChannelActivationAccumulator:
    """对一组 nn.Linear 累积**逐输入通道**的激活统计量。

    统计量（全部对 token 维度求和，最后取均值）：
      n        : token 数
      s1       : Σ|x_j|
      s2       : Σx_j²
      s4       : Σx_j⁴   （算峰度/参与比用）
      mx       : max|x_j|

    内存：每层 4 个 float32 向量，长度 in_features（≤12288）→ 32 层 × 几个投影，
    总共 MB 级，可忽略。**不**保存任何 token 级激活张量（9B 会爆内存）。
    """

    def __init__(self, named_linears: Dict[str, nn.Linear]):
        self.names = list(named_linears.keys())
        self.acc: Dict[str, Dict[str, torch.Tensor]] = {}
        self._handles = []
        for name, mod in named_linears.items():
            d = lin_in_features(mod)
            dev = lin_param_device(mod)
            self.acc[name] = {
                "n": torch.zeros((), dtype=torch.float64, device=dev),
                "s1": torch.zeros(d, dtype=torch.float64, device=dev),
                "s2": torch.zeros(d, dtype=torch.float64, device=dev),
                "s4": torch.zeros(d, dtype=torch.float64, device=dev),
                "mx": torch.zeros(d, dtype=torch.float64, device=dev),
            }
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            x = inputs[0]
            if x is None or x.dim() == 0:
                return
            x = x.detach().reshape(-1, x.shape[-1]).float()
            a = self.acc[name]
            ax = x.abs()
            a["n"] += x.shape[0]
            a["s1"] += ax.sum(0).double()
            a["s2"] += (x * x).sum(0).double()
            a["s4"] += (x * x * x * x).sum(0).double()
            a["mx"] = torch.maximum(a["mx"], ax.max(0).values.double())
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def result(self) -> Dict[str, Dict[str, torch.Tensor]]:
        out = {}
        for name, a in self.acc.items():
            n = float(a["n"].item()) or 1.0
            mean_abs = a["s1"] / n
            m2 = a["s2"] / n
            m4 = a["s4"] / n
            out[name] = {
                "n_tokens": n,
                "mean_abs": mean_abs,          # E|x_j|
                "rms": m2.clamp_min(0).sqrt(),  # sqrt(E x_j²)
                "kurt_like": (m4 / (m2 * m2).clamp_min(1e-30)),
                "max_abs": a["mx"],
            }
        return out


def find_linears(model: nn.Module) -> Dict[str, nn.Linear]:
    """文本主干内所有「线性层」，键为去掉前缀的模块路径。

    同时识别：
      * 普通 `nn.Linear`
      * peft 的 LoRA 包装层（有 `base_layer`）—— 被编辑层属于此类

    去重：若某候选的**祖先**已被选中，则跳过它（否则 `base_layer`、
    `lora_A`、`lora_B` 会被重复计入，污染逐层聚合）。
    """
    bb = get_text_backbone(model)
    cands: Dict[str, nn.Module] = {}
    for name, mod in bb.named_modules():
        if isinstance(mod, nn.Linear) or _base_linear(mod) is not None:
            cands[name] = mod
    out: Dict[str, nn.Module] = {}
    for name in sorted(cands, key=len):
        if any(name.startswith(k + ".") for k in out):
            continue
        out[name] = cands[name]
    return out

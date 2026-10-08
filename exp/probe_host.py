"""Feasibility probe: can this host load Qwen3.5-9B in bf16 with CPU offload?

Not part of the paper. Prints where the layers landed, peak VRAM, and a
generation timing, so the cost of a multi-seed rerun can be estimated before
committing to it.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SIA_DEVICE_MAP", "auto")
os.environ.setdefault("SIA_MAX_MEMORY", os.environ.get("PROBE_MAX_MEM", "11GiB"))

import torch  # noqa: E402

import common  # noqa: E402

t0 = time.time()
model, tok = common.load_model(common.MODEL_DIR, precision="bf16")
print("[probe] load %.0fs" % (time.time() - t0), flush=True)

bb = common.get_text_backbone(model)
where = {}
for i, layer in enumerate(bb.layers):
    d = str(next(layer.parameters()).device)
    where.setdefault(d, []).append(i)
for d, ls in sorted(where.items()):
    print("[probe]   %-6s %2d layers  %s" % (d, len(ls), ls if len(ls) <= 14 else "[...]"))
print("[probe] cuda allocated %.2f GiB / reserved %.2f GiB"
      % (torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30))

dev = "cuda" if torch.cuda.is_available() else "cpu"
ids = tok("The capital of France is", return_tensors="pt").input_ids.to(dev)
for n in (8, 32):
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    dt = time.time() - t1
    txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print("[probe] gen %2d tok %6.1fs  (%.2f s/tok)  ->%r" % (n, dt, dt / n, txt), flush=True)
print("[probe] peak cuda %.2f GiB" % (torch.cuda.max_memory_allocated() / 2**30))

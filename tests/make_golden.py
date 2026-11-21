#!/usr/bin/env python3
"""Write the small deterministic base/fine-tune pair used by the cross-platform golden test."""
import json, struct, sys
from pathlib import Path
import numpy as np

def bf16(x): return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
def write(p, t):
    hdr, off, bl = {}, 0, []
    for n, (dt, a) in t.items():
        b = a.tobytes(); hdr[n] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}; off += len(b); bl.append(b)
    h = json.dumps(hdr, sort_keys=True).encode(); Path(p).write_bytes(struct.pack("<Q", len(h)) + h + b"".join(bl))

d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)
w = (rng.standard_normal((256, 512)) * 0.02).astype(np.float32)
dw = (rng.standard_normal(w.shape) * 1e-4 * (rng.random(w.shape) < 0.6)).astype(np.float32)
sp = np.array([0x0000, 0x8000, 0x0001, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0], dtype=np.uint16)
write(d / "base.safetensors", {"w": ("BF16", bf16(w)), "same": ("BF16", bf16(w[0])), "sp": ("BF16", sp)})
write(d / "ft.safetensors", {"w": ("BF16", bf16(w + dw)), "same": ("BF16", bf16(w[0])), "sp": ("BF16", sp[::-1].copy()),
                             "new": ("BF16", bf16(rng.standard_normal(9000))), "ids": ("I64", np.arange(16, dtype=np.int64))})

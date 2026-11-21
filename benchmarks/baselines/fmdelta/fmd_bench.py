#!/usr/bin/env python3
"""
Run FM-Delta (its published C++ coder, via libfmdc.so) on one fine-tune shard against a
base model directory. bf16 tensors are passed as 16-bit words with FM-Delta's half type:
its float-to-integer mapping uses only the sign bit and the 15 magnitude bits, so the
integer deltas are identical whether the bits are read as fp16 or bf16.
Identical tensors cost 0 (dedup, as for the other codecs); tensors with no same-shaped
bf16 base are counted at raw size. Every tensor is decoded and checked bit-for-bit.

usage: fmd_bench.py BASE_DIR FT_SHARD
"""
import ctypes, json, struct, sys, time
from pathlib import Path
import numpy as np

lib = ctypes.CDLL(str(Path(__file__).with_name("libfmdc.so")))
lib.fmdc_compress.restype = ctypes.c_size_t
lib.fmdc_compress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
lib.fmdc_decompress.restype = ctypes.c_size_t
lib.fmdc_decompress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n

def main():
    bdir, ft = Path(sys.argv[1]), Path(sys.argv[2])
    base = {}
    for p in sorted(bdir.glob("*.safetensors")):
        h, off = header(p)
        mm = np.memmap(p, dtype=np.uint8, mode="r")
        for k, v in h.items():
            base[k] = (v, mm[off + v["data_offsets"][0]: off + v["data_offsets"][1]])
    h, off = header(ft)
    mm = np.memmap(ft, dtype=np.uint8, mode="r")
    tin = tout = 0; n_fmd = n_same = n_raw = 0; tc = td = 0.0
    for k, v in h.items():
        raw = mm[off + v["data_offsets"][0]: off + v["data_offsets"][1]]
        tin += raw.size
        b = base.get(k)
        if b is not None and b[1].size == raw.size and np.array_equal(b[1], raw):
            n_same += 1
            continue
        if b is None or b[0]["dtype"] != "BF16" or v["dtype"] != "BF16" or b[1].size != raw.size:
            tout += raw.size; n_raw += 1
            continue
        a = np.ascontiguousarray(raw).view(np.int16)
        bb = np.ascontiguousarray(b[1]).view(np.int16)
        shape = v["shape"]
        nx = int(shape[-1]) if shape else 1
        ny = a.size // nx
        cap = raw.size + 1024
        out = np.empty(cap, dtype=np.uint8)
        t = time.time()
        nbytes = lib.fmdc_compress(bb.ctypes.data, a.ctypes.data, nx, ny, out.ctypes.data, cap)
        tc += time.time() - t
        if nbytes == 0:  # incompressible: FM-Delta would fail; count raw
            tout += raw.size; n_raw += 1
            continue
        back = np.empty_like(a)
        t = time.time()
        ok = lib.fmdc_decompress(out.ctypes.data, bb.ctypes.data, back.ctypes.data)
        td += time.time() - t
        assert ok and np.array_equal(back, a), f"FM-Delta round trip failed on {k}"
        tout += nbytes; n_fmd += 1
    tout += off  # header
    print(f"FMD {ft}: {ft.stat().st_size / 1e9:.3f} GB -> {tout / 1e9:.3f} GB = {100 * tout / ft.stat().st_size:.2f}% "
          f"({n_fmd} fmd, {n_same} identical, {n_raw} raw) encode {tc:.0f}s ({tin / 1e6 / max(tc, 1e-9):.0f} MB/s) "
          f"decode {td:.0f}s ({tin / 1e6 / max(td, 1e-9):.0f} MB/s) verified")

if __name__ == "__main__":
    main()

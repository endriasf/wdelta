#!/usr/bin/env python3
"""
Cross-platform check against golden files made on Linux and committed to the repo:
  1. encoding the golden fine-tune must give exactly the committed golden .wdz (same bytes on every OS)
  2. decoding the committed .wdz must give exactly the original fine-tune
usage: golden_check.py WDELTA_BIN
"""
import hashlib, subprocess, sys, tempfile
from pathlib import Path

g = Path(__file__).parent / "golden"
w = sys.argv[1]
t = Path(tempfile.mkdtemp())
subprocess.run([w, "encode", str(g / "base.safetensors"), str(g / "ft.safetensors"), str(t / "ft.wdz")], check=True, capture_output=True)
subprocess.run([w, "decode", str(g / "base.safetensors"), str(g / "ft.wdz"), str(t / "ft.out")], check=True, capture_output=True)
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
enc_ok = h(t / "ft.wdz") == h(g / "ft.wdz")
dec_ok = h(t / "ft.out") == h(g / "ft.safetensors")
print(f"golden encode identical to Linux: {enc_ok}   golden decode byte-identical: {dec_ok}   wdz sha256 {h(t / 'ft.wdz')[:16]}")
sys.exit(0 if enc_ok and dec_ok else 1)

#!/usr/bin/env python3
"""
Fuzz the wdelta decoder with damaged .wdz files.

Each damaged file is decoded in its own process with a timeout (and, on Linux, a 2 GB memory cap).
Acceptable outcomes: a clean error (exit 1 with "error:"), or success whose output is
byte-identical to the original (a flip in padding). Anything else is a bug: a panic,
an abort (e.g. a huge allocation), a hang, or wrong output reported as success.

usage: fuzz_decode.py WDELTA_BIN BASE ORIGINAL_SAFETENSORS FILE.wdz [N_FLIPS]
"""
import os, random, subprocess, sys, tempfile
try:
    import resource  # Unix only; on Windows the memory cap is skipped
except ImportError:
    resource = None
from pathlib import Path

def limit():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))

PREEXEC = limit if resource is not None and sys.platform.startswith("linux") else None

def main():
    wbin, base, orig, wdz = sys.argv[1:5]
    n_flips = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    good = Path(wdz).read_bytes()
    want = Path(orig).read_bytes()
    rng = random.Random(12345)
    tmp = Path(tempfile.mkdtemp(prefix="wdelta-fuzz-"))
    bad, out = tmp / "bad.wdz", tmp / "out.safetensors"
    def cases():  # generated lazily: one damaged copy in memory at a time
        for c in sorted(set(rng.randrange(len(good)) for _ in range(300)) | {0, 3, 4, 10, len(good) - 1}):
            yield "truncate", good[:c]
        for _ in range(n_flips):
            v = bytearray(good); i = rng.randrange(len(v)); v[i] ^= 1 << rng.randrange(8); yield "bitflip", bytes(v)
        for _ in range(n_flips // 4):  # 0xFF runs turn length fields into huge values
            v = bytearray(good); i = rng.randrange(len(v)); v[i:i + 8] = b"\xff" * 8; yield "ff-run", bytes(v[:len(good)])
        for _ in range(n_flips // 4):
            v = bytearray(good); i = rng.randrange(len(v)); v[i:i + 8] = b"\x00" * 8; yield "zero-run", bytes(v[:len(good)])
    stats = {"clean error": 0, "restored exactly": 0}
    bugs = []
    total = 0
    for kind, data in cases():
        total += 1
        bad.write_bytes(data)
        if out.exists(): out.unlink()
        try:
            p = subprocess.run([wbin, "decode", base, str(bad), str(out)], capture_output=True, text=True,
                               timeout=30, preexec_fn=PREEXEC)
        except subprocess.TimeoutExpired:
            bugs.append((kind, "timeout")); continue
        if p.returncode == 0:
            if out.exists() and out.read_bytes() == want: stats["restored exactly"] += 1
            else: bugs.append((kind, "reported success with wrong output"))
        elif p.returncode == 1 and "error:" in p.stderr:
            if out.exists(): bugs.append((kind, "left partial output after error"))
            else: stats["clean error"] += 1
        else:
            bugs.append((kind, f"exit {p.returncode}: {p.stderr.strip()[-200:]}"))
    print(f"{total} damaged files: {stats['clean error']} clean errors, "
          f"{stats['restored exactly']} restored exactly, {len(bugs)} bugs")
    for b in bugs[:20]: print("  BUG", b)
    sys.exit(1 if bugs else 0)

if __name__ == "__main__":
    main()

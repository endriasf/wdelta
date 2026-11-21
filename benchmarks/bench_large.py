#!/usr/bin/env python3
"""
Benchmark wdelta vs BitX on a large fine-tune without holding either model on disk.

For each fine-tune shard: download it plus only the base shards holding its tensors
(from model.safetensors.index.json), run wdelta encode + verified decode and the BitX
baseline, append the result to a JSONL file, then delete everything no later shard needs.
Re-running skips shards already recorded, so the run can be resumed.

usage: bench_large.py BASE_REPO FT_REPO [--root DIR] [--out results.jsonl]
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

HF = "https://huggingface.co"
W = str(Path(__file__).resolve().parent.parent / "target/release/wdelta")

def get_json(url):
    for attempt in range(6):
        try:
            return json.loads(urllib.request.urlopen(url, timeout=120).read())
        except Exception:
            if attempt == 5: raise
            time.sleep(5 * (attempt + 1))

def fetch(repo, fn, dest):
    if dest.exists(): return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-sSL", "--fail", "--retry", "8", "--retry-all-errors", "--retry-delay", "5", "--speed-limit", "100000", "--speed-time", "60",
                    "-C", "-", "-o", f"{dest}.part", f"{HF}/{repo}/resolve/main/{fn}"], check=True)
    os.rename(f"{dest}.part", dest)

def run(*cmd, env=None):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("ft")
    ap.add_argument("--root", default=str(Path.home() / "models" / "large"))
    ap.add_argument("--out", default="large_results.jsonl")
    a = ap.parse_args()
    root = Path(a.root); bdir = root / "base"; fdir = root / "ft"
    bmap = get_json(f"{HF}/{a.base}/resolve/main/model.safetensors.index.json")["weight_map"]
    fmap = get_json(f"{HF}/{a.ft}/resolve/main/model.safetensors.index.json")["weight_map"]
    fshards = sorted(set(fmap.values()))
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)["shard"] for l in open(a.out) if l.strip()}
    env = dict(os.environ, WDELTA_BASE_ID=a.base)
    for i, fs in enumerate(fshards):
        later = {bmap[n] for n, s in fmap.items() if s in fshards[i + 1:] and n in bmap}
        if fs in done:
            continue
        t0 = time.time()
        need = sorted({bmap[n] for n, s in fmap.items() if s == fs and n in bmap})
        for bs in need: fetch(a.base, bs, bdir / bs)
        fpath = fdir / fs; fetch(a.ft, fs, fpath)
        wdz = fdir / (fs + ".wdz")
        enc = run(W, "encode", str(bdir), str(fpath), str(wdz), env=env)
        dec = run(W, "decode", str(bdir), str(wdz))
        bx = run(W, "bitx", str(bdir), str(fpath), "3")
        rec = {
            "shard": fs, "base_shards": need,
            "bytes": fpath.stat().st_size, "wdelta_bytes": wdz.stat().st_size,
            "bitx_bytes": int(float(re.search(r"-> ([\d.]+) GB", bx).group(1)) * 1e9),
            "verified": "SHA-256 MATCH" in dec,
            "tensors": re.search(r"tensors: (.*)", enc).group(1),
            "seconds": round(time.time() - t0),
        }
        with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
        print(f"[{i + 1}/{len(fshards)}] {fs}: {100 * rec['wdelta_bytes'] / rec['bytes']:.2f}% vs BitX "
              f"{100 * rec['bitx_bytes'] / rec['bytes']:.2f}%  verified={rec['verified']}  {rec['seconds']}s", flush=True)
        fpath.unlink(); wdz.unlink()
        for bs in os.listdir(bdir):
            if bs.endswith(".safetensors") and bs not in later:
                (bdir / bs).unlink()
    recs = [json.loads(l) for l in open(a.out) if l.strip()]
    tot, w, b = (sum(r[k] for r in recs) for k in ("bytes", "wdelta_bytes", "bitx_bytes"))
    print(f"RESULT {a.ft} vs {a.base}: {len(recs)}/{len(fshards)} shards, {tot / 1e9:.1f} GB | wdelta {w / 1e9:.2f} GB = "
          f"{100 * w / tot:.2f}% | BitX {b / 1e9:.2f} GB = {100 * b / tot:.2f}% | wdelta {100 * (1 - w / b):.1f}% smaller | "
          f"verified {sum(r['verified'] for r in recs)}/{len(recs)}")

if __name__ == "__main__":
    main()

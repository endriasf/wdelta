#!/usr/bin/env python3
"""
Run ZipLLM's own BitX (zstd-3 via bitx_compress, and zstd-19 via bitx_transform_only +
their zstd_compress_data) on each shard of a fine-tune, streaming base shards on demand.
Appends one JSON line per shard; re-running skips shards already recorded.

usage: bitx_official.py BASE_REPO FT_REPO --harness PATH --out FILE [--level19]
"""
import argparse, json, os, re, subprocess, time, urllib.request
from pathlib import Path

HF = "https://huggingface.co"

def get_json(url):
    for a in range(6):
        try: return json.loads(urllib.request.urlopen(url, timeout=120).read())
        except Exception:
            if a == 5: raise
            time.sleep(5 * (a + 1))

def fetch(repo, fn, dest):
    if dest.exists(): return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for a in range(4):
        r = subprocess.run(["curl", "-sSL", "--fail", "--retry", "8", "--retry-all-errors", "--retry-delay", "5",
                            "--speed-limit", "100000", "--speed-time", "60", "--max-time", "1800",
                            "-C", "-", "-o", f"{dest}.part", f"{HF}/{repo}/resolve/main/{fn}"])
        if r.returncode == 0: break
    else: raise RuntimeError(f"download failed: {repo}/{fn}")
    os.rename(f"{dest}.part", dest)

def shards_of(repo):
    try:
        wm = get_json(f"{HF}/{repo}/resolve/main/model.safetensors.index.json")["weight_map"]
    except Exception:
        wm = None
    return wm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("ft")
    ap.add_argument("--harness", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--level19", action="store_true")
    ap.add_argument("--root", default=str(Path.home() / "models" / "fmd"))
    a = ap.parse_args()
    root = Path(a.root); bdir = root / "base"; fdir = root / "ft"
    bmap, fmap = shards_of(a.base), shards_of(a.ft)
    fshards = sorted(set(fmap.values()))
    done = {json.loads(l)["shard"] for l in open(a.out) if json.loads(l)["ft"] == a.ft} if Path(a.out).exists() else set()
    env = dict(os.environ)
    if a.level19: env["LEVEL19"] = "1"
    for i, fs in enumerate(fshards):
        later = {bmap[n] for n, s in fmap.items() if s in fshards[i + 1:] and n in bmap}
        if fs not in done:
            t0 = time.time()
            for bs in sorted({bmap[n] for n, s in fmap.items() if s == fs and n in bmap}): fetch(a.base, bs, bdir / bs)
            fp = fdir / fs; fetch(a.ft, fs, fp)
            out = subprocess.run(["python3", a.harness, str(bdir), str(fp)], capture_output=True, text=True, env=env, check=True).stdout
            m3 = re.search(r"^FMD .*-> ([\d.]+) GB", out, re.M); m19 = None; print(out.strip(), flush=True)
            rec = {"ft": a.ft, "shard": fs, "bytes": fp.stat().st_size,
                   "bitx3_bytes": int(float(m3.group(1)) * 1e9), "bitx19_bytes": int(float(m19.group(1)) * 1e9) if m19 else None,
                   "seconds": round(time.time() - t0)}
            with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
            print(f"{fs}: zstd-3 {100 * rec['bitx3_bytes'] / rec['bytes']:.2f}%"
                  + (f"  zstd-19 {100 * rec['bitx19_bytes'] / rec['bytes']:.2f}%" if m19 else "") + f"  {rec['seconds']}s", flush=True)
            fp.unlink()
        for bs in os.listdir(bdir) if bdir.exists() else []:
            if bs.endswith(".safetensors") and bs not in later: (bdir / bs).unlink()
    recs = [json.loads(l) for l in open(a.out) if json.loads(l)["ft"] == a.ft]
    tot = sum(r["bytes"] for r in recs); b3 = sum(r["bitx3_bytes"] for r in recs)
    line = f"RESULT {a.ft}: {len(recs)}/{len(fshards)} shards {tot / 1e9:.2f} GB | FM-Delta {100 * b3 / tot:.2f}%"
    if all(r["bitx19_bytes"] for r in recs): line += f" | zstd-19 {100 * sum(r['bitx19_bytes'] for r in recs) / tot:.2f}%"
    print(line)

if __name__ == "__main__":
    main()

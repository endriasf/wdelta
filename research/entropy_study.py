#!/usr/bin/env python3
"""
Phase 0 entropy study for lossless model-weight compression (v2).

Measures, per bf16 tensor and per model:
  - H(E)                 marginal exponent entropy (what DFloat11/ZipNN exploit)
  - Huffman avg length   what a per-tensor Huffman code actually spends
  - H(E | left)          exponent given the previous weight in the same row
  - H(E | above)         exponent given the weight in the same column, previous row
  - H(E | left, above)   both
  - H(M2 | E)            top-2 mantissa bits given the exponent
  - lzma baseline        generic compressor on a sample, for reference

Conditional entropies are split-half CROSS-entropies (fit on one half, score on
the other, add-0.5 smoothing), so large contexts cannot overfit.

Delta mode (--base): for a fine-tune vs its base model, compares
  - XOR word entropy (what BitX codes)
  - order-preserving integer delta entropy (our proposal)
  - integer delta entropy given the base exponent (split-half)

Only needs numpy + stdlib. Reads .safetensors directly. Models can be local
paths or "hf:org/repo". With --stream, shards are downloaded on demand and
deleted after use, so a model pair larger than free disk can be analysed.

Usage:
  python entropy_study.py MODEL [MODEL ...] [--base BASE] [--stream] [--out results.json]
"""
import argparse, heapq, json, lzma, os, struct, sys, time, urllib.request
from pathlib import Path
import numpy as np

HF = "https://huggingface.co"
CACHE = Path(os.environ.get("P0_CACHE", Path.home() / ".cache" / "phase0"))

# ---------------------------------------------------------------- IO

def _get(url, dest=None):
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=120) as r:
        if dest is None:
            return r.read()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        tmp.rename(dest)

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n

class Model:
    """A model as a set of shards, fetched lazily (hf:) or read in place (local)."""
    def __init__(self, spec, stream=False):
        self.spec, self.stream, self.local = spec, stream, not spec.startswith("hf:")
        self.weight_map = None
        if self.local:
            p = Path(spec)
            files = sorted(p.glob("*.safetensors")) if p.is_dir() else [p]
            self.shards = [str(f) for f in files]
            self.weight_map = {n: s for s in self.shards for n in read_header(s)[0]}
        else:
            self.repo = spec[3:]
            info = json.loads(_get(f"{HF}/api/models/{self.repo}"))
            sib = [s["rfilename"] for s in info.get("siblings", [])]
            self.shards = sorted(f for f in sib if f.endswith(".safetensors"))
            if not self.shards:
                sys.exit(f"{self.repo}: no .safetensors files")
            if "model.safetensors.index.json" in sib:
                idx = json.loads(_get(f"{HF}/{self.repo}/resolve/main/model.safetensors.index.json"))
                self.weight_map = idx["weight_map"]
            elif len(self.shards) == 1:
                self.weight_map = None  # everything lives in shards[0]
        self._open = {}

    def shard_of(self, name):
        return self.shards[0] if self.weight_map is None else self.weight_map.get(name)

    def fetch(self, shard):
        if shard in self._open:
            return self._open[shard]
        if self.local:
            path = Path(shard)
        else:
            path = CACHE / self.repo.replace("/", "__") / shard
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                t = time.time()
                print(f"  downloading {self.repo}/{shard} ...", end="", flush=True)
                _get(f"{HF}/{self.repo}/resolve/main/{shard}", path)
                print(f" {path.stat().st_size / 1e9:.2f} GB in {time.time() - t:.0f}s", flush=True)
        header, base = read_header(path)
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        self._open[shard] = (path, header, base, mm)
        return self._open[shard]

    def tensor(self, name):
        shard = self.shard_of(name)
        if shard is None:
            return None, None, None
        path, header, base, mm = self.fetch(shard)
        meta = header.get(name)
        if meta is None:
            return None, None, None
        a, b = meta["data_offsets"]
        return meta["dtype"], meta["shape"], mm[base + a: base + b]

    def release(self, shard):
        ent = self._open.pop(shard, None)
        if ent and self.stream and not self.local:
            path = ent[0]
            del ent
            try:
                path.unlink()
            except FileNotFoundError:
                pass

# ---------------------------------------------------------------- entropy helpers

def H(counts):
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

def huffman_avg_len(counts):
    c = [int(x) for x in counts if x > 0]
    if len(c) == 1:
        return 1.0
    total, heap, k, cost = sum(c), [(x, i) for i, x in enumerate(c)], len(c), 0
    heapq.heapify(heap)
    while len(heap) > 1:
        a, _ = heapq.heappop(heap); b, _ = heapq.heappop(heap)
        cost += a + b
        heapq.heappush(heap, (a + b, k)); k += 1
    return cost / total

def _remap(c):
    """Map non-negative int context ids to 0..k-1 without sorting (bincount LUT)."""
    c = c.astype(np.int64)
    if c.size and c.max() < (1 << 26):
        used = np.bincount(c) > 0
        lut = np.cumsum(used) - 1
        return lut[c], int(used.sum())
    u, inv = np.unique(c, return_inverse=True)
    return inv, len(u)

def cond_xent(ctx_tr, sym_tr, ctx_te, sym_te, nsym):
    """Cross-entropy (bits/symbol) of sym_te under P(sym | ctx) fit on train.

    Uses PPM method-C escapes, like a real adaptive coder: a symbol seen in its
    context costs c/(n+q); an unseen one pays the escape q/(n+q) and is coded by
    the context-free model, which in turn escapes to uniform over unseen symbols.
    This avoids add-constant smoothing, whose penalty grows with alphabet size
    and would bias comparisons between alphabets of 256, 65536 and 131071.
    """
    n = len(ctx_tr)
    ids, k = _remap(np.concatenate([ctx_tr, ctx_te]))
    itr, ite = ids[:n], ids[n:]
    sym_tr = sym_tr.astype(np.int64); sym_te = sym_te.astype(np.int64)
    joint = np.bincount(itr * nsym + sym_tr, minlength=k * nsym)
    J = joint.reshape(k, nsym)
    n_ctx = J.sum(1).astype(np.float64); q_ctx = (J > 0).sum(1).astype(np.float64)
    cm = np.bincount(sym_tr, minlength=nsym).astype(np.float64)
    n_m, q_m = cm.sum(), float((cm > 0).sum())
    unseen = max(1.0, nsym - q_m)
    pm = np.where(cm > 0, cm / (n_m + q_m), (q_m / (n_m + q_m)) / unseen)
    c = joint[ite * nsym + sym_te].astype(np.float64)
    nc, qc = n_ctx[ite], q_ctx[ite]
    denom = np.maximum(nc + qc, 1.0)
    p = np.where(nc == 0, pm[sym_te], np.where(c > 0, c / denom, (qc / denom) * pm[sym_te]))
    return float(-np.log2(p).mean())

# ---------------------------------------------------------------- per-tensor analysis

def analyze_bf16(raw, shape, max_elems):
    u = raw.view(np.uint16)
    if len(shape) >= 2:
        cols = int(shape[-1]); rows = u.size // cols
        keep = max(2, min(rows, max_elems // cols)) & ~1
        m = np.asarray(u.reshape(rows, cols)[:keep])
    else:
        m = np.asarray(u[:max_elems]).reshape(1, -1)
    e = ((m >> 7) & 0xFF).astype(np.int64)
    m2 = ((m >> 5) & 0x3).astype(np.int64)
    ce = np.bincount(e.ravel(), minlength=256)
    r = {"n": int(e.size), "H_E": H(ce), "huff_E": huffman_avg_len(ce)}
    if m.shape[0] >= 2:
        tr, te = slice(0, None, 2), slice(1, None, 2)
    else:
        h = m.shape[1] // 2
        tr, te = (slice(None), slice(0, h)), (slice(None), slice(h, None))
    r["H_M2"] = H(np.bincount(m2.ravel(), minlength=4))
    r["H_M2_given_E"] = cond_xent(e[tr].ravel(), m2[tr].ravel(), e[te].ravel(), m2[te].ravel(), 4)
    z = lambda a: np.zeros(a.size, np.int64)
    r["H_E_split"] = cond_xent(z(e[tr]), e[tr].ravel(), z(e[te]), e[te].ravel(), 256)
    L = np.concatenate([np.zeros((e.shape[0], 1), np.int64), e[:, :-1]], 1)
    r["H_E_left"] = cond_xent(L[tr].ravel(), e[tr].ravel(), L[te].ravel(), e[te].ravel(), 256)
    if m.shape[0] >= 4:
        A, E, Lb = e[:-1], e[1:], L[1:]
        t2, s2 = slice(0, None, 2), slice(1, None, 2)
        r["H_E_above"] = cond_xent(A[t2].ravel(), E[t2].ravel(), A[s2].ravel(), E[s2].ravel(), 256)
        r["H_E_left_above"] = cond_xent(Lb[t2].ravel() * 256 + A[t2].ravel(), E[t2].ravel(),
                                        Lb[s2].ravel() * 256 + A[s2].ravel(), E[s2].ravel(), 256)
    return r

def lzma_ratio(raw, nbytes=8 << 20):
    s = bytes(np.asarray(raw[:nbytes]))
    return len(lzma.compress(s, preset=6)) / max(1, len(s))

def bf16_ord(u):
    u = u.astype(np.int32)
    return np.where(u & 0x8000, 0x7FFF - (u & 0x7FFF), u + 0x8000)  # monotone in float value

def analyze_delta(ft, base, max_elems):
    a = np.asarray(ft.view(np.uint16)[:max_elems]); b = np.asarray(base.view(np.uint16)[:max_elems])
    x = (a ^ b).astype(np.int64)
    d = (bf16_ord(a) - bf16_ord(b)).astype(np.int64) + 65535      # 0 .. 131070
    eb = ((b >> 7) & 0xFF).astype(np.int64)
    r = {"n": int(a.size), "frac_identical": float((x == 0).mean())}
    r["H_xor"] = H(np.bincount(x, minlength=65536))
    r["H_delta"] = H(np.bincount(d, minlength=131071))
    h = a.size // 2
    r["H_xor_split"] = cond_xent(np.zeros(h, np.int64), x[:h], np.zeros(a.size - h, np.int64), x[h:], 65536)
    r["H_delta_given_Ebase"] = cond_xent(eb[:h], d[:h], eb[h:], d[h:], 131071)
    r["H_xor_given_Ebase"] = cond_xent(eb[:h], x[:h], eb[h:], x[h:], 65536)
    return r

# ---------------------------------------------------------------- driver

def wavg(rows, key):
    rs = [(r[key], r["n"]) for r in rows if r.get(key) is not None]
    return sum(v * n for v, n in rs) / sum(n for _, n in rs) if rs else None

def summarize_single(rows, lz):
    s = {k: wavg(rows, k) for k in
         ["H_E", "huff_E", "H_E_split", "H_E_left", "H_E_above", "H_E_left_above", "H_M2", "H_M2_given_E"]}
    best = min(v for k, v in s.items() if k.startswith("H_E_") and v is not None)
    s["pct_dfloat11_like"] = 100 * (1 + s["huff_E"] + 7) / 16
    s["pct_floor_independent"] = 100 * (1 + s["H_E"] + 7) / 16
    s["pct_best_context"] = 100 * (1 + best + 5 + s["H_M2_given_E"]) / 16
    s["headroom_points"] = s["pct_dfloat11_like"] - s["pct_best_context"]
    s["lzma_pct_sample"] = 100 * float(np.mean(lz)) if lz else None
    return s

def summarize_delta(rows):
    s = {k: wavg(rows, k) for k in ["frac_identical", "H_xor", "H_xor_split", "H_xor_given_Ebase",
                                    "H_delta", "H_delta_given_Ebase"]}
    if s["H_xor_split"]:
        best_xor = min(s["H_xor_split"], s["H_xor_given_Ebase"])
        s["pct_xor"] = 100 * best_xor / 16
        s["pct_delta"] = 100 * s["H_delta_given_Ebase"] / 16
        s["delta_vs_xor_saving_pct"] = 100 * (1 - s["H_delta_given_Ebase"] / best_xor)
    return s

def run(spec, base_spec, max_elems, min_elems, stream, delta_only=False):
    ft = Model(spec, stream)
    base = Model(base_spec, stream) if base_spec else None
    single, delta, lz, t0 = [], [], [], time.time()
    shards = list(ft.shards)
    for i, shard in enumerate(shards):
        _, header, _, _ = ft.fetch(shard)
        for name, meta in header.items():
            dt, shape, raw = ft.tensor(name)
            if dt != "BF16" or raw.size // 2 < min_elems:
                continue
            if not delta_only:
                r = analyze_bf16(raw, shape, max_elems); r["name"] = name; single.append(r)
                if len(lz) < 8:
                    lz.append(lzma_ratio(raw))
            if base:
                bdt, _, braw = base.tensor(name)
                if bdt == "BF16" and braw.size == raw.size:
                    d = analyze_delta(raw, braw, max_elems); d["name"] = name; delta.append(d)
        ft.release(shard)
        if base:
            later = set()
            if ft.weight_map is not None:
                later = {base.shard_of(n) for n, s in ft.weight_map.items() if s in shards[i + 1:]}
            for bs in list(base._open):
                if bs not in later:
                    base.release(bs)
        print(f"  shard {i + 1}/{len(shards)} done, {len(single)} tensors, {time.time() - t0:.0f}s", flush=True)
    out = {"model": spec, "seconds": round(time.time() - t0, 1),
           "single": summarize_single(single, lz) if single else None, "per_tensor": single}
    if base:
        out.update(base=base_spec, delta=summarize_delta(delta), per_tensor_delta=delta,
                   delta_tensors=len(delta), delta_weights=sum(d["n"] for d in delta))
    return out

def fmt(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--base", help="base model for delta analysis (applies to every MODEL)")
    ap.add_argument("--stream", action="store_true", help="delete downloaded shards after use")
    ap.add_argument("--delta-only", action="store_true", help="skip single-model analysis")
    ap.add_argument("--max-elems", type=int, default=8_000_000, help="elements sampled per tensor")
    ap.add_argument("--min-elems", type=int, default=65_536, help="skip tensors smaller than this")
    ap.add_argument("--out", default="phase0_results.json")
    a = ap.parse_args()
    results = []
    for spec in a.models:
        print(f"[model] {spec}" + (f" vs base {a.base}" if a.base else ""), flush=True)
        r = run(spec, a.base, a.max_elems, a.min_elems, a.stream, a.delta_only); results.append(r)
        s = r["single"]
        if s: print(f"  H(E)={fmt(s['H_E'])} huff={fmt(s['huff_E'])} |left={fmt(s['H_E_left'])} "
              f"|above={fmt(s['H_E_above'])} |left,above={fmt(s['H_E_left_above'])} "
              f"H(M2)={fmt(s['H_M2'])} H(M2|E)={fmt(s['H_M2_given_E'])}")
        if s: print(f"  DFloat11-like {s['pct_dfloat11_like']:.2f}%  best-context {s['pct_best_context']:.2f}%  "
              f"headroom {s['headroom_points']:.2f} pts  lzma(sample) {fmt(s['lzma_pct_sample'], 1)}%")
        if "delta" in r:
            d = r["delta"]
            print(f"  identical={fmt(d['frac_identical'])} H(xor)={fmt(d['H_xor_split'])} "
                  f"H(xor|Eb)={fmt(d['H_xor_given_Ebase'])} H(delta)={fmt(d['H_delta'])} "
                  f"H(delta|Eb)={fmt(d['H_delta_given_Ebase'])}")
            print(f"  matched {r['delta_tensors']} tensors, {r['delta_weights'] / 1e9:.2f}B weights sampled")
            print(f"  XOR {fmt(d.get('pct_xor'), 2)}%  integer delta {fmt(d.get('pct_delta'), 2)}%  "
                  f"saving vs xor {fmt(d.get('delta_vs_xor_saving_pct'), 1)}%")
        Path(a.out).write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()

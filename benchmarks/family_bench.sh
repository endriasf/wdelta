#!/usr/bin/env bash
# Compare wdelta (with full decode + SHA-256 check) and BitX on fine-tunes of one base model.
# Processes one shard at a time so only the base plus one shard needs to fit on disk.
# usage: family_bench.sh BASE_REPO FT_REPO [FT_REPO ...]
set -euo pipefail
W="$(cd "$(dirname "$0")/.." && pwd)/target/release/wdelta"
ROOT="${BENCH_ROOT:-$HOME/models}"; HF=https://huggingface.co
shards() { curl -sSL --retry 5 --retry-all-errors "$HF/api/models/$1" | python3 -c "import json,sys;print('\n'.join(sorted(s['rfilename'] for s in json.load(sys.stdin)['siblings'] if s['rfilename'].endswith('.safetensors') and '/' not in s['rfilename'])))"; }
fetch() { [ -f "$2" ] || { curl -sSL --fail --retry 8 --retry-all-errors --retry-delay 5 -C - -o "$2.part" "$HF/$1/resolve/main/$3" && mv "$2.part" "$2"; }; }

BASE="$1"; shift
BDIR="$ROOT/${BASE//\//__}"; mkdir -p "$BDIR"
for s in $(shards "$BASE"); do fetch "$BASE" "$BDIR/$s" "$s"; done
echo "base $BASE ready ($(du -sh "$BDIR" | cut -f1))"
for FT in "$@"; do
  D="$ROOT/ft"; mkdir -p "$D"; tin=0; tw=0; tb=0; ok=0; n=0
  for s in $(shards "$FT"); do
    fetch "$FT" "$D/$s" "$s"
    "$W" encode "$BDIR" "$D/$s" "$D/$s.wdz" > "$D/enc.log"
    if "$W" decode "$BDIR" "$D/$s.wdz" | grep -q MATCH; then ok=$((ok+1)); fi
    bx=$("$W" bitx "$BDIR" "$D/$s" 3 | sed -E 's/.*-> ([0-9.]+) GB.*/\1/')
    tin=$((tin + $(stat -c %s "$D/$s"))); tw=$((tw + $(stat -c %s "$D/$s.wdz")))
    tb=$(python3 -c "print($tb + int($bx*1e9))"); n=$((n+1))
    sed -n 3p "$D/enc.log" | sed "s|^|  $s:|"
    rm -f "$D/$s" "$D/$s.wdz"
  done
  python3 -c "i,w,b=$tin,$tw,$tb; print(f'RESULT {\"$FT\"}: {i/1e9:.2f} GB | wdelta {w/1e9:.2f} GB = {100*w/i:.2f}% | BitX {b/1e9:.2f} GB = {100*b/i:.2f}% | wdelta {100*(1-w/b):.1f}% smaller | verified $ok/$n shards')"
done

#!/usr/bin/env bash
# Benchmark wdelta on fine-tunes of a base model, one shard at a time.
# usage: bench.sh BASE_REPO FT_REPO [FT_REPO ...]
set -euo pipefail
W="$(dirname "$0")/../target/release/wdelta"
ROOT="${BENCH_ROOT:-$HOME/models}"
HF="https://huggingface.co"
shards() { curl -sSL "$HF/api/models/$1" | python3 -c "import json,sys;print('\n'.join(sorted(s['rfilename'] for s in json.load(sys.stdin)['siblings'] if s['rfilename'].endswith('.safetensors') and '/' not in s['rfilename'])))"; }
fetch() { mkdir -p "$(dirname "$2")"; [ -f "$2" ] || curl -sSL --fail -o "$2.part" "$HF/$1/resolve/main/$3" && mv -f "$2.part" "$2" 2>/dev/null || true; }

BASE="$1"; shift
BDIR="$ROOT/${BASE//\//__}"
for s in $(shards "$BASE"); do fetch "$BASE" "$BDIR/$s" "$s"; done
echo "base ready: $BDIR ($(du -sh "$BDIR" | cut -f1))"

for FT in "$@"; do
  echo "=== $FT"
  FDIR="$ROOT/${FT//\//__}"; ODIR="$ROOT/out/${FT//\//__}"; mkdir -p "$FDIR" "$ODIR"
  tin=0; tout=0
  for s in $(shards "$FT"); do
    fetch "$FT" "$FDIR/$s" "$s"
    "$W" encode "$BDIR" "$FDIR/$s" "$ODIR/$s.wdz" | sed 's/^/  /'
    "$W" decode "$BDIR" "$ODIR/$s.wdz" | sed 's/^/  /'
    tin=$((tin + $(stat -c %s "$FDIR/$s"))); tout=$((tout + $(stat -c %s "$ODIR/$s.wdz")))
    rm -f "$FDIR/$s"
  done
  python3 -c "print(f'TOTAL $FT: {$tin/1e9:.2f} GB -> {$tout/1e9:.2f} GB = {100*$tout/$tin:.2f}% of original')"
done

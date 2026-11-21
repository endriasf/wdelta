#!/usr/bin/env bash
# Run the BitX reimplementation on each shard of the given fine-tunes vs Qwen3-8B.
set -euo pipefail
W="$(cd "$(dirname "$0")/.." && pwd)/target/release/wdelta"
HF=https://huggingface.co; ROOT=$HOME/models; BDIR=$ROOT/Qwen__Qwen3-8B
for FT in "$@"; do
  D=$ROOT/${FT//\//__}; mkdir -p "$D"
  for s in $(curl -sSL "$HF/api/models/$FT" | python3 -c "import json,sys;print('\n'.join(sorted(s['rfilename'] for s in json.load(sys.stdin)['siblings'] if s['rfilename'].endswith('.safetensors') and '/' not in s['rfilename'])))"); do
    curl -sSL --fail -o "$D/$s" "$HF/$FT/resolve/main/$s"
    "$W" bitx "$BDIR" "$D/$s" 3
    rm -f "$D/$s"
  done
done

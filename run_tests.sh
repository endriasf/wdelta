#!/usr/bin/env bash
# Run every wdelta test. Works on Linux, macOS and Windows (Git Bash).
# Usage: ./run_tests.sh   (WDELTA_NET_TESTS=1 adds the Hugging Face round trip)
#   1. Rust unit and file tests (every bf16 value, every delta, specials, damaged files, wrong base)
#   2. Golden cross-platform check (output must be byte-identical to the file made on Linux)
#   3. Python package tests (pack/unpack/from_pretrained/CLI on local models)
#   4. Decoder fuzz test (truncated and corrupted files, each decoded in its own process)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python3 || command -v python)}"

echo "== 1/4 Rust tests"
cargo test --release --quiet 2>&1 | grep -E "test result|FAILED|panicked"
cargo build --release --quiet
if [ -f target/release/wdelta.exe ]; then BIN="$PWD/target/release/wdelta.exe"; else BIN="$PWD/target/release/wdelta"; fi
export WDELTA_BIN="$BIN"

echo "== 2/4 Golden cross-platform check"
"$PY" tests/golden_check.py "$WDELTA_BIN"

echo "== 3/4 Python package tests"
"$PY" -m pytest -q python/tests/test_package.py

echo "== 4/4 Decoder fuzz test"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
"$PY" tests/make_golden.py "$T"
"$WDELTA_BIN" encode "$T/base.safetensors" "$T/ft.safetensors" "$T/ft.wdz" > /dev/null
"$PY" tests/fuzz_decode.py "$WDELTA_BIN" "$T/base.safetensors" "$T/ft.safetensors" "$T/ft.wdz" "${FUZZ_N:-800}"
echo "All tests passed."

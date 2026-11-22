# wdelta

[![tests](https://github.com/endriasf/wdelta/actions/workflows/tests.yml/badge.svg)](https://github.com/endriasf/wdelta/actions/workflows/tests.yml)

Lossless compression for fine-tuned models. A fine-tune is stored as a delta against its
base model, so it downloads and stores at a fraction of its size, and is restored
byte-for-byte.

> **Status: experimental (v0.3).** The file format may still change before 1.0. Keep your
> original files until you have verified a restore, and don't use it as your only copy of
> anything you need to keep long term.

## How well it works

Measured on real fine-tunes, compressed against the base model named on each model card.
Sizes are % of the original bf16 file; every wdelta output was restored and checked
byte-for-byte (SHA-256).

| Fine-tune | Base | BitX | FM-Delta | wdelta |
| --- | --- | --- | --- | --- |
| Llama-3.1-Nemotron-70B-Instruct | Llama-3.1-70B-Instruct | 17.66% | not run | **7.38%** (141 GB → 10.4 GB) |
| Llama-3.1-8B-Instruct-MetaSecAlign | Llama-3.1-8B-Instruct | 12.68% | 7.66% | **5.34%** |
| Llama-3.1-SuperNova-Lite | Llama-3.1-8B-Instruct | 22.33% | 16.26% | **13.96%** |
| KernelLLM | Llama-3.1-8B-Instruct | 47.60% | 39.11% | **35.40%** |
| Huihui-Qwen3-8B-abliterated-v2 | Qwen3-8B | 10.62% | not run | **7.87%** |
| WebExplorer-8B | Qwen3-8B | 40.71% | not run | **29.06%** |
| Goedel-Prover-V2-8B | Qwen3-8B | 50.99% | not run | **40.23%** |
| OpenHermes-2.5-Mistral-7B | Mistral-7B-v0.1 | 46.18% | not run | **34.05%** |
| zephyr-7b-beta | Mistral-7B-v0.1 | 49.82% | not run | **37.75%** |
| Mistral-7B-Instruct-v0.1 | Mistral-7B-v0.1 | 62.37% | not run | **53.44%** |

- **BitX** ([ZipLLM](https://github.com/ds2-lab/ZipLLM), NSDI 2026) at its default zstd
  level 3. ZipLLM's own code was run on the three Llama-3.1-8B fine-tunes and matched our
  reimplementation (used for the other rows) or gave slightly larger output. At zstd level 19
  on MetaSecAlign, BitX reaches 10.11%.
- **FM-Delta** ([NeurIPS 2024](https://github.com/ningwanyi/FM-Delta)), run from its
  published C++ coder. It has no bf16 type; bf16 weights were passed as its float16 type,
  which gives identical integer deltas for its sign-magnitude mapping.
- **Without a base model**, wdelta falls back to exponent coding and reaches about 66.5% on
  Qwen3-8B, the same as single-model tools such as ZipNN and DFloat11. The gains above come
  entirely from having the base.

All runs used a 2-vCPU machine; decode ran at about 220–240 MB/s on a typical 8B fine-tune.
The scripts in `benchmarks/` reproduce every number.

## Prior work

wdelta does not introduce a new idea. It implements a known approach and measures it at
scale on today's bf16 models:

- **fpzip** (Lindstrom and Isenburg, 2006) maps floats to integers and codes residuals.
- **[FM-Delta](https://github.com/ningwanyi/FM-Delta)** (Ning et al., NeurIPS 2024) applies
  that mapping to fine-tuned vs pre-trained weights.
- **[ULPack](https://arxiv.org/abs/2610.14184)** (Ashok, NeurIPS 2026) codes order-preserving
  integer deltas with interleaved rANS conditioned on the reference weight's exponent.
  wdelta's coder was developed independently and closely matches that design. We found no
  public ULPack code, so it is not compared above.
- **[ZipLLM / BitX](https://github.com/ds2-lab/ZipLLM)** (Wang et al., NSDI 2026) codes the
  XOR of the two weights with zstd, and showed how much hub storage fine-tunes take.

What wdelta adds: a fast, multithreaded implementation with bf16 support, a container that
pins the exact base (commit and per-tensor hashes), automatic base download, and direct use
with Hugging Face transformers.

## Install

**Prebuilt binaries:** download the archive for your system from the
[latest release](https://github.com/endriasf/wdelta/releases/latest) (macOS Apple Silicon and
Intel, Windows x64, Linux x64). It contains the `wdelta` command-line codec.

**Python package (whole-model pack/unpack), from source:**

```bash
cargo build --release
mkdir -p python/wdelta/bin && cp target/release/wdelta python/wdelta/bin/
pip install ./python[hub]
```

On Windows, copy `target\release\wdelta.exe` instead. Without the `[hub]` extra, base
models are downloaded over plain HTTPS. Requires Rust 1.85 or newer to build.

## Use

```bash
# compress a fine-tune against its base (local folders or Hugging Face repo ids)
wdelta-model pack my-org/my-finetune --base meta-llama/Llama-3.1-8B-Instruct -o packed/

# restore it; the base is downloaded automatically at the exact commit recorded at pack time
wdelta-model unpack packed/ -o restored/
```

```python
import wdelta
from transformers import AutoModelForCausalLM

path = wdelta.from_pretrained("packed/")      # local folder or a Hub repo holding a packed model
model = AutoModelForCausalLM.from_pretrained(path)
```

The low-level codec works on single safetensors files:

```bash
wdelta encode BASE_DIR model-00001-of-00004.safetensors out.wdz   # base "none" for no base
wdelta decode BASE_DIR out.wdz restored.safetensors               # omit output to verify only
wdelta info out.wdz                                               # which base a file needs
```

`from_pretrained` unpacks once into `~/.cache/wdelta` (override with `WDELTA_CACHE`) and
returns the folder on later calls. Every restored shard is checked against the SHA-256
recorded at pack time, and every base tensor against a hash stored in the packed file, so a
wrong or modified base is rejected instead of producing a different model. Gated base models
(for example Meta's Llama) need `HF_TOKEN` set.

## What a packed model contains

- `*.safetensors.wdz`: one compressed file per weight shard
- every non-weight file of the original (config, tokenizer, generation config, ...)
- `wdelta.json`: the base model's repo id and commit, and the size and SHA-256 of each shard

## Limitations

- Only bf16 tensors are compressed. fp16 and fp32 tensors are currently stored uncompressed.
- Restoring a fine-tune needs its base. If the base is deleted from the Hub, it cannot be fetched.
- This is a storage and transfer format. Models are unpacked before loading, so it does not
  reduce GPU memory during inference.
- Tested on dense transformer LLMs (Qwen3, Mistral, Llama 3.1, 7B–70B). Mixture-of-experts,
  vision and speech models are untested.

## Testing

```bash
./run_tests.sh                      # Rust tests, golden cross-platform check, Python tests, fuzz test
WDELTA_NET_TESTS=1 ./run_tests.sh   # also packs and restores Qwen3-0.6B from the Hub (~3 GB)
```

The tests check every one of the 65,536 bf16 values as both base and target, every possible
delta, special values (±0, ±inf, NaN, subnormals), every record type, a wrong or incomplete
base, and thousands of truncated or corrupted files, which must fail with a clear error and
leave no output behind. A golden check requires every platform to produce byte-identical
output to the reference file made on Linux. CI runs all of it on macOS (Apple Silicon and
Intel), Windows and Linux on every push.

## License

Apache License 2.0. See [LICENSE](LICENSE).

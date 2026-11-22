# wdelta

[![tests](https://github.com/endriasf/wdelta/actions/workflows/tests.yml/badge.svg)](https://github.com/endriasf/wdelta/actions/workflows/tests.yml)

Lossless compression for fine-tuned models. A fine-tune is stored as a delta against its
base model, so it downloads and stores at a fraction of its size, and is restored
byte-for-byte.

> **Status: experimental (v0.3).** The file format may still change before 1.0. Keep your
> original files until you have verified a restore, and don't use it as your only copy of
> anything you need to keep long term.

## What it can and cannot do

**It can:**

- Shrink a fine-tuned model to 5–54% of its size (bf16 safetensors, tested on 10 fine-tunes
  from 4 base models), when the base model it was fine-tuned from is available.
- Restore the fine-tune bit-for-bit. Nothing is approximated: every restored shard is checked
  against the SHA-256 of the original.
- Compress bf16 weights without a base model, to about 66.5%. That matches single-model tools
  such as ZipNN and DFloat11; it does not beat them.
- Pin the exact base: the packed model records the base's Hub repo and commit, plus a hash of
  every base tensor it uses, so a wrong or modified base is rejected instead of producing a
  different model.
- Pack and restore whole Hugging Face models (weights, config, tokenizer), download the base
  automatically, and hand the restored folder to `transformers`.

**It cannot:**

- **Help much without the right base.** Savings come from the base model. A model trained from
  scratch, or compared against the wrong base, gets the ~66.5% base-less result at best.
- **Promise a fixed ratio.** The result depends on how far the fine-tune moved from its base:
  5.3% for a light security fine-tune (MetaSecAlign), 53% for
  Mistral-7B-Instruct-v0.1. The more the training changed the weights, the less there is to gain.
- **Make downloads smaller if you don't already have the base.** Restoring needs the base; if
  it isn't on your machine it is downloaded too. The saving is largest when many fine-tunes
  share one base you already have.
- **Compress anything other than bf16.** fp16, fp32, int8 and other quantized tensors are stored
  uncompressed. GGUF and other non-safetensors formats are not supported.
- **Reduce memory or speed up inference.** It is a storage and transfer format, like zip.
  Models are fully restored before loading. It is not quantization.
- **Restore a model whose base has disappeared.** If the base is deleted from the Hub and you
  have no copy, the fine-tune can't be restored.
- **Prove where a file came from.** The hashes detect corruption and a wrong base. They are not
  signatures: anyone can build a valid `.wdz` of a different model. Trust a packed model only as
  much as you trust its source.
- **Vouch for untested model types.** Testing covered dense transformer LLMs (Qwen3, Mistral,
  Llama 3.1, 7B–70B). Mixture-of-experts, vision, speech and diffusion models are untested.
- **Match zstd's speed.** Decoding runs at about 220–240 MB/s on 2 CPU cores, well below
  general-purpose decompressors such as zstd, which typically decode at over 1 GB/s.

Whole-model packing (`wdelta-model pack`) compresses every `*.safetensors` file in the model
folder and its subfolders, and copies every other file (`.bin`, `.pt`, `.onnx`, config,
tokenizer, ...) **without compression**, so a repo that also ships `pytorch_model.bin` copies of
its weights will pack to a much larger size. The command lists any large uncompressed files.
Hidden files and folders (names starting with `.`, such as `.gitattributes`) are not packed.

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

## What makes it different

All delta compressors for fine-tunes rely on the same fact: fine-tuning moves most weights only
a little, so a fine-tune is mostly predictable from its base. They differ in how they measure
"a little" and how they encode it.

| | Difference it encodes | How it encodes it |
| --- | --- | --- |
| BitX / ZipLLM | bitwise XOR of base and fine-tune | zstd (a general-purpose byte compressor) on two byte streams |
| FM-Delta | integer difference (fpzip mapping) | range coder on the difference's bit length, low bits stored raw |
| wdelta | integer difference, in steps between adjacent bf16 values | rANS entropy coder, statistics conditioned on the base weight's exponent |

wdelta's gains come from three choices:

1. **It measures change as a number, not as a bit pattern.** XOR is small only when the two
   values share most of their bits. A weight that moves from 1.0 to 0.99609375, the very next
   bf16 value below it, flips 8 of its 16 bits under XOR (`0x3F80 ^ 0x3F7F = 0x00FF`) because it
   crosses a power of two. As an integer step it is simply −1. Fine-tuning moves millions of
   weights across exponent boundaries, and XOR pays for every one of them.
2. **It uses a coder built for this data.** zstd looks for repeated byte sequences, which
   weight deltas rarely contain. wdelta instead counts how often each delta occurs and codes
   each one with close to the minimum number of bits those counts allow (entropy coding with
   rANS).
3. **It conditions on the base weight.** The same numerical change is a different number of
   steps depending on the weight's size: large weights have coarse steps, small weights fine
   ones. wdelta keeps separate statistics for each base exponent, so deltas of small weights
   and of large weights are each coded with their own probabilities. FM-Delta does not condition
   on the base weight, and stores the low bits of each delta uncoded.

How much each choice contributes varies by model; we have not published a per-step breakdown.
The total effect is in the table above: 14–58% smaller than BitX (median 26%) and 9.5–30%
smaller than FM-Delta on the models where both ran.

## How it works

A bf16 number is 16 bits: 1 sign bit, 8 exponent bits and 7 mantissa bits.

**1. Map each weight to an ordered integer.** Each 16-bit pattern `u` is mapped to an integer
`ord(u)` (0–65535) such that a larger float always gets a larger integer, and neighbouring
bf16 values get neighbouring integers, including across zero and across powers of two. The
mapping is one-to-one, so it is exactly reversible.

**2. Take the difference.** For each weight, `d = ord(fine-tune) − ord(base)`: the number of
bf16 steps the weight moved. Unchanged weights give 0; most changed weights give a small `d`.

**3. Turn differences into symbols.** Each `d` becomes one of 55 symbols:

- −15 to +15 each get their own symbol (31 symbols). These cover most weights.
- Larger values get a symbol for their sign and bit length (24 symbols), followed by their
  remaining low bits written as-is. Those bits are close to random, so modelling them gains
  almost nothing.

**4. Entropy-code the symbols, per base exponent.** Weights are processed in chunks of
4,194,304. For each chunk, wdelta counts how often each symbol occurs for each base exponent
and stores those counts as small tables (12-bit precision, only non-zero entries). It then
codes every symbol with the table for its base weight's exponent, using rANS, an entropy coder
that spends close to the theoretical minimum number of bits for those counts. Four rANS states
are interleaved so the CPU can work on several symbols at once.

Decoding reverses the steps. The decoder already has the base, so it knows which table to use
for each weight before reading its symbol. Chunks are independent, so they are encoded and
decoded in parallel on all cores.

**Other record types.** Each tensor in a shard is stored one of four ways, whichever is
smallest:

- **Reference:** identical to the base tensor. Only its name and the base tensor's hash are
  stored.
- **Delta:** the coding above, for bf16 tensors whose base has the same name and size.
- **Exponent:** for bf16 tensors with no usable base (new tensors, a resized vocabulary, or no
  base at all). Exponents are entropy-coded (they carry most of the redundancy in trained
  weights); the sign and mantissa bits are stored as-is. This gives the ~66.5% base-less
  result.
- **Raw:** everything else (non-bf16 tensors, small tensors, the file header).

**The file.** A `.wdz` file holds the original safetensors header, one record per tensor, a
128-bit xxh3 hash of each base tensor it depends on, and the SHA-256 of the original file. On
decode, every base tensor is hashed and compared before use, and the restored file is checked
against the SHA-256; any mismatch is an error and no output is left behind.

**Measuring it honestly.** During development, gains were estimated as cross-entropies (model
fitted on one part of a tensor, scored on another) so larger context tables could not look good
by overfitting. The final numbers in this README are actual file sizes, every one restored and
checked. `research/entropy_study.py` is the analysis script used for those estimates.

## Prior work

wdelta does not introduce a new idea. It implements a known approach and measures it at
scale on today's bf16 models:

- **fpzip** (Lindstrom and Isenburg, 2006) maps floats to integers and codes residuals.
- **[FM-Delta](https://github.com/ningwanyi/FM-Delta)** (Ning et al., NeurIPS 2024) applies
  that mapping to fine-tuned vs pre-trained weights.
- **[ULPack](https://arxiv.org/abs/2610.14184)** (Ashok, NeurIPS 2026) codes order-preserving
  integer deltas with interleaved rANS conditioned on the reference weight's exponent.
  wdelta's coder was developed independently and closely matches that design, so the
  compression advantages above should be expected of ULPack too. We found no public ULPack
  code, so it is not compared.
- **[ZipLLM / BitX](https://github.com/ds2-lab/ZipLLM)** (Wang et al., NSDI 2026) codes the
  XOR of the two weights with zstd, and showed how much hub storage fine-tunes take.

What wdelta adds is engineering rather than a new method: a fast, multithreaded, open
implementation with bf16 support, a container that pins the exact base (commit and per-tensor
hashes), automatic base download, direct use with Hugging Face transformers, and prebuilt
binaries for macOS, Windows and Linux.

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

- `*.safetensors.wdz`: one compressed file per safetensors file, at the same path. A file in a
  subfolder (for example `unet/`) is coded against the same subfolder of the base when the
  base has one, otherwise against the base's top-level weights.
- every other file of the original, copied as-is (config, tokenizer, `.bin` files, ...)
- `wdelta.json`: the base model's repo id and commit, and the size and SHA-256 of every file.
  Unpacking checks all of them.

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

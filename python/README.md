# wdelta

Lossless compression for fine-tuned models. A fine-tune is stored as a delta against its
base model, so it downloads and stores at a fraction of its size and is restored
byte-for-byte.

Measured on 10 fine-tunes from Qwen3, Mistral and Llama 3.1 (7B to 70B): 5–53% of the
original size, and 14–58% smaller than BitX (ZipLLM, NSDI 2026). Llama-3.1-Nemotron-70B
goes from 141 GB to 10.4 GB.

## Install

```bash
cargo build --release                          # in the repo root: builds the codec
mkdir -p python/wdelta/bin && cp target/release/wdelta python/wdelta/bin/
pip install ./python[hub]
```

On Windows, copy `target\release\wdelta.exe` instead. Without the `[hub]` extra, base
models are downloaded over plain HTTPS. Tested on Linux; macOS and Windows are untested.

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

`from_pretrained` unpacks once into `~/.cache/wdelta` (override with `WDELTA_CACHE`) and
returns the folder on later calls. Every restored shard is checked against the SHA-256
recorded at pack time, and every base tensor against a hash stored in the packed file, so a
wrong or modified base model is rejected instead of producing a different model.

## What a packed model contains

- `*.safetensors.wdz`: one compressed file per weight shard
- every non-weight file of the original (config, tokenizer, generation config, ...)
- `wdelta.json`: the base model's repo id and commit, and the size and SHA-256 of each shard

Gated base models (for example Meta's Llama) need `HF_TOKEN` set.

## Testing

```bash
./run_tests.sh                      # Rust tests, Python package tests, decoder fuzz test
WDELTA_NET_TESTS=1 ./run_tests.sh   # also packs and restores Qwen3-0.6B from the Hub (~3 GB)
```

The tests check every one of the 65,536 bf16 values as both base and target, every possible
delta, special values (±0, ±inf, NaN, subnormals), every record type, a wrong or incomplete
base, and thousands of truncated or corrupted files, which must fail with a clear error and
leave no output behind.

---
library_name: vllm
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model:
- xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text
base_model_relation: quantized
tags:
- gemma4
- compressed-tensors
- w4a16
- int4
- qat
- vllm
- text-only
---

# Gemma 4 E2B-it QAT, W4A16 with int4 embeddings, text only (unofficial repack)

**This is an unofficial repack, made and published independently of Google.** It is
[`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text)
with its three embedding tables packed as int4 too, so every large tensor in the
model is 4-bit. Google's model card is kept unchanged in `ORIGINAL_README.md`.

On one Tesla T4 with vLLM 0.29.0 it loads in **2.86 GiB**, against 6.33 GiB for the
`-text` build, and decodes **34% faster**, with the same greedy output on every
prompt tried.

## Why this is not a new quantization

Google's quantization-aware training (QAT) put the embedding tables on the same
4-bit grid as the linear layers: within each group of 32 values along a row, every
value is `step × level` with level from −8 to 7. Measured on
`google/gemma-4-E2B-it-qat-q4_0-unquantized`: **0 of 73,400,320** groups of the
per-layer-embedding table and 0 of `embed_tokens`' groups are off that grid, while
the same test on the bf16 `google/gemma-4-E2B-it` finds no group on it. This repack
recovers each group's step and stores the levels; it does not choose new ones.

## What changed from the `-text` build

| Tensor | `-text` | This repack |
|---|---:|---:|
| `embed_tokens_per_layer` (PLE, 262144 × 8960) | 4.375 GiB bf16 | 1.230 GiB int4 |
| `embed_tokens` (262144 × 1536) | 0.750 GiB bf16 | 0.211 GiB int4 |
| `lm_head` | tied to `embed_tokens` | 0.211 GiB int4, untied copy |
| Checkpoint | 6.11 GiB | **2.64 GiB** |

- **Format:** compressed-tensors `pack-quantized`, symmetric int4, group 32, **fp16
  scales**, the same layout as the linear layers. `config.json` adds one config group
  per table (`re:.*\.embed_tokens_per_layer$`, `re:.*\.embed_tokens$`, `re:^lm_head$`).
- **Untied `lm_head`.** vLLM ties the output layer by copying the embedding's
  `.weight`, which a packed embedding does not have, so `tie_word_embeddings` is
  `false` and `lm_head` is written as a second copy of the same levels and scales.
  The model was trained tied, so both copies hold the trained values. vLLM runs
  `lm_head` as an int4 Marlin linear, which is where the decode speedup comes from.
- **Fidelity.** 74.3% of PLE values and 74.0% of `embed_tokens` values reconstruct
  bit-identically. The rest differ by at most 1.25 bf16 ulps of the source value
  (99.9% within one): the source values are themselves bf16 roundings of
  `step × level`, so no single stored step reproduces every one.
- **Unchanged:** every other tensor, byte for byte.

Made with [`embed_int4.py`](https://huggingface.co/xbill9/gpu-vllm-t4-2b-w4a16/blob/main/repack/embed_int4.py)
(`--embed-tokens`, default fp16 scales).

## Measured on one Tesla T4

vLLM 0.29.0, `--dtype float16 --gpu-memory-utilization 0.90 --max-model-len 16384
--max-num-seqs 8`, compile cache warm:

| | `-text` | This repack |
|---|---:|---:|
| Model loading | 6.33 GiB | **2.86 GiB** |
| KV cache | 711,539 tokens | **1,099,362 tokens** |
| Decode, one stream, 256 tokens | 81.6 tok/s | **109.7 tok/s** |
| Greedy output, 8 prompts × 160 tokens | reference | 8 of 8 token-identical |

The 8-prompt check is a spot check, not a benchmark suite. Evidence:
[`evidence/`](https://huggingface.co/xbill9/gpu-vllm-t4-2b-w4a16/tree/main/evidence) in the serving rig.

## Serving it

**The serving rig is public:**
[`xbill9/gpu-vllm-t4-2b-w4a16`](https://huggingface.co/xbill9/gpu-vllm-t4-2b-w4a16)
holds the tools that built this checkpoint (`repack/`), the vLLM launcher and config
used for every number on this card (`vllm-t4`, `tpu.env`), the Turing patch, and the
evidence. Maintained copy:
[gemma4-dev/gpu-vllm-t4-2b-w4a16](https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-t4-2b-w4a16).

Needs **vLLM 0.29 or later**, which added compressed-tensors int4 embeddings
(`CompressedTensorsEmbeddingWNA16Int`). Older vLLM rejects the config.

```bash
vllm serve xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4 --dtype float16 --max-model-len 16384
```

**Restart once after the first start.** The first start compiles from scratch and
vLLM counts the compiler's memory as activation when it sizes the KV cache (980,210
tokens on that start). vLLM keys that cache on the model identifier, so this
recurs whenever the model name or path changes. On Turing, vLLM's Triton attention
also needs a shared-memory clamp; see
`patch_triton_turing.py` in the serving rig.

## Limitations

- Text only: image and audio inputs are not supported.
- Tested on one T4 (fp16) with vLLM 0.29.0 only. On a bf16 GPU, vLLM casts the fp16
  scales to bf16, which is no worse than storing bf16.
- Other runtimes may not load int4 embeddings.
- Unofficial. Report problems here, not to Google.

## License and attribution

Gemma 4 is released by Google DeepMind under the
[Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This
repository redistributes Google's weights in a changed container format, under the
same license. The weights, training and the original model card are Google
DeepMind's; the repack and this card are not affiliated with or endorsed by Google.

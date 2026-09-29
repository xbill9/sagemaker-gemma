---
library_name: vllm
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model:
- xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct
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

# Gemma 4 E2B-it QAT, compressed-tensors W4A16, text only (unofficial repack)

**This is an unofficial repack, made and published independently of Google.** It is
[`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct)
with the vision and audio towers removed, loadable as a text-only `Gemma4ForCausalLM`.
Everything that remains is byte-identical to that repack; see its card for how the
QAT weights were recovered, verified and measured. Google's model card is kept
unchanged in `ORIGINAL_README.md`.

## What changed

- **Dropped:** every tensor under `model.vision_tower.`, `model.audio_tower.`,
  `model.embed_vision.` and `model.embed_audio.` (1,411 tensors, 0.88 GiB).
- **Kept, byte for byte:** all 1,092 language-model tensors, names unchanged
  (vLLM's `Gemma4ForCausalLM` maps `model.language_model.*` itself). Checked
  tensor by tensor against the source: none missing, none extra, none different.
- **Config:** the source `text_config` (`model_type: gemma4_text`) with
  `architectures: ["Gemma4ForCausalLM"]`, the source `quantization_config`, and
  the tower modules removed from its `ignore` list (only `lm_head` remains).
  `processor_config.json` is not included.
- **Size:** 6.11 GiB, against 7.00 GiB for the multimodal repack.

Made with [`text_only.py`](https://github.com/xbill9/gemma4-dev/blob/main/jev-tpu-31b/text_only.py).

## Serving it

No `--language-model-only` or `--limit-mm-per-prompt` flag is needed; there is no
multimodal path to disable.

**NVIDIA T4 (tested).** vLLM 0.29.0, one Tesla T4 (SM 7.5), float16 because
Turing has no bf16:

```bash
vllm serve xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text --dtype float16 --max-model-len 16384
```

vLLM resolves `Gemma4ForCausalLM`, selects `MarlinLinearKernel` for the W4A16
linears, and reports 6.33 GiB for model loading. With `--gpu-memory-utilization 0.90`
and `--max-num-seqs 8` the KV cache holds **711,539 tokens** (43x a full 16,384-token
request), slightly more than the multimodal repack under `--language-model-only`
(707,617). On Turing, vLLM's Triton attention needs a shared-memory clamp; see
[gpu-vllm-t4-2b](https://github.com/xbill9/gemma4-dev/tree/main/gpu-vllm-t4-2b).

**Restart once after the first start.** The first start compiles from scratch
(about 2 minutes on 2 vCPU), and vLLM counts the compiler's memory as activation
when it sizes the KV cache: that start measured 4.67 GiB of peak activation and
only 224,728 KV tokens. The next start loads the compile cache and gets the
figures above. vLLM keys that cache on the model identifier, so this recurs
whenever the model name or path changes.
[Evidence](https://github.com/xbill9/gemma4-dev/blob/main/gpu-vllm-t4-2b/evidence/2026-09-29-cold-compile-kv.txt).

## Limitations

- Text only: image and audio inputs are not supported.
- Tested on one T4 with a handful of prompts; not benchmarked separately from the
  multimodal repack, whose language model it shares exactly.
- Unofficial. Report problems here, not to Google.

## License and attribution

Gemma 4 is released by Google DeepMind under the
[Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license). This
repository redistributes Google's weights in a changed container format, under the
same license. The weights, training and the original model card are Google
DeepMind's; the repack and this card are not affiliated with or endorsed by Google.

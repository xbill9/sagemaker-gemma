# 2026-09-28 — Gemma 4 12B W4A16 repack against Google's 12B QAT, L4 and L40S

[`xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct)
is Google's `-qat-q4_0-unquantized` 12B written out as compressed-tensors W4A16
(int4, group 32, bf16 scales), the method of the 26B A4B repack. Google also
publishes its own `-qat-w4a16-ct` for this size, so the two can be set side by
side. Each ran on the instance Google's build was measured on the same day:
`ml.g6.xlarge` (1× L4, `../2026-09-28-12b-qat-l4/`) and `ml.g6e.xlarge`
(1× L40S, `../2026-09-28-12b-qat-vs-bf16/`). Same container digest
(vLLM 0.30.0), standard settings (`max_model_len` 8192, GPU memory
utilization 0.9, temperature 0), measurement script and region (us-east-2).
The checkpoint served is Hugging Face revision `2a96908`.

## Results (`compare-l4.json`, `compare-l40s.json`; ratios by `compare.py combine`)

| Measure | L4 Google | L4 repack | L40S Google | L40S repack |
| --- | ---: | ---: | ---: | ---: |
| Weights in GPU memory (GiB) | 8.28 | 8.28 | 8.28 | 8.28 |
| KV cache (tokens) | 41,651 | 41,651 | 126,058 | 126,058 |
| Decode, one request (tokens/s) | 29.3 | 29.2 | 77.0 | 78.3 |
| 1 request, 256 tokens (tokens/s) | 27.0 | 27.3 | 66.2 | 66.9 |
| 4 parallel (tokens/s) | 105.35 | 104.7 | 253.55 | 251.25 |
| 16 parallel (tokens/s) | 358.05 | 357.7 | 853.4 | 860.8 |
| Questions correct (of 40) | 39 | 40 | 39 | 40 |
| $ per million tokens, 16 parallel | 0.874 | 0.875 | 0.848 | 0.841 |

- Repack / Google ratios on every rate: 0.99–1.01 on the L4, 0.99–1.02 on the
  L40S. Memory and KV cache are identical.
- 512-token calls: L4 repack 18.216–18.388 s (Google 18.168–18.277), L40S
  repack 7.126–7.225 s (Google 7.132–7.366).
- 16-parallel batches: L4 repack 356.5 and 358.9, L40S repack 860.2 and
  861.4 tokens/s.
- Dollars per million tokens: $1.1267/h (L4) and $2.6054/h (L40S) on-demand
  hosting, us-east-2, divided by the 16-parallel rate.

## Quality

The same 40 seeded questions (15 two-digit multiplications, 15 three-number
sums, 10 capitals), temperature 0, scored by regex. On both GPUs, 39 of 40
answers are byte-identical to Google's build. The one difference is the same on
both: add-14, where the repack answers `948` (correct, as 12B at full size does)
and Google's `-qat-w4a16-ct` answers `1048`. The repack's scales are the Q4_0
export's and some differ from Google's by a bf16 rounding step; one question
out of 40 shows the two builds are not bit-identical and cannot rank them.

## The checkpoint's config.json

The first upload (revision `04375ba`) carried `config.json` from
`-qat-q4_0-unquantized`, whose format (`transformers` 5.10.0.dev0) predates
Google's `-qat-w4a16-ct` config (5.10.1). vLLM 0.30.0 needs two changes, both
copied from `google/gemma-4-12B-it-qat-w4a16-ct` and uploaded to the repo:

| Revision | Change | vLLM error without it |
| --- | --- | --- |
| `f386b0e` | add `vision_config.num_soft_tokens` (280), `vision_config.model_patch_size` (48), and top-level `hidden_size`, `intermediate_size`, `hidden_act`, `num_experts`, `num_experts_per_tok`, `moe_intermediate_size` | `AttributeError: 'Gemma4UnifiedVisionConfig' object has no attribute 'num_soft_tokens'` |
| `2a96908` | `quantization_config.ignore`: Google's 17 entries in place of 2 | `ValueError: There is no module or parameter named 'vision_embedder.patch_dense.weight'` |

The repack keeps three linear layers in bf16 (`vision_embedder.patch_dense`,
`embed_vision.embedding_projection`, `embed_audio.embedding_projection`); an
ignore list that omits one makes vLLM build it as int4 and look for
`weight_packed`. Checked against `repack-index.json`: Google's list names all
three and none of the 328 quantized modules. Weights were not changed. The
configs are kept here as `config-repack.json`, `config-repack-fixed.json`,
`config-repack-fixed2.json` and `config-google-qat.json`.

## Start-up and teardown

L4: created 00:40:07Z, weights loaded 00:49:01Z (69.6 s load), InService
00:52:28Z, deleted 00:57:16Z. L40S: created 00:57:21Z, weights loaded
01:04:20Z (52.4 s load), InService 01:07:23Z, deleted 01:10:33Z. Both placed at
the first attempt. `watchdog.log` shows nothing left in any of the three US
regions. One deployment of each; not re-measured.

## Files

- `run.sh`, `run.log`, `watchdog.sh`, `watchdog.log`, `rerun.sh`, `reap-l4.sh`
- `measure-*.json`, `measure-*.log`, `compare-l4.json`, `compare-l40s.json`
- `config-*.json`, `repack-index.json`
- `config-error/`, `ignore-error/` and `../2026-09-28-12b-repack-l4-{config,ignore}-error/`
  — the two deployments on the earlier configs, with their container logs
- `../2026-09-28-12b-repack-{l4,l40s}/` — deploy attempt, timeline, run
  environment, status at InService, container log, delete time

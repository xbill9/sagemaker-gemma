# 2026-09-29 — Gemma 4 E2B text-only W4A16 repack on one L4

[`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text)
(revision `315aec0`) is the E2B W4A16 repack with the vision and audio towers removed,
declared as `Gemma4ForCausalLM` (`gemma4_text`). Measured on `ml.g6.xlarge` (1× L4) the same
day and on the same instance type as the full repack (`../2026-09-29-e2b-e4b-repack/`), with
the same container tag (vLLM 0.30.0), standard settings, script and region (us-east-2).

Checked offline before the deploy (`config-text.json`, `index-text.json`): 1,092 tensors,
276 quantized modules (as in the full repack), no vision or audio tensors, and no bf16 linear
layer left for the ignore list to miss.

## Results (`compare-vs-repack.json`, `compare-vs-google.json`)

| Measure | Google QAT (09-25) | Full repack (09-29) | Text-only (09-29) |
| --- | ---: | ---: | ---: |
| Architecture | `Gemma4ForConditionalGeneration` | same | `Gemma4ForCausalLM` |
| Checkpoint (GB) | 8.32 | 7.51 | 6.56 |
| Weights in GPU memory (GiB) | 8.01 | 7.26 | 6.33 |
| Peak activation at start-up (GiB) | – | 0.79 | 5.13 |
| KV cache (tokens) | 867,999 | 929,454 | 650,783 |
| Decode, one request (tokens/s) | 105.1 | 107.2 | 103.0 |
| 1 request, 256 tokens (tokens/s) | 85.35 | 82.35 | 75.65 |
| 4 parallel (tokens/s) | 328.7 | 322.0 | 277.05 |
| 16 parallel (tokens/s) | 1077.25 | 1059.25 | 927.0 |
| Questions correct (of 40) | 37 | 37 | 36 |

- 512-token calls: text-only 5.752–6.036 s (median 5.884), full repack 5.434–5.718 s.
  16-parallel batches: 928.9 and 925.1 tokens/s.
- Per-call fixed cost (CLI, signing, network): text-only 0.912 s, full repack 0.678 s. The
  parallel rates include it; see `../2026-09-29-e2b-e4b-repack/NOTES.md`, "The parallel
  figures include the client". At the decode rates, 256 tokens take 2.485 s (text-only) and
  2.388 s (full repack), so about 4 % of the gap is generation and the rest is the client.

## Why the KV cache is smaller

Both endpoints started on fresh instances and compiled from scratch. vLLM sizes the KV cache
from what is left after weights, peak activation and CUDA graphs, and it measured very
different peaks (`../2026-09-29-e2b-text-l4/logs-full.txt`,
`../2026-09-29-e2b-repack/logs-full.txt`, `gpu_worker.py:888`):

| | Full repack | Text-only |
| --- | ---: | ---: |
| Weights + non-torch (GiB) | 7.63 | 6.70 |
| Peak activation (GiB) | 0.79 | 5.13 |
| CUDA graphs (GiB) | 0.28 | 0.28 |
| KV cache memory (GiB) | 11.34 | 7.94 |

The model card records the same effect on a T4 (4.67 GiB of peak activation on a first
start, gone after a restart that loads the compile cache). A SageMaker endpoint is always a
first start, so this is the figure it gets. The full repack also compiled cold here and did
not show it, so it follows the `Gemma4ForCausalLM` path, not the cold start alone. A warm
restart could not be tried on SageMaker.

## Quality

39 of 40 answers match the full repack. The difference is add-8 (782 + 571 − 534 = 819):
the full repack answers `819`, the text-only build `829`. Against Google's build, 38 of 40
match (add-8 and add-14).

## Start-up and teardown

Created 18:42:39Z, weights loaded 18:51:10Z (46.2 s load), InService 18:53:57Z, measured
18:54:03–18:57:09Z, deleted 18:57:15Z. `watchdog.log` shows nothing left in any of the three
US regions. One deployment.

## Files

- `run.sh`, `run.log`, `watchdog.sh`, `watchdog.log`
- `measure-gemma-4-e2b-text.json`, `.log`, `compare-vs-repack.json`, `compare-vs-google.json`
- `config-text.json`, `index-text.json`, `card-text.md`
- `../2026-09-29-e2b-text-l4/` — deploy attempt, timeline, run environment, status at
  InService, container log, delete time

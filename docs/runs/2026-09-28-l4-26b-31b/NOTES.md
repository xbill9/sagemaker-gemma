# 2026-09-28 — Gemma 4 26B A4B W4A16 and 31B QAT on one SageMaker L4

The two larger 4-bit checkpoints on a single NVIDIA L4 (24 GB), the GPU of the
E2B, E4B and 12B QAT runs. Same container digest (vLLM 0.30.0), measurement
script and region (us-east-2). Each instance type was set as a single
`InstanceType` (`../2026-09-28-l4-*/run-env.txt`).

| Endpoint | Model | Instance | vLLM settings |
| --- | --- | --- | --- |
| `gemma-4-26b-w4a16-l4` | [`xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct) | `ml.g6.xlarge` (1× L4, 16 GiB host RAM) | standard: `max_model_len` 8192, memory 0.9 |
| `gemma-4-31b-qat-l4-2xl` | `google/gemma-4-31B-it-qat-w4a16-ct` | `ml.g6.2xlarge` (1× L4, 32 GiB host RAM) | `max_model_len` 1024, memory 0.97, `max_num_seqs` 4, `max_num_batched_tokens` 1024, image/video/audio input off |

## 26B A4B W4A16 on the L4

Standard settings, directly comparable with the E2B, E4B and 12B QAT L4 rows
and with the same checkpoint on the L40S (`../2026-09-28-31b-26b-l40s/`).
Dollars per million output tokens at 16 parallel = $1.1267/h (on-demand
hosting, us-east-2, AWS Pricing API) ÷ measured throughput.

| Measure | L4 | L40S | L4 / L40S |
| --- | ---: | ---: | ---: |
| Weights in GPU memory (GiB) | 15.88 | 15.88 | – |
| KV cache (tokens) | 14,623 | 141,961 | – |
| Concurrency at 8,192 tokens | 1.79× | 17.33× | – |
| Decode, one request (tokens/s) | 64.8 | 164.7 | 0.39 |
| 1 request, 256 tokens (tokens/s) | 55.95 | 118.85 | – |
| 4 parallel (tokens/s) | 179.1 | 378.6 | – |
| 16 parallel (tokens/s) | 515.4 | 1090.45 | 0.47 |
| Questions correct (of 40) | 40 | 40 | – |
| $ per million tokens, 16 parallel | 0.61 | 0.66 | – |

- 512-token calls ran 8.396–8.554 s (median 8.475 s; 16-token median
  0.822 s). 16-parallel batches: 522.3 and 508.5 tokens/s.
- Created 18:13:08Z, weights loaded 18:21:26Z (8.3 min), InService 18:24:24Z
  (11.3 min), measured 18:24:27–18:27:49Z, deleted 18:27:55Z.

## 31B QAT on the L4

With vLLM's standard settings on `ml.g6.xlarge` the checkpoint does not start:
its 19.77 GiB of weights exceed the 0.9 memory cap on the L4's 21.96 GiB, and
the start-up profiling run's 2.62 GiB allocation finds 1.84 GiB free
(`torch.OutOfMemoryError`, `../2026-09-28-l4-31b-qat/logs-oom.txt`). It serves
with four changes:

| Change | Why |
| --- | --- |
| memory cap 0.97, `max_num_seqs` 4, `max_num_batched_tokens` 1024 | shrinks the start-up profiling allocation below the free memory |
| `max_model_len` 1024 | the measurement's prompts are about 30 tokens and replies at most 512 |
| `--limit-mm-per-prompt` image, video and audio 0 | with image input on, vLLM needs 2,496 tokens per step for one image and refuses 1,024 (`../2026-09-28-l4-31b-qat-squeezed/logs-failed.txt`); off, the vision weights are not loaded (18.7 GiB against 19.77) |
| `ml.g6.2xlarge` (32 GiB host RAM) | on `ml.g6.xlarge` (16 GiB) mapping the 23.3 GB checkpoint file failed with `Cannot allocate memory` (`../2026-09-28-l4-31b-qat-textonly/logs-failed.txt`); same single L4 GPU |

Dollars per million tokens use $1.2220/h for `ml.g6.2xlarge`. The L40S column
is the standard-settings run, so the rows compare the two deployments rather
than the GPUs alone.

| Measure | L4, settings above | L40S, standard |
| --- | ---: | ---: |
| Weights in GPU memory (GiB) | 18.7 | 19.78 |
| KV cache (tokens) | 2,555 | 28,846 |
| Decode, one request (tokens/s) | 12.6 | 35.1 |
| 1 request, 256 tokens (tokens/s) | 12.3 | 32.55 |
| 4 parallel (tokens/s) | 47.4 | 126.1 |
| 16 parallel (tokens/s) | 39.45 | 441.95 |
| Questions correct (of 40) | 40 | 40 |
| $ per million tokens, 16 parallel | 8.60 | 1.64 |

- All 40 answers are byte-identical to the L40S run.
- 512-token calls ran 41.102–41.202 s (median 41.198 s; 16-token median
  1.921 s). 16-parallel batches: 39.5 and 39.4 tokens/s.
- At 16 parallel the engine ran at most 4 requests while the rest waited
  ("Running: 4 reqs, Waiting: 12 reqs"), and the KV cache peaked at 44.5 %
  in use (`../2026-09-28-l4-31b-qat-2xl/logs-full.txt`). The 16-parallel rate
  is set by `max_num_seqs` 4, not by KV memory; 4 parallel costs $7.16 per
  million tokens.
- The first vLLM start on the instance ran out of memory on the same 2.62 GiB
  allocation; the container restarted and the second start came up. Created
  20:36:24Z, weights loaded 20:47:21Z (10.9 min), InService 20:49:11Z
  (12.8 min), measured 20:49:13–20:59:49Z, deleted 20:59:58Z.

## Order and limits

One deployment of each measured endpoint; neither re-measured. The three
31B deployments that did not start ran from 18:28Z to 20:35Z; SageMaker will
not delete an endpoint while it is Creating, so each ran until its health check
failed. `watchdog.log` and `run-31b-2xl.log` show no endpoint, endpoint config
or model left in any of the three US regions.

## Files

- `run.sh`, `run.log` — 26B and the standard-settings 31B attempt;
  `run-31b-squeezed.sh`, `run-31b-textonly.sh`, `run-31b-2xl.sh` and their
  logs — the 31B attempts in order; `watchdog.sh`, `watchdog.log`
- `measure-gemma-4-26b-w4a16-l4.*`, `measure-gemma-4-31b-qat-l4-2xl.*`
- `../2026-09-28-l4-*/` — deploy attempts, timelines, run environment,
  container logs, delete times

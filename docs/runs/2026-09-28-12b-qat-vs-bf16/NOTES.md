# 2026-09-28 — Gemma 4 12B full size vs QAT w4a16 on one SageMaker L40S

Same container, settings and measurement script as the E2B (2026-09-25) and E4B
(2026-09-27) comparisons, run one after the other on two endpoints by `run.sh`.
The instance is `ml.g6e.xlarge` (1× L40S, 48 GB) instead of `ml.g6.xlarge`: 12B
bf16 weights are 22.4 GiB, which leaves no KV cache room on a 24 GB L4. Numbers
here compare with each other, not with the L4 runs.

| | Full size | QAT |
| --- | --- | --- |
| Model | `google/gemma-4-12B-it` | `google/gemma-4-12B-it-qat-w4a16-ct` |
| Architecture | `Gemma4UnifiedForConditionalGeneration` | same |
| Endpoint | `gemma-4-12b` | `gemma-4-12b-qat` |
| Region / instance | us-east-2 / `ml.g6e.xlarge` (1× L40S) | us-east-2 / `ml.g6e.xlarge` (1× L40S) |
| Container | vLLM 0.30.0, `vllm@sha256:cc456bd7…` | same digest |
| vLLM quantization | none | `compressed-tensors` |
| Measured | `measure-gemma-4-12b.json` | `measure-gemma-4-12b-qat.json` |

Settings (both): `max_model_len` 8192, GPU memory utilization 0.9,
temperature 0. Method in `compare.py`. The instance type was set as a single
`InstanceType` (`../2026-09-28-12b-*/run-env.txt`), so SageMaker had no other
type to place either endpoint on; `compare.json` records `instance_types` as
null because the status call reads it from instance pools only.

## Results (`compare.json`, ratios computed by `compare.py combine`)

| Measure | Full size | QAT | QAT / full |
| --- | ---: | ---: | ---: |
| Weights in GPU memory (GiB) | 22.83 | 8.28 | 0.36 |
| KV cache (tokens) | 65,877 | 126,058 | 1.91 |
| Concurrency at 8,192 tokens | 8.04× | 15.39× | – |
| Weight load (s) | 187.0 | 65.4 | – |
| Decode, one request (tokens/s) | 30.0 | 77.0 | 2.57 |
| 1 request, 256 tokens (tokens/s) | 28.2 | 66.2 | 2.35 |
| 4 parallel (tokens/s) | 105.6 | 253.55 | 2.40 |
| 16 parallel (tokens/s) | 392.2 | 853.4 | 2.18 |
| Questions correct (of 40) | 40 | 39 | – |

- Decode rate is (512 − 16) / (median wall at 512 tokens − median wall at 16
  tokens), 5 calls per length, `ignore_eos`. Medians: full 17.726 s and
  1.178 s, QAT 7.179 s and 0.740 s. The same arithmetic puts the per-call
  aws CLI and network cost at 0.644 s (full) and 0.532 s (QAT).
- Spread of the 512-token calls: full 17.690–17.924 s, QAT 7.132–7.366 s.
  The ranges are far apart.
- Parallel figures are total output tokens / wall time of the whole batch,
  median of 2 batches, 256 tokens per request. The 16-parallel batches were
  395.5 and 388.9 (full), 857.3 and 849.5 (QAT).
- vLLM's own "Avg generation throughput" log lines peaked at 407.5 (full) and
  409.6 (QAT) tokens/s. They are 10-second averages over batches that last
  4–10 s, so they understate the QAT batch rate.

## Quality

40 fixed questions with exact answers (15 two-digit multiplications, 15
three-number sums, 10 capitals), temperature 0, scored by regex. Same seeded
questions as the E2B and E4B comparisons.

| Kind | Full size | QAT |
| --- | ---: | ---: |
| Multiplication | 15 / 15 | 15 / 15 |
| a + b − c | 15 / 15 | 14 / 15 |
| Capitals | 10 / 10 | 10 / 10 |

- 39 of 40 answers are byte-identical between the two models.
- The one difference is add-14: full size answers `948` (correct), QAT `1048`.
  E4B at both precisions also missed add-14.
- 40 questions detect a large quality loss; they cannot resolve a difference
  of one.

## Start-up (minutes after `create-endpoint`)

| | Full size | QAT |
| --- | ---: | ---: |
| Weights loaded | 9.1 | 7.1 |
| `InService` | 12.2 | 10.1 |

Full size created 13:25:17Z, weights loaded 13:34:25Z, InService 13:37:27Z;
QAT created 13:42:10Z, weights loaded 13:49:17Z, InService 13:52:13Z
(`../2026-09-28-12b-*/attempt-1/timeline.txt`, container log). Both were
placed in us-east-2 at the first attempt.

## Order and limits

Full size measured 13:37:29–13:42:01Z and deleted 13:42:05Z; QAT deployed on
a fresh `ml.g6e.xlarge` in the same region, measured 13:52:15–13:55:14Z and
deleted 13:55:18Z. Each endpoint ran on a different physical instance. One
deployment of each; neither has been re-measured. `watchdog.log` shows no 12B
endpoint, endpoint config or model left in any of the three US regions.

## Files

- `run.sh`, `run.log` — driver and its output; `watchdog.sh`,
  `watchdog.log` — deletes anything `run.sh` leaves in any region
- `measure-*.json`, `measure-*.log`, `compare.json`
- `../2026-09-28-12b-bf16/`, `../2026-09-28-12b-qat/` — deploy attempt,
  timeline, run environment, status at InService, container log tail, delete
  time

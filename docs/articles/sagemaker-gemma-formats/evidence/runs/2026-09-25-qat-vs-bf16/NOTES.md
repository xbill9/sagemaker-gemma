# 2026-09-25 — Gemma 4 E2B full size vs QAT w4a16 on one SageMaker L4

Same region, same instance type, same container, same measurement script, run
one after the other on two endpoints.

| | Full size | QAT |
| --- | --- | --- |
| Model | `google/gemma-4-E2B-it` | `google/gemma-4-E2B-it-qat-w4a16-ct` |
| Endpoint | `gemma-4-e2b` | `gemma-4-e2b-qat` |
| Region / instance | us-east-2 / `ml.g6.xlarge` (1× L4) | us-east-2 / `ml.g6.xlarge` (1× L4) |
| Container | vLLM 0.30.0, `vllm@sha256:cc456bd7…` | vLLM 0.30.0, same tag |
| vLLM quantization | none | `compressed-tensors` |
| Measured | `measure-gemma-4-e2b.json` | `measure-gemma-4-e2b-qat.json` |

Settings (both): `max_model_len` 8192, GPU memory utilization 0.9,
temperature 0. Method in `compare.py`.

## Results (`compare.json`, ratios computed by `compare.py combine`)

| Measure | Full size | QAT | QAT / full |
| --- | ---: | ---: | ---: |
| Weights in GPU memory (GiB) | 9.75 | 8.01 | 0.82 |
| KV cache (tokens) | 723,484 | 867,999 | 1.2 |
| Weight load (s) | 82.8 | 66.2 | – |
| Decode, one request (tokens/s) | 51.3 | 105.1 | 2.05 |
| 1 request, 256 tokens (tokens/s) | 45.6 | 85.35 | 1.87 |
| 4 parallel (tokens/s) | 171.6 | 328.7 | 1.92 |
| 16 parallel (tokens/s) | 619.1 | 1077.25 | 1.74 |
| Questions correct (of 40) | 37 | 37 | – |

- Decode rate is (512 − 16) / (median wall at 512 tokens − median wall at 16
  tokens), 5 calls per length, `ignore_eos`. It removes the aws CLI start-up
  and network time, which the same arithmetic puts at 0.633 s (full) and
  0.562 s (QAT) per call.
- Spread of the 512-token calls: full 10.491–10.661 s, QAT 5.392–5.526 s.
  The ranges are far apart.
- Parallel figures are total output tokens / wall time of the whole batch,
  median of 2 batches, 256 tokens per request. The 16-parallel batches were
  618.9 and 619.3 (full), 1051.6 and 1102.9 (QAT).
- Parallel figures include per-call client cost; the 1-request figure is
  therefore below the decode rate.
- vLLM's own "Avg generation throughput" log lines peaked at 335.3 (full) and
  409.6 (QAT) tokens/s. Those are 10-second averages, and a 16-request batch
  lasts 4–7 s, so they understate the batch rate; they agree in direction.

## Quality

40 fixed questions with exact answers (15 two-digit multiplications, 15
three-number sums, 10 capitals), temperature 0, scored by regex.

| Kind | Full size | QAT |
| --- | ---: | ---: |
| Multiplication | 15 / 15 | 15 / 15 |
| a + b − c | 12 / 15 | 12 / 15 |
| Capitals | 10 / 10 | 10 / 10 |

- 35 of 40 answers are byte-identical between the two models.
- Both miss three sums, and different ones: full size misses add-4, add-8,
  add-14; QAT misses add-7, add-9, add-14. Errors (answer − expected):
  full +100, +10, −30; QAT +300, −10, +62.
- 40 questions detect a large quality loss; they cannot resolve a small one.

## Start-up (minutes after `create-endpoint`)

| | Full size | QAT |
| --- | ---: | ---: |
| `InService` | 9.9 | 10.1 |

QAT created 18:40:32Z, InService 18:50:39Z (`../2026-09-25-e2b-qat/timeline-us-east-2.txt`).

## Capacity and quota on the way

- QAT in us-east-2 on `ml.g6.2xlarge`/`4xlarge` while the full-size endpoint
  ran: refused with `ResourceLimitExceeded` — the full-size endpoint's
  InstancePools list held quota for all three types
  (`../2026-09-25-e2b-qat/attempt-1/error.txt`).
- QAT in us-west-2: `InsufficientInstanceCapacity` after about 31 minutes.
- QAT in us-east-1: stopped after a few minutes in favour of reusing us-east-2.

## Order and limits

Full size measured first (window starts in `measure-gemma-4-e2b.json`), then
deleted, then QAT deployed on a fresh `ml.g6.xlarge` in the same region and
measured. Each endpoint ran on a different physical instance. One deployment
of each; the full-size model has not been re-measured after QAT.

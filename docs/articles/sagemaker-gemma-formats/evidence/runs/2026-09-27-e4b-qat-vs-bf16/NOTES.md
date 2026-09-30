# 2026-09-27 — Gemma 4 E4B full size vs QAT w4a16 on one SageMaker L4

Same region, same instance type, same container, same measurement script as the
2026-09-25 E2B comparison (`../2026-09-25-qat-vs-bf16/`), run one after the
other on two endpoints by `run.sh`.

| | Full size | QAT |
| --- | --- | --- |
| Model | `google/gemma-4-E4B-it` | `google/gemma-4-E4B-it-qat-w4a16-ct` |
| Endpoint | `gemma-4-e4b` | `gemma-4-e4b-qat` |
| Region / instance | us-east-2 / `ml.g6.xlarge` (1× L4) | us-east-2 / `ml.g6.xlarge` (1× L4) |
| Container | vLLM 0.30.0, `vllm@sha256:cc456bd7…` | same digest |
| vLLM quantization | none | `compressed-tensors` (Marlin WNA16 kernel) |
| Measured | `measure-gemma-4-e4b.json` | `measure-gemma-4-e4b-qat.json` |

Settings (both): `max_model_len` 8192, GPU memory utilization 0.9,
temperature 0. Method in `compare.py`. The container was pinned to the
`vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1` tag, which resolves
to the same digest the E2B runs used; the newest tag in the region on this date
was `sagemaker-v1.2`.

## Results (`compare.json`, ratios computed by `compare.py combine`)

| Measure | Full size | QAT | QAT / full |
| --- | ---: | ---: | ---: |
| Weights in GPU memory (GiB) | 15.08 | 11.04 | 0.73 |
| KV cache (tokens) | 94,853 | 200,972 | 2.12 |
| Concurrency at 8,192 tokens | 11.58× | 24.53× | – |
| Weight load (s) | 125.6 | 91.1 | – |
| Decode, one request (tokens/s) | 26.2 | 60.7 | 2.32 |
| 1 request, 256 tokens (tokens/s) | 23.15 | 46.45 | 2.01 |
| 4 parallel (tokens/s) | 88.3 | 179.35 | 2.03 |
| 16 parallel (tokens/s) | 323.6 | 599.5 | 1.85 |
| Questions correct (of 40) | 36 | 36 | – |

- Decode rate is (512 − 16) / (median wall at 512 tokens − median wall at 16
  tokens), 5 calls per length, `ignore_eos`. Medians: full 20.788 s and
  1.843 s, QAT 9.611 s and 1.438 s. The same arithmetic puts the per-call
  aws CLI and network cost at 1.232 s (full) and 1.174 s (QAT).
- Spread of the 512-token calls: full 20.738–20.975 s, QAT 9.582–9.659 s.
  The ranges are far apart.
- Parallel figures are total output tokens / wall time of the whole batch,
  median of 2 batches, 256 tokens per request. The 16-parallel batches were
  326.2 and 321.0 (full), 605.7 and 593.3 (QAT).
- vLLM's own "Avg generation throughput" log lines peaked at 299.5 (full) and
  409.6 (QAT) tokens/s. They are 10-second averages over batches that last
  7–13 s, so they understate the batch rate; they agree in direction.

## Quality

40 fixed questions with exact answers (15 two-digit multiplications, 15
three-number sums, 10 capitals), temperature 0, scored by regex. Same seeded
questions as the E2B comparison.

| Kind | Full size | QAT |
| --- | ---: | ---: |
| Multiplication | 15 / 15 | 15 / 15 |
| a + b − c | 11 / 15 | 11 / 15 |
| Capitals | 10 / 10 | 10 / 10 |

- 37 of 40 answers are byte-identical between the two models.
- Both miss the same four sums: add-0, add-3, add-9, add-14. Errors
  (answer − expected): full −190, +10, +300, +10; QAT −200, −20, +300, +110.
  add-9 is the same wrong answer (`25`) from both.
- E2B missed three sums at either precision; E4B misses four. 40 questions
  detect a large quality loss; they cannot resolve a difference of one.

## Start-up (minutes after `create-endpoint`)

| | Full size | QAT |
| --- | ---: | ---: |
| Weights loaded | 8.1 | 7.6 |
| `InService` | 11.8 | 10.6 |

Full size created 20:26:20Z, InService 20:38:06Z; QAT created 20:43:57Z,
InService 20:54:35Z (`../2026-09-27-e4b-*/attempt-1/timeline.txt`). Both were
placed on `ml.g6.xlarge` in us-east-2 at the first attempt; no capacity
failures this run.

## Against E2B (2026-09-25, same instance, container and script)

| Measure | E2B full | E2B QAT | E4B full | E4B QAT |
| --- | ---: | ---: | ---: | ---: |
| Weights (GiB) | 9.75 | 8.01 | 15.08 | 11.04 |
| KV cache (tokens) | 723,484 | 867,999 | 94,853 | 200,972 |
| Decode (tokens/s) | 51.3 | 105.1 | 26.2 | 60.7 |
| 16 parallel (tokens/s) | 619.1 | 1077.25 | 323.6 | 599.5 |
| Questions correct (of 40) | 37 | 37 | 36 | 36 |

- E4B at full size runs at about half of E2B at full size throughout.
- QAT saves more on E4B (4.04 GiB, 0.73×) than on E2B (1.74 GiB, 0.82×).
  On a 24 GB L4 that saving is a large share of what is left after the
  weights, so the E4B KV cache grows 2.12× where E2B's grew 1.2×.
- E4B QAT decodes faster than E2B full size (60.7 vs 51.3 tokens/s) and is
  within 4 % of it at 16 parallel requests (599.5 vs 619.1).

## Order and limits

Full size measured 20:38:12–20:43:42Z and deleted 20:43:48Z; QAT deployed on
a fresh `ml.g6.xlarge` in the same region, measured 20:54:40–20:58:31Z and
deleted 20:58:37Z. Each endpoint ran on a different physical instance. One
deployment of each; neither has been re-measured. All three US regions
showed no gemma endpoints after the run.

## Files

- `run.sh`, `run.log` — driver and its output; `watchdog.sh`,
  `watchdog.log` — deletes the QAT endpoint if `run.sh` dies (it did not)
- `measure-*.json`, `measure-*.log`, `compare.json`
- `STATUS.md` — mid-run status written for the unattended part of the run
- `../2026-09-27-e4b-bf16/`, `../2026-09-27-e4b-qat/` — deploy attempt,
  timeline, status at InService, container log tail, delete time

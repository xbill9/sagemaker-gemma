# 2026-09-28 — Gemma 4 12B QAT on one L4 against one L40S

`google/gemma-4-12B-it-qat-w4a16-ct` on `ml.g6.xlarge` (1× L4, 24 GB), the
instance the E2B and E4B comparisons used, set beside the same checkpoint on
`ml.g6e.xlarge` (1× L40S, 48 GB) from `../2026-09-28-12b-qat-vs-bf16/`. Same
container digest (vLLM 0.30.0), settings (`max_model_len` 8192, GPU memory
utilization 0.9, temperature 0), measurement script and region (us-east-2).
The instance type was set as a single `InstanceType` (`deploy/run-env.txt`).

## Results (`compare-l4-vs-l40s.json`, ratios computed by `compare.py combine`)

| Measure | L40S | L4 | L4 / L40S |
| --- | ---: | ---: | ---: |
| Weights in GPU memory (GiB) | 8.28 | 8.28 | 1.00 |
| KV cache (tokens) | 126,058 | 41,651 | 0.33 |
| Concurrency at 8,192 tokens | 15.39× | 5.08× | – |
| Weight load (s) | 65.4 | 75.3 | – |
| Decode, one request (tokens/s) | 77.0 | 29.3 | 0.38 |
| 1 request, 256 tokens (tokens/s) | 66.2 | 27.0 | 0.41 |
| 4 parallel (tokens/s) | 253.55 | 105.35 | 0.42 |
| 16 parallel (tokens/s) | 853.4 | 358.05 | 0.42 |
| Questions correct (of 40) | 39 | 39 | – |

- Decode rate is (512 − 16) / (median wall at 512 tokens − median wall at 16
  tokens), 5 calls per length, `ignore_eos`. L4 medians 18.207 s and
  1.280 s; per-call aws CLI and network cost 0.734 s. The L4's 512-token calls
  ran 18.168–18.277 s, the L40S's 7.132–7.366 s.
- The L4's 16-parallel batches were 356.4 and 359.7 tokens/s.
- All 40 answers are byte-identical between the two GPUs.

## Cost

SageMaker hosting, on demand, us-east-2 (AWS Pricing API, 2026-09-28):
`ml.g6.xlarge` $1.1267/h, `ml.g6e.xlarge` $2.6054/h.

| | L40S | L4 |
| --- | ---: | ---: |
| $ per hour | 2.6054 | 1.1267 |
| $ per million output tokens at 16 parallel | 0.848 | 0.874 |

The L4 costs 1.03× the L40S per token at 16 parallel requests. It would need
369.1 tokens/s to match the L40S; it delivered 358.05, 3.0 % short. The cost
per token assumes the endpoint is kept busy at that load.

## Start-up (minutes after `create-endpoint`)

Created 16:30:23Z, weights loaded 16:46:17Z (15.9 min), InService 16:49:22Z
(19.0 min). Measured 16:49:25–16:54:05Z; deleted 16:54:06Z in us-east-2, and
the cleanup found nothing in us-west-2 or us-east-1 (`run.log`). One
deployment; not re-measured.

## Files

- `run.sh`, `run.log` — driver, output and cleanup
- `measure-gemma-4-12b-qat-l4.json`, `.log`, `compare-l4-vs-l40s.json`
- `deploy/` — attempt, timeline, run environment, status at InService,
  container log tail, delete time

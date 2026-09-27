# 2026-09-27 — Gemma 4 E4B bf16 vs QAT: run status

Status as of 20:56 UTC. The run continues unattended; this file is not updated
by it. `run.log` is the live record.

## Where it stands

| Step | State |
| --- | --- |
| bf16 `google/gemma-4-E4B-it`, endpoint `gemma-4-e4b` | **Done.** Measured 20:38–20:43, deleted 20:43:50 |
| QAT `google/gemma-4-E4B-it-qat-w4a16-ct`, endpoint `gemma-4-e4b-qat` | InService 20:54:35; measuring since 20:54:38 |
| QAT measurement, then delete | Running; ~5 min, then automatic delete |
| `compare.json` (QAT / bf16 ratios) | Automatic after both measurements; `run.log` ends with `ALL DONE` |
| NOTES.md write-up | **Not done** — needs a Claude session, from the files below |

Both endpoints: us-east-2, `ml.g6.xlarge` (1× L4), vLLM 0.30.0
`vllm@sha256:cc456bd7…` (same image as the 09-25 E2B runs, pinned to the
`sagemaker-v1.1` tag; newest is now v1.2), `max_model_len` 8192, GPU memory
utilization 0.9.

## Results so far

| Measure | E4B bf16 | E4B QAT | E2B bf16 (09-25) |
| --- | ---: | ---: | ---: |
| Weights in GPU memory (GiB) | 15.08 | 11.04 | 9.75 |
| Weight load (s) | 125.6 | 91.1 | 82.8 |
| KV cache (tokens) | 94,853 | 200,972 | 723,484 |
| Decode, one request (tokens/s) | 26.2 | pending | 51.3 |
| 1 request, 256 tokens (tokens/s) | 23.15 | pending | 45.6 |
| 4 parallel (tokens/s) | 88.3 | pending | 171.6 |
| 16 parallel (tokens/s) | 323.6 | pending | 619.1 |
| Questions correct (of 40) | 36 (mul 15, add 11, cap 10) | pending | 37 |
| `create-endpoint` → InService (min) | 11.8 | 10.6 | 9.9 |

## Safety nets

- `watchdog.sh` (own session, survives Claude Code exiting): if `run.sh`
  (pid 20173) dies while `gemma-4-e4b-qat` still exists, it deletes the
  endpoint. Output in `watchdog.log`.
- A `systemd-inhibit` block on lid-close and sleep lasts as long as `run.sh`.
- If the laptop suspends anyway, whatever endpoint is up keeps billing.

## Checking afterwards

```bash
tail docs/runs/2026-09-27-e4b-qat-vs-bf16/run.log        # expect "ALL DONE"
cat  docs/runs/2026-09-27-e4b-qat-vs-bf16/watchdog.log   # expect "endpoint NotFound"
AWS_REGION=us-east-2 python3 sm.py list                  # expect count 0 = nothing billing
AWS_REGION=us-east-2 ENDPOINT_NAME=gemma-4-e4b-qat python3 sm.py destroy   # if not
```

If `run.log` lacks `ALL DONE` but both `measure-*.json` exist:

```bash
python3 compare.py combine docs/runs/2026-09-27-e4b-qat-vs-bf16/compare.json \
  docs/runs/2026-09-27-e4b-qat-vs-bf16/measure-gemma-4-e4b.json \
  docs/runs/2026-09-27-e4b-qat-vs-bf16/measure-gemma-4-e4b-qat.json
```

## Files

- `run.sh`, `run.log` — driver and its output
- `measure-gemma-4-e4b.{json,log}` — bf16 measurement (done)
- `measure-gemma-4-e4b-qat.{json,log}` — QAT measurement (pending)
- `compare.json` — combined, pending
- `../2026-09-27-e4b-bf16/`, `../2026-09-27-e4b-qat/` — deploy attempts,
  timelines, status at InService, log tail, delete time

Nothing is committed yet.

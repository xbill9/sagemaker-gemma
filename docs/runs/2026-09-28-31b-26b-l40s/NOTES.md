# 2026-09-28 — Gemma 4 26B A4B W4A16 and 31B QAT vs NVFP4 on one SageMaker L40S

Three 4-bit checkpoints on `ml.g6e.xlarge` (1× L40S, 48 GB), the instance of
the 12B comparison (`../2026-09-28-12b-qat-vs-bf16/`), one endpoint at a time.
Same container digest (vLLM 0.30.0), settings (`max_model_len` 8192, GPU
memory utilization 0.9, temperature 0), measurement script and region
(us-east-2). Each instance type was set as a single `InstanceType`
(`../2026-09-28-*/run-env.txt`).

| Endpoint | Model | vLLM quantization | Kernel |
| --- | --- | --- | --- |
| `gemma-4-26b-w4a16` | [`xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct) | `compressed-tensors` | – |
| `gemma-4-31b-qat` | `google/gemma-4-31B-it-qat-w4a16-ct` | `compressed-tensors` | – |
| `gemma-4-31b-nvfp4` | `nvidia/Gemma-4-31B-IT-NVFP4` | `modelopt_fp4` | `MarlinNvFp4LinearKernel` |

The L40S has no FP4 hardware. For the NVFP4 checkpoint vLLM logged: "Your GPU
does not have native support for FP4 computation but FP4 quantization is being
used. Weight-only FP4 compression will be used leveraging the Marlin kernel."
(`../2026-09-28-31b-nvfp4/`, container log 14:47:14Z). The NVFP4 figures here
measure that checkpoint with 4-bit weights and 16-bit arithmetic; native NVFP4
needs a Blackwell GPU. The checkpoint keeps every attention layer in bf16, which
is why it holds more memory than the W4A16 QAT build.

## Results (`compare-31b-nvfp4-vs-qat.json` and `measure-*.json`)

Ratios computed by `compare.py combine`; dollars per million output tokens at
16 parallel = $2.6054/h (on-demand hosting, us-east-2, AWS Pricing API) ÷
measured throughput.

| Measure | 26B A4B W4A16 | 31B QAT | 31B NVFP4 | NVFP4 / QAT |
| --- | ---: | ---: | ---: | ---: |
| Weights in GPU memory (GiB) | 15.88 | 19.78 | 31.18 | 1.58 |
| KV cache (tokens) | 141,961 | 28,846 | 21,870 | 0.76 |
| Concurrency at 8,192 tokens | 17.33× | 3.52× | 2.67× | – |
| Weight load (s) | 127.5 | 168.5 | 235.2 | – |
| Decode, one request (tokens/s) | 164.7 | 35.1 | 22.2 | 0.63 |
| 1 request, 256 tokens (tokens/s) | 118.85 | 32.55 | 21.1 | 0.65 |
| 4 parallel (tokens/s) | 378.6 | 126.1 | 81.5 | 0.65 |
| 16 parallel (tokens/s) | 1090.45 | 441.95 | 307.65 | 0.70 |
| Questions correct (of 40) | 40 | 40 | 40 | – |
| $ per million tokens, 16 parallel | 0.66 | 1.64 | 2.35 | – |

- Decode rate is (512 − 16) / (median wall at 512 tokens − median wall at 16
  tokens), 5 calls per length, `ignore_eos`. 512-token medians and ranges:
  26B 3.775 s (3.724–3.918), 31B QAT 15.179 s (15.128–15.216), 31B NVFP4
  23.681 s (23.566–23.754). The ranges do not overlap.
- 16-parallel batches: 26B 1083.0 and 1097.9, 31B QAT 438.8 and 445.1,
  31B NVFP4 306.7 and 308.6 tokens/s.
- vLLM's "Avg generation throughput" log lines peaked at 409.6 (26B), 352.9
  (31B QAT) and 313.2 (31B NVFP4) tokens/s. They are 10-second averages over
  batches that last 2–14 s, so they understate the shorter batches.
- 26B A4B activates about 4B of its 26B parameters per token, which is why it
  decodes 4.7× faster than the dense 31B QAT on the same GPU (164.7 vs 35.1).

## Quality

40 fixed questions with exact answers (15 two-digit multiplications, 15
three-number sums, 10 capitals), temperature 0, scored by regex; the same
seeded questions as the E2B, E4B and 12B comparisons. All three answer 40 of
40. 31B QAT and 31B NVFP4 give byte-identical answers to all 40.

## Start-up (minutes after `create-endpoint`)

| | 26B A4B W4A16 | 31B QAT | 31B NVFP4 |
| --- | ---: | ---: | ---: |
| Weights loaded | 7.8 | 17.1 | 9.5 |
| `InService` | 11.7 | 20.8 | 12.7 |

Created / InService / deleted: 26B 14:09:21 / 14:21:01 / 14:23:42Z; 31B NVFP4
14:37:46 / 14:50:28 / 14:55:59Z; 31B QAT 16:54:28 / 17:15:13 / 17:19:35Z
(`../2026-09-28-*/attempt-1/timeline.txt`). The first 31B QAT deployment
(14:23:46–14:37:41Z, `../2026-09-28-31b-qat-throttled/`) loaded the same
19.78 GiB and 28,846 KV tokens; its measurement did not complete
(`measure-gemma-4-31b-qat-throttled.log`), and the table uses the second.

## 31B full size: not measured

`google/gemma-4-31B-it` (62.5 GB) does not fit one L40S, so it was queued on
`ml.g6e.12xlarge` (4× L40S) at tensor parallel 4. SageMaker returned
`InsufficientInstanceCapacity` in us-east-2, us-west-2 and us-east-1, each
after about 31 minutes (`../2026-09-28-31b-bf16/attempt-{1,2,3}/`). No
instance started.

## Order and limits

One deployment of each measured endpoint; none re-measured. `watchdog.log`
and `rerun-31b-qat.log` show no endpoint, endpoint config or model left in any
of the three US regions.

## Files

- `run.sh`, `run.log` — the queue; `rerun-31b-qat.sh`, `rerun-31b-qat.log` —
  the second 31B QAT deployment; `watchdog.sh`, `watchdog.log`
- `measure-*.json`, `measure-*.log`, `compare-31b-nvfp4-vs-qat.json`
- `../2026-09-28-{26b-w4a16,31b-qat,31b-nvfp4,31b-bf16}/` — deploy attempts,
  timelines, run environment, status at InService, container log tail,
  delete time

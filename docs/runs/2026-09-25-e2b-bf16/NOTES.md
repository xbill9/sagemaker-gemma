# 2026-09-25 — Gemma 4 E2B (full size) on a SageMaker real-time endpoint

Baseline for the full-size vs QAT comparison. Same prompts and sampling will be
used for `google/gemma-4-E2B-it-qat-w4a16-ct`.

## Configuration

| Setting | Value |
| --- | --- |
| Model | `google/gemma-4-E2B-it` (bf16, Apache-2.0, not gated) |
| Container | `763104351884.dkr.ecr.<region>.amazonaws.com/vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1` (digest below) |
| Region | us-east-1 (attempts 1–2, no capacity), **us-east-2** (attempt 3, served) |
| Instance | attempt 1: `ml.g6.xlarge`; attempts 2–3: pools `ml.g6.xlarge` → `ml.g6.2xlarge` → `ml.g6.4xlarge` (all 1× NVIDIA L4, 24 GB); placed on `ml.g6.xlarge` |
| vLLM settings | `SM_VLLM_MAX_MODEL_LEN=8192`, `SM_VLLM_GPU_MEMORY_UTILIZATION=0.9` |
| Endpoint quota | 1 for each of the three instance types (`make quota`) |
| Deployed with | three aws CLI calls: `create-model`, `create-endpoint-config`, `create-endpoint` |

## Finding: GPU capacity sets the time to a serving endpoint

| Attempt | Instance types | Creating at (UTC) | Outcome |
| --- | --- | --- | --- |
| 1 | `ml.g6.xlarge` | 14:57:52 | **Failed after 30.9 min**: `InsufficientInstanceCapacity` |
| 2 | `ml.g6.xlarge`, `ml.g6.2xlarge`, `ml.g6.4xlarge` (instance pools) | 15:29:34 | **Failed after 31.5 min**: `InsufficientInstanceCapacity` |
| 3 | same pools, **us-east-2** | 16:01:09 | **InService after 9.9 min** on `ml.g6.xlarge` |

From the first request to a serving endpoint: 73.2 minutes, of which 9.9 were
the model starting.

- The account holds a quota of 1 for each of these instance types, and us-east-1
  had no L4 instance free to place against it.
- SageMaker reports a capacity failure about 30 minutes after the request.
  Attempt 1 sat in `Creating` for 30.9 minutes with no instance, no container
  and no CloudWatch log group, then failed. The one outside sign of a capacity
  wait is that the log group `/aws/sagemaker/Endpoints/<name>` never appears.
- No charge accrues while an endpoint waits for capacity.
- The failure message recommends `InstancePools` in the endpoint config: up to
  five instance types tried in priority order. It is a field on the production
  variant in `aws sagemaker create-endpoint-config` and replaces `InstanceType`.
  Listing only same-GPU types keeps a benchmark valid whichever one is placed.

## Start-up on the L4 (attempt 3, us-east-2)

| Phase | Minutes after `create-endpoint` |
| --- | --- |
| Weights loaded (9.75 GiB, load itself 82.8 s) | 7.4 |
| KV cache sized: 723,484 tokens, 88.32× concurrency at 8,192 tokens | 9.2 |
| `InService` | 9.9 |

Image digest: `vllm@sha256:cc456bd79ed75d757d888015486da47520d297e63266faa450253dc3bcd54ef7`.
The request format in `docs/DEPLOY.md` Step 8 (OpenAI chat body to
`invoke-endpoint`) works unchanged against this container.

## Benchmark (bench.py, 5 prompts × 3 repeats, temperature 0, max 512 tokens)

| Prompt | Output tokens | Wall s (median) | Tokens/s (median) | Repeats identical | Correct |
| --- | ---: | ---: | ---: | --- | --- |
| short-fact | 3 | 0.646 | 4.6 | yes | 3 of 3 |
| arithmetic | 4 | 0.706 | 5.7 | yes | 3 of 3 |
| explain | 71 | 2.001 | 35.5 | yes | – |
| code | 155 | 3.582 | 43.3 | yes | – |
| long | 346 | 7.280 | 47.5 | no (330–346 tokens) | – |

- Wall time is measured around the `aws sagemaker-runtime invoke-endpoint`
  process, so it includes CLI start-up and the round trip from the client. The
  short prompts (3–4 tokens in about 0.65 s) show that fixed cost; tokens/s is
  meaningful only for the longer outputs, where generation runs at 43–48 tokens/s
  for a single request.
- SageMaker's own routing overhead (CloudWatch `OverheadLatency`) averaged
  0.033 s.
- At temperature 0 four of five prompts gave identical text on every repeat;
  the 346-token answer varied in length.

Per-call records: `bench-calls.json`; computed summary and CloudWatch data:
`bench.json`.

Raw records: `attempt-1/` (deploy output, failure status, timeline),
`attempt-2/`, `attempt-3/`, `failover.log`, `describe-endpoint.json`,
`quotas-us.txt`, `step8-req.json`, `step8-out.json`.

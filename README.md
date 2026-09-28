# sagemaker-gemma

Gemma 4 on an Amazon SageMaker real-time endpoint, deployed and called with the
`aws` CLI, and exposed to Gemini CLI or Claude Code as a stdio MCP server.

`server.py` is the MCP server, laid out like `~/gemma4-dev/gpu-vllm-g5g-2b`
(`MCPServer`, titled tools with read-only / write / destructive annotations).
Every AWS call is an `aws` CLI subprocess in `sm.py`; there is no boto3, so
`aws login` sessions renew on their own inside a long-running server. Runs on
the system `python3`.

## Setup

```bash
aws login
make install          # python3 -m pip install -r requirements.txt (mcp 2.x)
cp .env.example .env  # region, model, instance types, endpoint name (gitignored)
make test             # offline unittest; the aws CLI is faked
```

## Deploy, call, tear down

```bash
make quota     # endpoint quota for INSTANCE_TYPE; 0 means request an increase first
make image     # newest SageMaker vLLM container in the region
make deploy    # create-model, create-endpoint-config, create-endpoint (billing starts)
make wait      # aws sagemaker wait endpoint-in-service
make invoke PROMPT="Why is the sky blue?"
make destroy   # delete endpoint, endpoint config and model (billing stops)
```

`make deploy` creates `sagemaker-gemma-execution-role` (trusts SageMaker, has
`AmazonSageMakerFullAccess`) if it does not exist. The container downloads the
model from Hugging Face at start-up; Gemma 4 is Apache-2.0 and not gated, so no
token is needed.

The raw CLI call behind `make invoke`:

```bash
echo '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":128}' > req.json
aws sagemaker-runtime invoke-endpoint --endpoint-name gemma-4-e2b \
  --content-type application/json --body fileb://req.json out.json
jq -r '.choices[0].message.content' out.json
```

## Results: full size vs QAT on SageMaker

### E2B and E4B on one L4

Same `ml.g6.xlarge` (1× L4, 24 GB), vLLM 0.30.0 container digest,
`max_model_len` 8192 and measurement script (`compare.py`) for all four
endpoints. QAT is the `-qat-w4a16-ct` checkpoint, served with vLLM's
`compressed-tensors` quantization. Figures are copied from the run records;
ratios are computed by `compare.py combine`.

| Measure | E2B full | E2B QAT | E4B full | E4B QAT |
| --- | ---: | ---: | ---: | ---: |
| Weights (GiB) | 9.75 | 8.01 | 15.08 | 11.04 |
| KV cache (tokens) | 723,484 | 867,999 | 94,853 | 200,972 |
| Decode, one request (tokens/s) | 51.3 | 105.1 | 26.2 | 60.7 |
| 16 parallel (tokens/s) | 619.1 | 1077.25 | 323.6 | 599.5 |
| Questions correct (of 40) | 37 | 37 | 36 | 36 |

E4B QAT against E4B full size: 0.73× the weight memory, 2.12× the KV cache,
2.32× the single-request decode rate, 1.85× at 16 parallel requests, and the
same 36 of 40 questions correct (37 of 40 answers byte-identical). E4B QAT
decodes faster than E2B full size (60.7 vs 51.3 tokens/s).

All four ran in us-east-2. E2B full size was deployed twice and the second
run matched the first within 1 % on every rate; the other three were deployed
once. Method, start-up times and per-question detail:

- E2B, 2026-09-25: [`docs/runs/2026-09-25-qat-vs-bf16/NOTES.md`](docs/runs/2026-09-25-qat-vs-bf16/NOTES.md)
- E4B, 2026-09-27: [`docs/runs/2026-09-27-e4b-qat-vs-bf16/NOTES.md`](docs/runs/2026-09-27-e4b-qat-vs-bf16/NOTES.md)

### 12B on one L40S

12B at full size holds 22.4 GiB of weights, which leaves no room for a KV cache
on a 24 GB L4, so both 12B endpoints ran on `ml.g6e.xlarge` (1× L40S, 48 GB),
same container, settings and script, us-east-2, 2026-09-28. Compare these two
columns with each other, not with the L4 table above.

| Measure | 12B full | 12B QAT | QAT / full |
| --- | ---: | ---: | ---: |
| Weights (GiB) | 22.83 | 8.28 | 0.36 |
| KV cache (tokens) | 65,877 | 126,058 | 1.91 |
| Decode, one request (tokens/s) | 30.0 | 77.0 | 2.57 |
| 16 parallel (tokens/s) | 392.2 | 853.4 | 2.18 |
| Questions correct (of 40) | 40 | 39 | – |

39 of 40 answers are byte-identical; QAT's one miss is a three-number sum.
One deployment of each:
[`docs/runs/2026-09-28-12b-qat-vs-bf16/NOTES.md`](docs/runs/2026-09-28-12b-qat-vs-bf16/NOTES.md).

### 26B MoE and 31B, 4-bit, on one L40S

Same `ml.g6e.xlarge`, container, settings and script as 12B, us-east-2,
2026-09-28. Neither full size fits one L40S (26B ≈ 48 GiB, 31B 62.5 GB).
26B is [`xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct),
since Google publishes no `-qat-w4a16-ct` for that size. Dollars per million
output tokens are the on-demand hosting price ($2.6054/h) divided by the
16-parallel rate.

| Measure | 26B A4B W4A16 | 31B QAT | 31B NVFP4 | 12B QAT |
| --- | ---: | ---: | ---: | ---: |
| Weights (GiB) | 15.88 | 19.78 | 31.18 | 8.28 |
| KV cache (tokens) | 141,961 | 28,846 | 21,870 | 126,058 |
| Decode, one request (tokens/s) | 164.7 | 35.1 | 22.2 | 77.0 |
| 16 parallel (tokens/s) | 1090.45 | 441.95 | 307.65 | 853.4 |
| Questions correct (of 40) | 40 | 40 | 40 | 39 |
| $ per million tokens | 0.66 | 1.64 | 2.35 | 0.85 |

- The 26B MoE activates about 4B parameters per token and decodes 4.7× faster
  than the dense 31B QAT, at the lowest cost per token of any L40S run.
- `nvidia/Gemma-4-31B-IT-NVFP4` runs on the L40S through vLLM's Marlin
  fallback (4-bit weights, 16-bit arithmetic; FP4 hardware is Blackwell only).
  It keeps attention in bf16, holds 1.58× QAT's weight memory, and decodes at
  0.63× the rate. Both 31B builds give the same 40 answers.
- 31B full size needs four GPUs; `ml.g6e.12xlarge` (4× L40S) had no capacity
  in us-east-2, us-west-2 or us-east-1 on the day.

Details: [`docs/runs/2026-09-28-31b-26b-l40s/NOTES.md`](docs/runs/2026-09-28-31b-26b-l40s/NOTES.md).

### Same checkpoint, L4 against L40S

12B QAT on `ml.g6.xlarge` (1× L4) against the L40S run above, 2026-09-28:

| Measure | L4 | L40S | L4 / L40S |
| --- | ---: | ---: | ---: |
| $ per hour | 1.1267 | 2.6054 | – |
| KV cache (tokens) | 41,651 | 126,058 | 0.33 |
| Decode, one request (tokens/s) | 29.3 | 77.0 | 0.38 |
| 16 parallel (tokens/s) | 358.05 | 853.4 | 0.42 |
| $ per million tokens | 0.874 | 0.848 | 1.03 |

The L40S costs 2.31× the L4 per hour and returns 2.38× the tokens at 16
parallel, so a busy endpoint costs about the same per token on either; one
user's reply arrives 2.6× faster on the L40S. All 40 answers are
byte-identical across the two GPUs:
[`docs/runs/2026-09-28-12b-qat-l4/NOTES.md`](docs/runs/2026-09-28-12b-qat-l4/NOTES.md).

## 12B, 26B MoE and 31B on TPU, full size vs QAT

These sizes run on TPU v6e through vLLM's JAX path, with the int4 support
from [vllm-project/tpu-inference#3653](https://github.com/vllm-project/tpu-inference/pull/3653)
(dense, merged) and [#3660](https://github.com/vllm-project/tpu-inference/pull/3660)
(MoE, open). The setup and measurement differ from the SageMaker runs above
(`max_model_len` 2048, 16 concurrent requests × 256 tokens, median of 3; a
3,880-record public suite scored from label probabilities), so compare the
columns here with each other, never with the SageMaker tables.

**12B, both on one v6e-1 chip, same image, patches and flags** (2026-09-26,
served as `Gemma4ForCausalLM` through `--hf_overrides`).

| Measure | Full size | QAT | QAT / full |
| --- | ---: | ---: | ---: |
| HBM used (GiB) | 22.18 | 9.46 | 0.43 |
| KV cache (tokens) | 20,480 | 60,160 | 2.94 |
| 16 parallel (tokens/s) | 680.9 | 992.2 | 1.46 |
| Suite correct | 76.0 % | 75.1 % | −0.9 points (−1.6 to −0.2) |

**26B A4B, both on one v6e-4 at TP=4, same image, patches and flags**
(2026-09-26 and 09-27). Google publishes no `-qat-w4a16-ct` for this size, so
the QAT column is [`xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct):
Google's `-qat-q4_0-unquantized` weights written out as compressed-tensors
W4A16 (group size 32), with every group back on its 4-bit grid.

| Measure | Full size | QAT | QAT / full |
| --- | ---: | ---: | ---: |
| HBM used, 4 chips (GiB) | 61.16 | 21.75 | 0.36 |
| KV cache (tokens) | 235,008 | 407,168 | 1.73 |
| 16 parallel (tokens/s) | 1982.0 | 2072.9 | 1.05 |
| Suite correct | 76.4 % | 75.3 % | −1.1 points (−1.9 to −0.3) |

QAT also fits on a single v6e-1, where full size does not: 17.43 GiB,
53,888 KV tokens, 1283.3 tokens/s, 75.3 % on the suite.

**31B: only QAT has been measured this way.** `google/gemma-4-31B-it-qat-w4a16-ct`
on one v6e-1 (2026-09-25): 21.67 GiB, 8,320 KV tokens, 499.9 tokens/s at
16 parallel, 77.6 % on the suite. Full size (58.4 GiB) needs four chips and has
been run only on a v6e-4 with a different vLLM build and benchmark, so there is
no matched full-size figure to set beside it.

## MCP tools

| Tool | Kind | What it does |
| --- | --- | --- |
| `get_help` | read | Resolved configuration and order of work |
| `get_deployment_config` | read | The aws CLI deploy commands, without running them |
| `find_vllm_image` | read | Highest pinned SageMaker vLLM image in a region |
| `check_quotas` | read | Endpoint quota per instance type across the US regions |
| `deploy_endpoint` | write | Creates the endpoint with a same-GPU fallback list (billed until deleted) |
| `get_endpoint_status` | read | Creating / InService / Failed with reason, placed instance type |
| `list_endpoints` | read | Matching endpoints across the US regions |
| `get_endpoint_logs` | read | Recent CloudWatch container logs |
| `verify_model_health` | read | One short chat request, checks for a non-empty reply |
| `query_model` | read | Prompt → reply, token counts, wall time |
| `delete_endpoint` | destructive | Deletes endpoint, config and model |

Registered as `python3 server.py` in `.mcp.json` (Claude Code) and
`.gemini/settings.json` (Gemini CLI).

## Configuration (`.env`)

| Variable | Default |
| --- | --- |
| `AWS_REGION` | `us-east-1` |
| `MODEL_ID` | `google/gemma-4-E2B-it` |
| `INSTANCE_TYPE` | `ml.g6.xlarge` (1× L4, 24 GB) |
| `INSTANCE_POOLS` | fallback list, e.g. `ml.g6.xlarge,ml.g6.2xlarge,ml.g6.4xlarge` |
| `ENDPOINT_NAME` | `gemma-4-e2b` |
| `MAX_MODEL_LEN` | `8192` |
| `TENSOR_PARALLEL_SIZE` | empty → 1 GPU; e.g. `4` on `ml.g6e.12xlarge` |
| `IMAGE_URI` | empty → newest SageMaker vLLM image |

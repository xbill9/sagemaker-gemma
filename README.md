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

### Every 4-bit size on one L4

The E2B, E4B and 12B QAT rows above, plus the 26B A4B repack and 31B QAT on
the same single L4, 2026-09-28. Dollars per million tokens at 16 parallel use
each instance's on-demand hosting price.

| Measure | E2B QAT | E4B QAT | 12B QAT | 26B A4B W4A16 | 31B QAT* |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instance | `ml.g6.xlarge` | `ml.g6.xlarge` | `ml.g6.xlarge` | `ml.g6.xlarge` | `ml.g6.2xlarge` |
| Weights (GiB) | 8.01 | 11.04 | 8.28 | 15.88 | 18.7 |
| KV cache (tokens) | 867,999 | 200,972 | 41,651 | 14,623 | 2,555 |
| Decode, one request (tokens/s) | 105.1 | 60.7 | 29.3 | 64.8 | 12.6 |
| 16 parallel (tokens/s) | 1077.25 | 599.5 | 358.05 | 515.4 | 39.45 |
| Questions correct (of 40) | 37 | 36 | 39 | 40 | 40 |
| $ per million tokens | 0.29 | 0.52 | 0.87 | 0.61 | 8.60 |

\*31B QAT does not start on an L4 with the standard settings: its 19.77 GiB
of weights exceed vLLM's 0.9 memory cap on the L4's 21.96 GiB. It serves with
a 0.97 cap, at most 4 requests at once, a 1,024-token context, image input
off, and 32 GiB of host RAM (`ml.g6.2xlarge`, the same GPU). Its 16-parallel
rate is set by the 4-request limit.

- The 26B MoE gets all 40 right, decodes faster than E4B QAT and costs less
  per token than 12B QAT, at a 14,623-token KV cache.
- 31B QAT on the L4 decodes at 0.36× its L40S rate and costs 5.2× as much per
  token; it fits, but the L40S is the instance for it.

Details: [`docs/runs/2026-09-28-l4-26b-31b/NOTES.md`](docs/runs/2026-09-28-l4-26b-31b/NOTES.md).

### Recommendation: the biggest responsive model on one L4

Serve the **26B A4B W4A16 repack**
([`xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct))
on `ml.g6.xlarge` ($1.1267/h) with the default settings. "512-token reply" is
the measured median wall time of one request for 512 tokens, aws CLI call
included.

| On one L4 | 512-token reply | Decode (tokens/s) | KV cache (tokens) | Correct (of 40) | $ per million tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2B QAT | 5.4 s | 105.1 | 867,999 | 37 | 0.29 |
| E4B QAT | 9.6 s | 60.7 | 200,972 | 36 | 0.52 |
| 12B QAT | 18.2 s | 29.3 | 41,651 | 39 | 0.87 |
| **26B A4B W4A16** | **8.5 s** | **64.8** | 14,623 | **40** | 0.61 |
| 31B QAT* | 41.2 s | 12.6 | 2,555 | 40 | 8.60 |

- It is the largest model that starts on an L4 with the standard settings. Only
  about 4B of its 26B parameters run per token, so it replies faster than E4B
  QAT and more than twice as fast as 12B QAT, costs less per token than 12B,
  and answered all 40 questions.
- Its limit is the 14,623-token KV cache: 1.79 requests at the full 8,192-token
  context. It suits short prompts and a few users at once; for long documents
  or many concurrent users on one L4, 12B QAT holds 2.85× the cache at 0.45× the
  speed.
- 31B QAT starts only with the reduced settings marked above and replies
  4.9× slower than the 26B, at 14× the cost per token.

### A W4A16 repack against Google's own build: 12B

The 26B repack exists because Google publishes no `-qat-w4a16-ct` for that
size. 12B has both, so the same repack of `-qat-q4_0-unquantized`,
[`xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct),
was set against `google/gemma-4-12B-it-qat-w4a16-ct` on each GPU, standard
settings, 2026-09-29.

| Measure | L4 Google | L4 repack | L40S Google | L40S repack |
| --- | ---: | ---: | ---: | ---: |
| Weights (GiB) | 8.28 | 8.28 | 8.28 | 8.28 |
| KV cache (tokens) | 41,651 | 41,651 | 126,058 | 126,058 |
| Decode, one request (tokens/s) | 29.3 | 29.2 | 77.0 | 78.3 |
| 16 parallel (tokens/s) | 358.05 | 357.7 | 853.4 | 860.8 |
| Questions correct (of 40) | 39 | 40 | 39 | 40 |

Every rate is within 0.99–1.02× of Google's build, and memory and KV cache are
identical. 39 of 40 answers match on both GPUs; on the 40th the repack gives
the answer 12B gives at full size. The repack serves on vLLM 0.30.0 once its
`config.json` carries the fields of Google's `-qat-w4a16-ct` config (vision
token count, top-level text sizes, and a quantization ignore list naming every
layer kept in bf16):
[`docs/runs/2026-09-28-12b-repack/NOTES.md`](docs/runs/2026-09-28-12b-repack/NOTES.md).

### E2B and E4B repacks, and a text-only E2B

The same repack at E2B and E4B, plus an E2B build with the vision and audio
towers removed (`-ct-text`, `Gemma4ForCausalLM`), all on `ml.g6.xlarge` (L4),
2026-09-29, against Google's `-qat-w4a16-ct` measured on the same instance.

| Measure | E2B Google | E2B repack | E2B text-only | E4B Google | E4B repack |
| --- | ---: | ---: | ---: | ---: | ---: |
| Weights (GiB) | 8.01 | 7.26 | 6.33 | 11.04 | 9.79 |
| KV cache (tokens) | 867,999 | 929,454 | 650,783 | 200,972 | 234,879 |
| Decode, one request (tokens/s) | 105.1 | 107.2 | 103.0 | 60.7 | 60.4 |
| Questions correct (of 40) | 37 | 37 | 36 | 36 | 36 |

- Google's E2B and E4B builds store `lm_head.weight` although the config ties
  it to `embed_tokens`; sampled rows are byte-identical. The repacks drop it,
  which is exactly the 0.75 GiB (E2B) and 1.25 GiB (E4B) difference in file
  size and GPU memory, and gives 7 % and 17 % more KV cache.
- Decode speed and scores match Google's builds. Parallel rates are left out of
  this table: they include each run's `aws` CLI overhead, which varied by up to
  0.39 s per call between runs and accounts for the E4B repack's apparent 1.11×.
- The text-only build loads 0.93 GiB less than the full repack, but on a first
  start vLLM measured 5.13 GiB of peak activation for it (0.79 GiB for the full
  repack), so its KV cache is smaller. Every SageMaker endpoint is a first start.
- Both repacks needed the quantization ignore list replaced with Google's (140
  audio-tower layers were missing); that was caught offline before any deploy.

Details: [`docs/runs/2026-09-29-e2b-e4b-repack/NOTES.md`](docs/runs/2026-09-29-e2b-e4b-repack/NOTES.md),
[`docs/runs/2026-09-29-e2b-text/NOTES.md`](docs/runs/2026-09-29-e2b-text/NOTES.md).

### Weight formats on one L4: int4 embeddings, FP8, int8

Text-only builds of the QAT weights on `ml.g6.xlarge`, 2026-09-29, same container and settings
(31B with its reduced settings on `ml.g6.2xlarge`). Decode in tokens/s, one request:

| Size | text-only (int4) | **emb4** (int4 + int4 embeddings, `lm_head`) | FP8 | int8 |
| --- | ---: | ---: | ---: | ---: |
| E2B | 103.0 | **141.7** | 73.0 | 75.5 |
| E4B | 62.1 | **79.9** | 39.4 | 40.1 |
| 12B | 29.0 | **35.0** | 16.8 | 17.0 |
| 26B A4B | 65.8 | **91.5** | – (does not fit) | – |

- int4 embeddings and `lm_head` are 1.21–1.39× faster than text-only at every size, with
  identical answers; packing only the per-layer table (E2B ple4) leaves decode unchanged, so the
  gain is the int4 `lm_head`.
- 8-bit linears run natively on the L4 but decode at 0.58–0.71× of int4: the L4 is bound by memory
  bandwidth, and 8-bit weights are twice the bytes.
- Every build scores 34–40 of 40.

Full table and the SageMaker start-up delay seen on one run:
[`docs/runs/2026-09-29-l4-format-sweep/NOTES.md`](docs/runs/2026-09-29-l4-format-sweep/NOTES.md).

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
| `SM_VLLM_*` | passed to the container as vLLM flags, e.g. `SM_VLLM_MAX_NUM_SEQS=4` |
| `IMAGE_URI` | empty → newest SageMaker vLLM image |

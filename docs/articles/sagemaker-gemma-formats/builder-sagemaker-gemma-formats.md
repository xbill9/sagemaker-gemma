# Gemma 4 on Amazon SageMaker: 4-Bit Embeddings Decode up to 1.39x Faster on One L4

This article repacks Gemma 4's quantization-aware trained (QAT) weights into several 4-bit and 8-bit formats and measures each one on the same Amazon SageMaker NVIDIA L4 endpoint. A suite of Python MCP tools is built to simplify management of the vLLM hosted deployment.

https://github.com/xbill9/sagemaker-gemma

| | |
|---|---|
| Models | Gemma 4 E2B, E4B, 12B, 26B A4B and 31B, from Google's QAT weights |
| Hardware | SageMaker `ml.g6.xlarge`, 1x NVIDIA L4, 24 GB (31B on `ml.g6.2xlarge`) |
| Region | `us-east-2` |
| Software | AWS vLLM SageMaker container, vLLM 0.30.0 |
| Result | 4-bit embeddings and `lm_head` decode **1.12x to 1.39x** faster than the same model with 16-bit embeddings, with identical answers; 8-bit linears decode at **0.58x to 0.71x** of 4-bit |

---

#### Where Do I Start?

This is part three of a series. Part one deploys Gemma 4 to a SageMaker endpoint with the aws CLI and an MCP server: https://dev.to/aws-builders/gemma-4-on-an-amazon-sagemaker-endpoint-aws-cli-nvidia-l4-and-an-mcp-server-2c9d

Part two measures Google's QAT checkpoint against the full-size bf16 release on the same L4: https://dev.to/aws-builders/gemma-4-on-amazon-sagemaker-qat-weights-decode-205x-faster-than-bf16-on-one-l4-318m

This article keeps the endpoint and the measurement from part two and changes how the QAT weights are stored.

---

#### At This Point You Should Have…

- The repository above, with `aws login` done and `mcp` 2.x installed
- A SageMaker quota of at least 1 for `ml.g6.xlarge` in a region with L4 capacity
- The deploy and measure steps from parts one and two working

---

#### Why Repack the QAT Weights?

Google publishes each Gemma 4 QAT model in several forms. The one vLLM loads as 4-bit is `-qat-w4a16-ct`, and it exists for four of the five sizes:

| Size | `-qat-w4a16-ct` | `-qat-q4_0-unquantized` |
| --- | --- | --- |
| E2B, E4B, 12B, 31B |  published |  published |
| 26B A4B |  not published |  published |

`-qat-q4_0-unquantized` stores the QAT weights at 16-bit, but the values already sit on a 4-bit grid: every group of 32 is a scale times an integer from −8 to 7. A repack recovers those integers and writes them as compressed-tensors W4A16, the format vLLM reads. The repack tool verifies every group:

```plaintext
model.language_model.embed_tokens_per_layer.weight: 4.375 GiB bf16 -> 1.230 GiB int4 + f16 scales; 0 off-grid groups; 74.26% of values bit-identical, worst error 6.58e-03 of the group max
```

That opens two things Google's exports leave closed: a W4A16 build of the 26B, and 4-bit **embeddings**. Google's `-w4a16-ct` files keep the embedding tables at 16-bit, and at E2B those tables are three quarters of the file.

---

#### Google's E2B and E4B Store the Output Layer Twice

The config of both models sets `tie_word_embeddings: true`, which means the output layer (`lm_head`) reuses the input embedding table. Google's `-w4a16-ct` files ship a separate `lm_head.weight` anyway. Reading the safetensors headers and seven rows of each table from the Hub:

| | E2B | E4B |
| --- | ---: | ---: |
| `lm_head.weight` | 262,144 x 1,536 bf16 | 262,144 x 2,560 bf16 |
| Bytes | 805,306,368 | 1,342,177,280 |
| Rows byte-identical to `embed_tokens` | 7 of 7 | 7 of 7 |
| GPU weights, Google's build | 8.01 GiB | 11.04 GiB |
| GPU weights, repack without the copy | 7.26 GiB | 9.79 GiB |

vLLM loads the copy, so the repack frees 0.75 GiB and 1.25 GiB of GPU memory at the same decode speed: 107.2 tok/s against 105.1 for E2B, 60.4 against 60.7 for E4B.

---

#### Five Ways to Store the Same Weights

Each build below starts from the same QAT weights and drops the vision and audio towers (text only). They differ in how the linear layers and the embedding tables are stored:

| Build | Linear layers | Embeddings and `lm_head` |
| --- | --- | --- |
| text-only | 4-bit (W4A16) | 16-bit |
| ple4 | 4-bit | 4-bit per-layer table only |
| emb4 | 4-bit | 4-bit, `lm_head` untied |
| FP8 | 8-bit float (W8A8) | 16-bit |
| int8 | 8-bit integer (W8A8) | 16-bit |

The L4 runs both 8-bit formats on its tensor cores natively. `emb4` stores `lm_head` as its own 4-bit table because vLLM ties the output layer by copying the embedding's 16-bit weight, which a packed table does not have. Every build is on Hugging Face under `xbill9/`, listed in the references.

---

####  Tip: The Config Must Name Every 16-Bit Layer

A compressed-tensors config quantizes every `Linear` layer except those on its `ignore` list. A layer the repack leaves at 16-bit but the list does not name makes vLLM look for 4-bit tensors that are not there:

```plaintext
ValueError: There is no module or parameter named 'vision_embedder.patch_dense.weight' in Gemma4UnifiedForConditionalGeneration. The available parameters belonging to vision_embedder.patch_dense (ColumnParallelLinear) are: {'vision_embedder.patch_dense.bias', 'vision_embedder.patch_dense.weight_packed', 'vision_embedder.patch_dense.weight_scale', 'vision_embedder.patch_dense.weight_shape'}
```

Google's own `-w4a16-ct` config for the same size carries the right list: 17 entries for 12B, 250 for E2B and E4B, where the audio tower alone adds 140 layers. Comparing the checkpoint's index against its `ignore` list before deploying finds this in seconds, where a failed SageMaker endpoint takes about 30 minutes to reach `Failed`.

---

#### How the Measurement Works

The measurement is part two's `compare.py`, unchanged: fixed-length replies of 16 and 512 tokens for the decode rate, 1, 4 and 16 requests at once for throughput, and 40 questions with exact answers, all at temperature 0 through `aws sagemaker-runtime invoke-endpoint`.

```shell
python3 compare.py measure docs/runs/2026-09-29-l4-format-sweep gemma-4-e2b-emb4-l4@us-east-2
```

The decode rate cancels the aws CLI start-up and the network; the parallel rates include them. That per-call cost varied from 0.532 s to 1.232 s across runs, so the tables below compare builds by decode rate and quote parallel rates only beside it.

Every build ran on its own `ml.g6.xlarge` in `us-east-2` with `max_model_len` 8192 and 90% GPU memory, one after the other.

---

#### Decode Speed: E2B, Every Combination

| E2B | Embeddings 16-bit | Embeddings 4-bit |
| --- | ---: | ---: |
| **4-bit linears** | 103.0 tok/s |  **141.7 tok/s** |
| **FP8 linears** | 73.0 tok/s | 89.7 tok/s |
| **int8 linears** | 75.5 tok/s | 92.1 tok/s |

Two effects, each visible in every row and column:

- **4-bit embeddings add 17 to 39 tok/s** whatever the linears are.
- **4-bit linears beat 8-bit by 1.36x to 1.58x** whatever the embeddings are.

The emb4 build replies to a 512-token request in 4.154 to 4.18 s, against 5.752 to 6.036 s for text-only.

---

#### Why the Embeddings Matter: `lm_head`

E2B has two large embedding tables: the per-layer table (4.375 GiB) and `embed_tokens`, which `lm_head` shares (0.75 GiB). The ple4 build packs only the per-layer table:

| E2B | Per-layer table | `embed_tokens` and `lm_head` | GPU weights | Decode |
| --- | --- | --- | ---: | ---: |
| text-only | 16-bit | 16-bit | 6.33 GiB | 103.0 tok/s |
| ple4 | 4-bit | 16-bit | 3.19 GiB | 104.1 tok/s |
| emb4 | 4-bit | 4-bit | 2.86 GiB |  141.7 tok/s |

Packing the per-layer table saves 3.14 GiB and leaves decode unchanged: each token reads one row of it. Packing `lm_head` gives the 1.36x: every generated token multiplies against all 262,144 rows of the output layer, so its size is read once per token.

---

#### Across All Five Sizes

| Size | text-only | emb4 | emb4 / text-only |
| --- | ---: | ---: | ---: |
| E2B | 103.0 |  141.7 | 1.38 |
| E4B | 62.1 |  79.9 | 1.29 |
| 12B | 29.0 |  35.0 | 1.21 |
| 26B A4B | 65.8 |  91.5 | 1.39 |
| 31B | 12.6 |  14.1 | 1.12 |

Decode in tok/s. The gain follows the output layer's share of each token's reads. It is largest for the small models and for the 26B, which is a mixture of experts that runs about 4B parameters per token, and smallest for the dense 31B. Every emb4 build gives the same 40 answers as its text-only twin, character for character.

31B runs on `ml.g6.2xlarge` (the same L4, 32 GiB of host memory) with 97% GPU memory, 4 requests at once and a 1,024-token context. Those are the settings under which it starts on an L4 at all.

---

#### What About FP8 and Int8?

| Size | 4-bit (text-only) | FP8 | int8 | FP8 / 4-bit |
| --- | ---: | ---: | ---: | ---: |
| E2B |  103.0 | 73.0 | 75.5 | 0.71 |
| E4B |  62.1 | 39.4 | 40.1 | 0.63 |
| 12B |  29.0 | 16.8 | 17.0 | 0.58 |

Decode in tok/s. Generating a token on the L4 is limited by how fast it reads weights from memory, and an 8-bit weight is twice the bytes of a 4-bit one. The native FP8 and int8 tensor cores do not change that, and the penalty grows with model size. FP8 and int8 land within 3% of each other at every size. At 26B and 31B the 8-bit weights, about 26 GB and 31 GB, do not fit on one L4.

---

#### Does It Still Answer Correctly?

| Size | text-only | emb4 | FP8 | int8 |
| --- | ---: | ---: | ---: | ---: |
| E2B | 36 | 36 | 35 | 37 |
| E4B | 36 | 36 | 36 | 38 |
| 12B | 40 | 40 | 40 | 40 |
| 26B A4B | 40 | 40 | – | – |
| 31B | 40 | 40 | – | – |

Correct out of 40. Every build scores within two questions of its size's text-only build, and emb4 matches text-only answer for answer at every size. Forty questions show a large loss and cannot resolve a one- or two-question difference.

---

#### What the Engine Allocates

Smaller weights leave more of the L4 for the KV cache:

| Size | text-only KV (tokens) | emb4 KV (tokens) |
| --- | ---: | ---: |
| E2B | 650,783 |  1,209,977 |
| E4B | 119,416 |  376,156 |
| 12B | 41,654 |  52,541 |
| 26B A4B | 19,183 |  30,456 |
| 31B | 2,555 |  3,241 |

At E2B and E4B the text-only builds lose more than their smaller weights would predict. vLLM sizes the cache after a start-up profiling run and counts the compiler's memory as activation on a first start: the E2B text-only build measured 5.13 GiB of peak activation against 0.79 GiB for the same model with its towers. Every SageMaker endpoint starts cold, so it always gets the smaller figure. The emb4 builds measured 1.79 GiB and keep the larger cache.

---

#### Compare to Other Deployments

The same Google E2B QAT checkpoint served by the same vLLM version on a plain EC2 `g6.xlarge`, measured on the instance itself, against the SageMaker endpoint:

| | SageMaker `ml.g6.xlarge` | EC2 `g6.xlarge` |
| --- | ---: | ---: |
| GPU weights | 8.01 GiB | 8.01 GiB |
| Decode | 105.1 tok/s | 105.1 tok/s |
| Identical answers | – | 40 of 40 |
| Per-call client cost | 0.562 s | 0.008 s |

The model server is the same on both. What differs is the path to it: SageMaker's figure includes signing and sending each request through the aws CLI, while the EC2 client ran on the instance.

---

#### And Price/Performance?

`ml.g6.xlarge` is $1.1267 an hour in `us-east-2`. Per million output tokens at 16 requests at once (arithmetic):

| Size | text-only | emb4 |
| --- | ---: | ---: |
| E2B | $0.338 |  $0.249 |
| E4B | $0.699 |  $0.369 |
| 12B | $0.873 |  $0.76 |
| 26B A4B | $0.709 |  $0.525 |

FP8 and int8 cost more per token than either at E2B: $0.386 and $0.379. The endpoint bills while it exists, busy or idle, so these figures hold only while it is kept busy.

---

####  Tip: `InService` Can Lag a Healthy Container

One endpoint in this series stayed `Creating` for 3 hours 26 minutes. Its CloudWatch log shows vLLM serving from 00:18:29 UTC and answering SageMaker's health check with 200 every 5 seconds:

```plaintext
00:18:39  (APIServer pid=14) INFO: 169.254.178.2:60670 - "GET /ping HTTP/1.1" 200 OK
...
03:35:58  (APIServer pid=14) INFO: 169.254.178.2:47052 - "GET /ping HTTP/1.1" 200 OK
```

SageMaker moved it to `InService` at 03:35:58, 3.29 hours after the container was healthy, and billed the instance throughout. The other 20 endpoints in the same sweep reached `InService` within 11.3 minutes. A deploy script that alerts on time spent in `Creating` catches this; a `Creating` endpoint refuses deletion until it leaves that state.

---

#### So, Which One?

| On one L4 | 4-bit + 16-bit embeddings | 4-bit + 4-bit embeddings | FP8 or int8 |
| --- | --- | --- | --- |
| Decode speed | baseline |  1.12x to 1.39x | 0.58x to 0.71x |
| KV cache | baseline |  larger at every size | smaller |
| Answers (40) | reference |  identical | within 2 |
| Fits 26B and 31B | yes |  yes | no |

On a SageMaker L4, serve Gemma 4 with 4-bit linears and 4-bit embeddings: the emb4 build is the fastest and smallest at every size and gives the same answers. FP8 and int8 trail it on every measure taken here; long prompts, where arithmetic sets the pace, were not measured.

---

#### What Stops the Meter

Each run's script ends with `delete_endpoint`, which removes the endpoint, its config and its model, and a watchdog repeats that for every endpoint name in all three regions:

```plaintext
2026-09-30T04:00:05Z us-east-1 gemma-4-e2b-fp8emb4-l4: {  "endpoint": "gemma-4-e2b-fp8emb4-l4",  "results": {    "endpoint": "not found",    "endpoint-config": "not found",    "model": "not found"  }}
```

---

#### Summary

The goal of this article was to measure how the storage format of Gemma 4's QAT weights changes serving speed on a SageMaker L4 endpoint. The key to the solution was repacking the same QAT weights several ways and changing only `SM_VLLM_MODEL` between deployments. The measured results were:

- 4-bit embeddings and `lm_head` decode **1.12x to 1.39x** faster than 16-bit embeddings, E2B to 31B, with identical answers
- The speed comes from `lm_head`: packing only the per-layer table saves 3.14 GiB at E2B and leaves decode unchanged
- Google's E2B and E4B QAT files store `lm_head` twice; dropping the copy frees 0.75 GiB and 1.25 GiB
- A W4A16 build of the 26B serves at **91.5 tok/s** with 4-bit embeddings, where Google publishes none
- FP8 and int8 decode at **0.58x to 0.71x** of 4-bit on the L4, and do not fit at 26B and 31B
- A repack's `ignore` list must name every 16-bit layer, or vLLM fails at load
- One endpoint stayed `Creating` for 3.29 hours while its container passed every health check

Scope: one account, SageMaker `ml.g6.xlarge` (31B on `ml.g6.2xlarge` with reduced settings) in `us-east-2`, vLLM 0.30.0 from the AWS container, one deployment per build on 2026-09-29, each on its own instance; Google's E2B and E4B QAT baselines were measured on 2026-09-25 and 2026-09-27. Prompts were short and long-prompt behaviour was not measured. Decode used five replies per length, parallel throughput two batches per level, and quality 40 questions at temperature 0, all through the aws CLI from one client machine.

The strategy for using MCP for SageMaker deployment and benchmarking was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/sagemaker-gemma
- Part one, deploying Gemma 4 to SageMaker: https://dev.to/aws-builders/gemma-4-on-an-amazon-sagemaker-endpoint-aws-cli-nvidia-l4-and-an-mcp-server-2c9d
- Part two, QAT against bf16: https://dev.to/aws-builders/gemma-4-on-amazon-sagemaker-qat-weights-decode-205x-faster-than-bf16-on-one-l4-318m
- E2B emb4: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4
- E4B emb4: https://huggingface.co/xbill9/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-emb4
- 12B emb4: https://huggingface.co/xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct-text-emb4
- 26B A4B emb4: https://huggingface.co/xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text-emb4
- 31B emb4: https://huggingface.co/xbill9/gemma-4-31B-it-qat-q4_0-w4a16-ct-text-emb4
- E2B FP8: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-fp8-text
- E2B int8: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-w8a8-int8
- Google Gemma 4 E2B QAT: https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct
- SageMaker real-time inference: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html
- compressed-tensors: https://github.com/neuralmagic/compressed-tensors

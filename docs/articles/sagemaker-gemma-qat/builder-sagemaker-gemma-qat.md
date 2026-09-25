# Gemma 4 on Amazon SageMaker: QAT Weights Decode 2.05x Faster Than bf16 on One L4

This article gives a short background on Amazon SageMaker real-time endpoints, then measures Gemma 4 E2B's quantization-aware trained (QAT) checkpoint against the full-size bf16 release on the same NVIDIA L4 endpoint. A suite of Python MCP tools is built to simplify management of the vLLM hosted deployment.

https://github.com/xbill9/sagemaker-gemma

| | |
|---|---|
| Models | `google/gemma-4-E2B-it` (bf16) and `google/gemma-4-E2B-it-qat-w4a16-ct` (QAT, 4-bit weights) |
| Hardware | SageMaker `ml.g6.xlarge`, 1x NVIDIA L4, 24 GB |
| Region | `us-east-2` |
| Software | AWS vLLM SageMaker container, vLLM 0.30.0 |
| Result | QAT decodes at **105.1 tok/s** against **51.3**, and serves **1077.25 tok/s** at 16 parallel requests against **619.1**, with the same score on 40 checked questions |

---

#### SageMaker Hosting in Five Minutes

SageMaker real-time inference is three objects, created in order:

| Object | What it holds |
| --- | --- |
| Model | A container image, its environment variables, and an IAM role |
| Endpoint config | Which model runs on which instance types, and how many instances |
| Endpoint | The running HTTPS service, billed per instance-hour while it exists |

Requests go through `aws sagemaker-runtime invoke-endpoint`, signed with your AWS credentials. SageMaker health-checks the container, routes traffic to it and writes its log to CloudWatch under `/aws/sagemaker/Endpoints/<name>`.

The container here is the vLLM build AWS publishes for SageMaker. It reads vLLM settings from `SM_VLLM_` environment variables, downloads the model from Hugging Face at start-up, and accepts OpenAI-style chat bodies. Switching checkpoints is one variable: `SM_VLLM_MODEL`.

SageMaker JumpStart also lists Gemma 4 as ready-made packages:

```shell
aws sagemaker list-hub-contents --hub-name SageMakerPublicHub --hub-content-type Model \
  --query "HubContentSummaries[].HubContentName" --output text | tr '\t' '\n' | grep gemma-4
```

```plaintext
huggingface-llm-gemma-4-31b-it-nvfp4
huggingface-vlm-gemma-4-12b-it
huggingface-vlm-gemma-4-26b-a4b-it
huggingface-vlm-gemma-4-31b-it
huggingface-vlm-gemma-4-31b-it-fp8-block
huggingface-vlm-gemma-4-e2b-instruct
huggingface-vlm-gemma-4-e4b-it
```

None of the seven is a Google QAT checkpoint, so this comparison uses the vLLM container with a Hugging Face model ID.

Three account limits shape every deployment:

- **Quota** is per instance type, per region, counted in instances. This account holds 1 for each single-L4 type in `us-east-1`, `us-east-2` and `us-west-2`.
- **Capacity** is separate. `us-east-1` and `us-west-2` each left an L4 request in `Creating` for about 30 minutes and then returned `InsufficientInstanceCapacity`. `us-east-2` placed one within minutes every time it was asked.
- **A fallback list** (`InstancePools` in the endpoint config) holds quota for every type in it, so two endpoints cannot share one region's single-L4 quota.

With one L4 per region to work with, the two checkpoints ran one after the other on the same instance type in the same region.

---

#### Where Do I Start?

The deployment itself, from quota check to teardown with the aws CLI and the MCP server, is the first article in this pair: https://github.com/xbill9/sagemaker-gemma/blob/main/docs/articles/sagemaker-gemma-deploy/devto-sagemaker-gemma-deploy.md

This article starts from a working endpoint and changes one thing: the checkpoint.

---

#### At This Point You Should Have…

- The repository above, with `aws login` done and `mcp` 2.x installed
- A SageMaker quota of at least 1 for `ml.g6.xlarge` in a region with L4 capacity
- The full-size endpoint from the first article deployed as `gemma-4-e2b`

---

#### What QAT Changes

Google trains the QAT checkpoint with 4-bit weights in the loop, then exports it in the `compressed-tensors` format vLLM reads natively. The `-w4a16-ct` suffix means 4-bit weights, 16-bit activations. Google publishes the same model in four QAT forms; only this one loads in vLLM:

| Checkpoint | vLLM |
| --- | --- |
| `-qat-w4a16-ct` |  loads, 4-bit weights |
| `-qat-q4_0-unquantized` |  stored at 16-bit, no memory saving |
| `-qat-q4_0-gguf` |  GGUF, for llama.cpp and Ollama |
| `-qat-mobile-*` |  on-device formats |

Deploying it is the first article's Steps 4 to 6 with two variables changed:

```shell
export NAME=gemma-4-e2b-qat
export MODEL_ID=google/gemma-4-E2B-it-qat-w4a16-ct
```

vLLM names the format when it starts:

```plaintext
quantization=compressed-tensors
```

---

#### What the Engine Allocates

The container log records the weights, the load time and the KV cache vLLM builds from what is left of the L4's memory:

```plaintext
Model loading took 9.75 GiB memory and 82.751806 seconds
GPU KV cache size: 723,484 tokens, Maximum concurrency for 8,192 tokens per request: 88.32x
```

```plaintext
Model loading took 8.01 GiB memory and 66.190265 seconds
GPU KV cache size: 867,999 tokens, Maximum concurrency for 8,192 tokens per request: 105.96x
```

| | bf16 | QAT | QAT / bf16 |
| --- | ---: | ---: | ---: |
| Weights (GiB) | 9.75 | 8.01 | 0.82 |
| KV cache (tokens) | 723,484 | 867,999 | 1.2 |
| Weight load (s) | 82.75 | 66.19 | |
| Create to `InService` (min) | 9.9 | 10.1 | |

The 4-bit weights save 18% of GPU memory. The checkpoint's own header shows why: only the transformer body is 4-bit.

| Part of the QAT file | GB | Share | Stored as |
| --- | ---: | ---: | --- |
| Per-layer embedding | 4.698 | 56.5% | BF16 |
| Vocabulary embedding | 1.611 | 19.4% | BF16 |
| Transformer body | 1.056 | 12.7% | packed 4-bit |
| Audio tower | 0.614 | 7.4% | BF16 |
| Vision tower | 0.337 | 4.1% | BF16 |

The memory the smaller weights free goes to the KV cache.

---

#### How the Measurement Works

`compare.py` runs the same three measurements against each endpoint, at temperature 0, through the same `aws sagemaker-runtime invoke-endpoint` call:

1. **Decode.** Fixed-length replies of 16 and 512 tokens (`ignore_eos`), five of each. The decode rate is (512 − 16) / (median time at 512 − median time at 16), which cancels the aws CLI start-up and the network round trip.
2. **Parallel requests.** 1, 4 and 16 requests at once, 256 tokens each, two batches per level. Throughput is total output tokens divided by the batch's wall time.
3. **Answers.** 40 fixed questions with exact answers: 15 two-digit multiplications, 15 three-number sums, 10 capitals. Scored by regular expression.

```shell
python3 compare.py measure docs/runs/2026-09-25-qat-vs-bf16 gemma-4-e2b-qat@us-east-2
```

```plaintext
wrote docs/runs/2026-09-25-qat-vs-bf16/measure-gemma-4-e2b-qat.json
{
  "decode_tokens_per_second": 105.1,
  "per_call_fixed_cost_seconds": 0.562,
  "load_tokens_per_second": {
    "1": 85.35,
    "4": 328.7,
    "16": 1077.25
  },
  "quality": "37/40"
}
```

`combine` computes every ratio from the two result files:

```shell
python3 compare.py combine compare.json measure-gemma-4-e2b.json measure-gemma-4-e2b-qat.json
```

```plaintext
                               gemma-4-e2b     gemma-4-e2b-qat  ratio
weights_gib                           9.75                8.01  0.82
kv_cache_tokens                     723484              867999  1.2
load_seconds                     82.751806           66.190265
decode_tokens_per_second              51.3               105.1  2.05
load_c1_tokens_per_second             45.6               85.35  1.87
load_c4_tokens_per_second            171.6               328.7  1.92
load_c16_tokens_per_second           619.1             1077.25  1.74
quality_correct                         37                  37
identical answers: 35/40
```

---

#### Decode Speed

| | bf16 | QAT |
| --- | ---: | ---: |
| Decode (tokens/s) | 51.3 |  105.1 |
| 512-token reply, fastest (s) | 10.491 |  5.392 |
| 512-token reply, slowest (s) | 10.661 |  5.526 |
| Per-call client cost (s) | 0.633 | 0.562 |

**QAT decodes 2.05x faster.** The slowest QAT reply finished in about half the time of the fastest bf16 one, so the gap is far larger than the spread between repeats.

Each decode step reads every transformer layer's weights from GPU memory, so decode speed follows how many bytes those layers take. Those are the layers the QAT export stores at 4 bits. The embeddings are looked up one row per token, so their 16-bit size costs memory and little time.

---

#### Parallel Requests

| Requests at once | bf16 tok/s | QAT tok/s | QAT / bf16 |
| ---: | ---: | ---: | ---: |
| 1 | 45.6 |  85.35 | 1.87 |
| 4 | 171.6 |  328.7 | 1.92 |
| 16 | 619.1 |  1077.25 | 1.74 |

QAT leads at every level. The ratio narrows at 16, where a batch shares each weight read across more requests and the per-token saving counts for less. The single-request figures sit below the decode rate because each call also pays the aws CLI start-up.

---

#### Does It Still Answer Correctly?

| Questions | bf16 | QAT |
| --- | ---: | ---: |
| Multiplication (15) | 15 | 15 |
| a + b − c (15) | 12 | 12 |
| Capitals (10) | 10 | 10 |
| **Total (40)** | **37** | **37** |

35 of the 40 answers are identical character for character. The other five are all sums, and each model misses three of them:

| Question | Expected | bf16 | QAT |
| --- | ---: | ---: | ---: |
| 876 + 608 − 558 | 926 | 1026 (wrong) | 926 |
| 257 + 388 − 290 | 355 | 355 | 655 (wrong) |
| 782 + 571 − 534 | 819 | 829 (wrong) | 819 |
| 470 + 238 − 983 | −275 | −275 | −285 (wrong) |
| 150 + 937 − 139 | 948 | 918 (wrong) | 1010 (wrong) |

The two models make the same number of mistakes on different questions. Forty questions are enough to show a large loss and too few to measure a small one.

---

#### Re-Measured in A-B-A Order

Each endpoint runs on its own physical instance, and the two ran 15 minutes apart. To check that the gap belongs to the checkpoint, the bf16 endpoint was deployed a second time after QAT and measured again:

| Run | Start (UTC) | Decode tok/s | 16 at once tok/s | Correct |
| --- | --- | ---: | ---: | ---: |
| bf16 | 18:36 | 51.3 | 619.1 | 37 |
| QAT | 18:50 | 105.1 | 1077.25 | 37 |
| bf16 again | 19:10 | 51.5 | 625.15 | 37 |

The two bf16 runs agree within 1.21% on every speed figure and give identical answers to all 40 questions. The second bf16 instance loaded its weights in 82.09 s against 82.75 s the first time.

---

#### Compare to Other Deployments

The same bf16 model on the same GPU has been measured on two other platforms in this series, with `vllm bench serve` at 128 output tokens:

| Platform | 1 request tok/s |
| --- | ---: |
| Cloud Run, NVIDIA L4 | 49.63 |
| EC2 `g6.2xlarge`, NVIDIA L4 | 46.09 |
| SageMaker `ml.g6.xlarge`, NVIDIA L4 | 45.6 |

The three sit within 9% of each other. SageMaker's figure includes the aws CLI start-up in every call; its decode rate with that removed is 51.3.

On a Tesla T4, QAT decoded 1.79x faster than bf16 for the same model. On the L4 the ratio is 2.05x.

The methods differ: the other runs used `vllm bench serve` with random prompts, other vLLM versions and their own hosts. Read the rows as a shape.

---

#### And Price/Performance?

`ml.g6.xlarge` in `us-east-2` is $1.1267 an hour on demand:

```shell
aws pricing get-products --region us-east-1 --service-code AmazonSageMaker \
  --filters Type=TERM_MATCH,Field=instanceName,Value=ml.g6.xlarge Type=TERM_MATCH,Field=regionCode,Value=us-east-2
```

```plaintext
ml.g6.xlarge USE2-Host:ml.g6.xlarge 1.1267000000 Hrs | $1.1267 per Hosting ml.g6.xlarge hour in US East (Ohio)
```

The same hourly price buys twice the tokens. Per million output tokens (arithmetic):

| Requests at once | bf16 $/M | QAT $/M |
| ---: | ---: | ---: |
| 1 | 6.86 |  3.67 |
| 4 | 1.82 |  0.95 |
| 16 | 0.51 |  0.29 |

The endpoint bills while it exists, busy or idle, so these figures hold only while it is kept busy.

---

#### So, Which One?

| | bf16 | QAT |
| --- | --- | --- |
| Decode speed | 51.3 tok/s |  105.1 tok/s |
| 16 requests at once | 619.1 tok/s |  1077.25 tok/s |
| GPU memory for weights | 9.75 GiB |  8.01 GiB |
| Checked answers | 37 / 40 | 37 / 40 |
| Cost per million tokens at 16 | $0.51 |  $0.29 |

On an L4 SageMaker endpoint, the QAT checkpoint is the default choice for Gemma 4 E2B: the same instance, one changed environment variable, about twice the tokens per dollar, and no measured change in answers. Keep bf16 as the reference when a task's accuracy needs a larger evaluation than 40 questions.

---

#### What Stops the Meter

`delete_endpoint` removes the endpoint, its config and its model:

```plaintext
`gemma-4-e2b-qat` in `us-east-2`
- endpoint: deleted
- endpoint-config: deleted
- model: deleted
```

A `Creating` endpoint refuses deletion, so a deploy runs until it reaches `InService` or `Failed` before it can be removed.

---

#### Summary

The goal of this article was to measure what Gemma 4 E2B's QAT checkpoint changes on a SageMaker L4 endpoint. The key to the solution was changing only `SM_VLLM_MODEL` between two deployments on the same instance type, and measuring decode speed with the per-call client cost removed. The measured results were:

- QAT decodes at **105.1 tok/s** against bf16's **51.3**, 2.05x
- At 16 requests at once QAT serves **1077.25 tok/s** against **619.1**, 1.74x
- Both score **37 of 40** on checked questions, with 35 identical answers
- A second bf16 deployment after QAT reproduced the first within 1.21%
- The 4-bit export saves 18% of weight memory, because the embeddings stay at 16-bit
- Getting an L4 took three regions: `us-east-1` and `us-west-2` returned `InsufficientInstanceCapacity` after about 30 minutes each

Scope: one account, SageMaker `ml.g6.xlarge` with one NVIDIA L4 in `us-east-2`, vLLM 0.30.0 from the AWS container, three deployments on 2026-09-25 each on its own instance, measured in bf16, QAT, bf16 order. Prompts were short; long-prompt behaviour was not measured. Decode used five replies per length, parallel throughput two batches per level, and quality 40 questions at temperature 0. Every request went through the aws CLI from one client machine.

The strategy for using MCP for SageMaker deployment and benchmarking was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/sagemaker-gemma
- Part one, deploying Gemma 4 to SageMaker: https://github.com/xbill9/sagemaker-gemma/blob/main/docs/articles/sagemaker-gemma-deploy/devto-sagemaker-gemma-deploy.md
- Gemma 4 E2B QAT w4a16: https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct
- Gemma 4 E2B: https://huggingface.co/google/gemma-4-E2B-it
- Gemma 4 on a Tesla T4, QAT vs bf16: https://dev.to/gde/gemma-4-on-a-tesla-t4-qat-weights-decode-179x-faster-than-bf16-2fi4
- 2B Gemma 4 on Cloud Run with an NVIDIA L4: https://dev.to/gde/2b-gemma-4-deployment-with-cloud-run-nvidia-l4-mcp-sdk-2x-and-claude-code-4ml3
- SageMaker real-time inference: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html
- AWS Deep Learning Containers: https://github.com/aws/deep-learning-containers
- compressed-tensors: https://github.com/neuralmagic/compressed-tensors

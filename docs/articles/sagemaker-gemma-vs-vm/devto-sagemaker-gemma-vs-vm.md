---
title: "Gemma 4 on Amazon SageMaker or a VM? The Same Model Server at 1.40x the Price"
published: false
description: "The same Gemma 4 build, vLLM version and GPU served from a SageMaker endpoint and from a plain EC2 instance, on a T4 and an L4: identical decode and answers, a different call path, and what the managed endpoint's 1.40x buys."
tags: aws, sagemaker, gemma, vllm
cover_image: https://raw.githubusercontent.com/xbill9/sagemaker-gemma/main/docs/articles/sagemaker-gemma-vs-vm/devto-cover.9bcf54b6.jpg
---

This article serves the same Gemma 4 build from an Amazon SageMaker endpoint and from a plain EC2 instance with the same GPU, on an NVIDIA T4 and an NVIDIA L4, and prices both against Compute Engine and Cloud Run. A suite of Python MCP tools is built to simplify management of the vLLM hosted deployment.

https://github.com/xbill9/sagemaker-gemma

| | |
|---|---|
| Model | Gemma 4 E2B, 4-bit weights (4-bit embeddings on the T4) |
| Hardware | 1x NVIDIA T4 (`ml.g4dn.xlarge` / `g4dn.xlarge`) and 1x NVIDIA L4 (`ml.g6.xlarge` / `g6.xlarge`) |
| Region | `us-east-2` |
| Software | vLLM 0.30.0 on both sides |
| Result | Decode and answers match within 2%. EC2 costs **0.71x** per hour, and its client, on the instance, pays **0.006 s** per call against **0.56 s** through the aws CLI |

---

#### Where Do I Start?

This is part five of a series. Part one deploys Gemma 4 to a SageMaker endpoint with the aws CLI and an MCP server: https://dev.to/aws-builders/gemma-4-on-an-amazon-sagemaker-endpoint-aws-cli-nvidia-l4-and-an-mcp-server-2c9d

Part two measures Google's QAT checkpoint against the full-size bf16 release: https://dev.to/aws-builders/gemma-4-on-amazon-sagemaker-qat-weights-decode-205x-faster-than-bf16-on-one-l4-318m

Part three repacks the QAT weights with 4-bit embeddings: https://dev.to/aws-builders/gemma-4-on-amazon-sagemaker-4-bit-embeddings-decode-up-to-139x-faster-on-one-l4-36mf

Part four serves those builds on SageMaker's smallest GPU, the T4: https://github.com/xbill9/sagemaker-gemma/blob/main/docs/articles/sagemaker-gemma-t4/devto-sagemaker-gemma-t4.md

This article asks what the managed endpoint adds over the same GPU without it.

---

#### At This Point You Should Have…

- The repository cloned and `make test` passing
- An `aws login` session with SageMaker endpoint quota and EC2 on-demand G-family vCPU quota in `us-east-2`
- Part four's T4 endpoint measurements, which this article compares against

---

#### What Is Being Compared?

Two ways to serve one model on one GPU:

| | SageMaker endpoint | EC2 instance |
|---|---|---|
| Provisioning | `create-endpoint` | `run-instances` with cloud-init |
| vLLM | AWS SageMaker container 0.30.0 | `vllm/vllm-openai:v0.30.0` |
| Turing patch on the T4 | derived image, built by CodeBuild | derived image, built on the instance |
| Access | IAM-signed `invoke-endpoint` | Systems Manager only, no inbound rules |
| Client | aws CLI on a workstation | `compare.py` on the instance, to localhost |

The model, context length (8,192), memory setting (0.90), data type and measurement script are the same on both sides. The client location differs, and the per-call figures below include it.

---

#### Step 1 — Serve the Same Model on EC2

The EC2 side is a rig in the author's `gemma4-dev` tree, `gpu-vllm-g4dn-2b-w4a16`, whose settings match the SageMaker endpoint:

```
INSTANCE_TYPE=g4dn.xlarge
VLLM_IMAGE=vllm/vllm-openai:v0.30.0
VLLM_PATCHED_IMAGE=vllm-openai:v0.30.0-sm75-patched
DTYPE=float16
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.90
MODEL=xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4
```

Cloud-init pulls the stock image, applies the same Turing patch as part four and builds the derived tag on the instance, in seconds:

```
[stage] image-pull-done +170s
[stage] patch-applied +191s
[stage] image-build-done +194s
[stage] patch-verified-in-image +209s
[stage] serving-started +211s
```

The instance served 9.7 minutes after launch. Its security group has no inbound rules; Systems Manager is the only way in.

---

#### Step 2 — Measure on the Instance

`ec2_measure.py` copies the project's `compare.py` onto the instance over Systems Manager and runs it against `localhost:8000`. It measures decode speed, 1 to 16 parallel requests and 40 questions at temperature 0, the same measurement every SageMaker endpoint in the series got. The instance is terminated in a `finally` block, and a watchdog terminates it if the driver dies.

---

#### T4: SageMaker Against EC2

`compare.py combine` sets the two runs side by side:

```
                            gemma-4-e2b-emb4-t4  gemma-4-e2b-emb4-g4dn  ratio
weights_gib                           2.86                2.86  1.0
kv_cache_tokens                     660033              660108  1.0
decode_tokens_per_second             108.5               111.2  1.02
load_c1_tokens_per_second             91.1              113.85  1.25
load_c4_tokens_per_second           308.25               394.9  1.28
load_c16_tokens_per_second          787.05              1005.1  1.28
quality_correct                         36                  36
identical answers: 40/40
```

The GPU does the same work on both: the same memory allocation, decode within 2% and all 40 answers byte-identical.

---

#### L4: SageMaker Against EC2

The same comparison on the L4, with Google's E2B QAT checkpoint and the stock vLLM 0.30.0 image:

| Measure | SageMaker `ml.g6.xlarge` | EC2 `g6.xlarge` | EC2 / SageMaker |
|---|---:|---:|---:|
| Decode, one request (tok/s) | 105.1 | 105.1 | 1.00 |
| 1 request, 256 tokens (tok/s) | 85.35 | 104.85 | 1.23 |
| 16 parallel (tok/s) | 1077.25 | 1485.9 | 1.38 |
| Per-call fixed cost (s) | 0.562 | 0.008 | – |
| Identical answers | 40 of 40 | | |

Two GPUs, two checkpoints, one result: the model server runs at the same speed on SageMaker and on EC2.

---

#### Where Does the Time Go?

`compare.py` fits each request's wall time as a fixed cost plus a per-token rate. The per-token rate is the decode speed, and it matches. The fixed cost is 0.56 s per call through SageMaker and under 0.01 s on the instance.

That 0.56 s covers everything between the client and vLLM: starting the aws CLI, signing the request, the network round trip from the workstation and the SageMaker front end. The EC2 figure has none of the first three, because the client runs beside vLLM. A client off the instance would pay its own network cost, and a long-lived SDK client would skip the CLI start-up; neither was measured here. The fixed cost per call is what lowers SageMaker's rate at 1 to 16 parallel requests, by 1.23x to 1.38x.

---

#### And Price/Performance?

On-demand list prices, from the AWS Price List API and the Google Cloud billing catalog on 2026-09-30:

| Option | T4 ($/h) | L4 ($/h) |
|---|---:|---:|
| SageMaker endpoint | 0.736 | 1.1267 |
| EC2 | 0.526 | 0.8048 |
| Compute Engine VM | 0.5241 | 0.7045 |
| Cloud Run, per running hour | – | 1.4209 |

SageMaker costs 1.40x EC2 per hour on both GPUs. The Compute Engine T4 is an `n1-standard-2` in `us-west2`; the L4 is a `g2-standard-4` in `us-east4`. Cloud Run is one L4 with 8 vCPU and 32 GiB in `us-east4`, instance-based billing, no zonal redundancy.

Per million output tokens at 16 parallel requests, each platform at its own measured rate:

| GPU | SageMaker | EC2 | EC2 / SageMaker |
|---|---:|---:|---:|
| T4 | $0.260 | $0.145 | 0.56 |
| L4 | $0.291 | $0.150 | 0.52 |

At equal throughput, the price alone makes EC2 0.71x. The rest comes from the faster call path on the instance.

---

#### 🔎 Tip: Cloud Run Bills Only While It Runs

Cloud Run's L4 costs 1.67x a Compute Engine `g2-standard-8` of the same size for every hour it runs, and with `--min-instances=0` it runs only while there is traffic. By arithmetic on the list prices, it is the cheaper of the two when the instance is up less than 60% of the day, about 14.4 hours. Each start after a quiet period loads the model again, which takes minutes for Gemma 4, so scale-to-zero suits a model that is called in bursts. Throughput on Compute Engine and Cloud Run was not measured here.

---

#### What Does the Endpoint Buy?

- 🟢 **vLLM from environment variables.** `SM_VLLM_MODEL`, `SM_VLLM_MAX_MODEL_LEN` and the rest become vLLM flags, on a container AWS maintains. On an L4 or newer GPU there is nothing to build.
- 🟢 **Access by IAM.** No open port and no security group; every call is a signed `invoke-endpoint`.
- 🟢 **Fallback when capacity runs out.** `InstancePools` tries the next instance type of the same GPU when the first has no capacity.
- 🟢 **Logs and metrics with no setup.** Every endpoint writes its container log to CloudWatch.
- 🟢 **Production features:** autoscaling, several instances behind one name, traffic splits and updates without downtime.
- ⚠️ **A slower start.** 13.3 minutes from the deploy call to `InService` on the T4, against 9.7 minutes to serving on EC2 with the image built on the instance.
- ⚠️ **An older GPU takes more work.** The T4 needed a derived image built in CodeBuild and a host image with driver 580; on EC2 the same patch ran in cloud-init.
- ❌ **1.40x the hourly price** of the same EC2 instance, whether the endpoint is busy or idle.

---

#### So, Which One?

| Stage | Where | Why |
|---|---|---|
| 🥇 Finding a working vLLM configuration | SageMaker endpoint | Maintained container, flags as settings, logs and capacity fallback |
| 🥇 Serving a small model after that | EC2 or Compute Engine | Same model server at 0.71x the hourly price |
| 🥈 Bursty traffic on Google Cloud | Cloud Run | Scales to zero |
| 🥇 Traffic that needs scaling and safe updates | SageMaker endpoint | The features the 1.40x pays for |

The endpoint is a quick way to find a vLLM configuration that works, with the logs and fallback to debug it. Once the settings are known, a model that fits one GPU serves at the same speed from a VM for 0.71x the price.

---

#### What Stops the Meter

The EC2 instance was terminated when the measurement finished:

```
2026-09-30T16:21:21Z measured
2026-09-30T16:28:11Z i-08bf5db2ef4c05ffb terminated
```

The SageMaker endpoints were deleted after their measurements, as in part four.

---

#### Summary

The goal of this article was to find what a SageMaker endpoint adds over the same GPU without it. The key to the solution was serving the same build with the same vLLM version both ways and measuring both with one script. The results were:

- 🟢 The model server runs at the same speed on SageMaker and EC2: decode within **2%**, 40 of 40 answers identical, on a T4 and an L4
- 🟢 EC2 costs **0.71x** SageMaker per hour on both GPUs; Compute Engine is within 1% of EC2 for the T4 and 0.875x for the L4
- ❌ Through the aws CLI, each SageMaker call carries **0.56 s** of fixed cost, and parallel throughput is **1.23x to 1.38x** lower than the on-instance client measured
- ⚠️ SageMaker's 1.40x buys managed provisioning, IAM access, capacity fallback, logs and production scaling
- ⚠️ Cloud Run's L4 costs 1.67x a Compute Engine VM per running hour, and nothing when scaled to zero

Scope: one account, `us-east-2`, one deployment of each on 2026-09-29 (L4) and 2026-09-30 (T4), vLLM 0.30.0 throughout. The SageMaker client was the aws CLI on a workstation, and the EC2 client ran on the instance, so the per-call figures include the client's location; a remote client for EC2 and an SDK client for SageMaker were not measured. The SageMaker T4 ran the AWS container with a derived patch and the EC2 T4 the public vLLM image with the same patch. Prices are on-demand list prices on 2026-09-30, with no Savings Plans, Spot, sustained-use or committed-use discounts, which differ by platform.

The strategy for using MCP for SageMaker deployment and benchmarking was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/sagemaker-gemma
- Part one, deploying Gemma 4 to SageMaker: https://dev.to/aws-builders/gemma-4-on-an-amazon-sagemaker-endpoint-aws-cli-nvidia-l4-and-an-mcp-server-2c9d
- Part two, QAT against bf16: https://dev.to/aws-builders/gemma-4-on-amazon-sagemaker-qat-weights-decode-205x-faster-than-bf16-on-one-l4-318m
- Part three, 4-bit embeddings: https://dev.to/aws-builders/gemma-4-on-amazon-sagemaker-4-bit-embeddings-decode-up-to-139x-faster-on-one-l4-36mf
- Part four, the NVIDIA T4: https://github.com/xbill9/sagemaker-gemma/blob/main/docs/articles/sagemaker-gemma-t4/devto-sagemaker-gemma-t4.md
- E2B emb4: https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4
- Google Gemma 4 E2B QAT: https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct
- vLLM OpenAI server image: https://hub.docker.com/r/vllm/vllm-openai
- SageMaker real-time inference: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html
- SageMaker ProductionVariant: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html
- Cloud Run GPUs: https://cloud.google.com/run/docs/configuring/services/gpu

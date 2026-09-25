---
title: "Gemma 4 on an Amazon SageMaker Endpoint: AWS CLI, NVIDIA L4, and an MCP Server"
published: false
description: "Step by step deployment of Gemma 4 E2B to a SageMaker real-time endpoint on one NVIDIA L4 with the AWS vLLM container, driven by the aws CLI and managed by a Python MCP server from Claude Code or Gemini CLI."
tags: aws, sagemaker, gemma, mcp
cover_image: COVER_PENDING
---

This article provides a step by step deployment guide for Gemma 4 E2B to an Amazon SageMaker hosted GPU enabled system. A suite of Python MCP tools is built to simplify management of the vLLM hosted deployment with Claude Code.

https://github.com/xbill9/sagemaker-gemma

---

#### What is this project trying to Do?

This project serves Gemma 4 E2B from a SageMaker real-time endpoint on one NVIDIA L4 GPU, using the vLLM container AWS publishes for SageMaker. Every AWS call is a plain `aws` CLI command, so each step can be run by hand or by the MCP server.

A SageMaker real-time endpoint is a managed HTTPS inference server. SageMaker places the container on a GPU instance, health-checks it, routes requests to it and writes its logs to CloudWatch. There is no instance to patch, no security group to open and no load balancer to build.

---

#### Where do I start?

The strategy for starting MCP development for model management is a incremental step by step approach.

First, the basic development environment is setup with the required system variables and a working Claude Code configuration.

Then, the Python MCP server is brought up over stdio and validated with Claude Code in the local environment. The deployment follows as eight steps, each shown as the raw `aws` command and the MCP tool that runs it.

---

#### At This Point You Should Have…

- An AWS account and the AWS CLI v2, signed in with `aws login`
- A SageMaker endpoint quota of at least 1 for a single-L4 instance type (`ml.g6.xlarge`, `ml.g6.2xlarge` or `ml.g6.4xlarge`)
- Python 3.11 or newer with `mcp` 2.x
- Claude Code or Gemini CLI installed and working
- `jq` for reading JSON replies

---

#### Setup the Basic Environment

Clone the repository and install the one requirement into the system Python:

```shell
git clone https://github.com/xbill9/sagemaker-gemma
cd sagemaker-gemma
python3 -m pip install -r requirements.txt
cp .env.example .env
```

`.env` is gitignored. It holds the settings every tool reads:

```shell
cat .env
```

```plaintext
# Copy to .env (gitignored) and edit. sm.py reads it, so the MCP server, CLI and Makefile all see it.
AWS_REGION=us-east-2
MODEL_ID=google/gemma-4-E2B-it
INSTANCE_TYPE=ml.g6.xlarge
ENDPOINT_NAME=gemma-4-e2b
ROLE_NAME=sagemaker-gemma-execution-role
MAX_MODEL_LEN=8192
# Leave empty to use the newest SageMaker vLLM image in the region.
IMAGE_URI=
# Fallback instance types, highest priority first (same GPU keeps runs comparable).
INSTANCE_POOLS=ml.g6.xlarge,ml.g6.2xlarge,ml.g6.4xlarge
```

Gemma 4 is Apache-2.0 and ungated on Hugging Face, so no Hugging Face token is needed.

---

#### Model Management Tool with MCP Stdio Transport

The simplest MCP transport is stdio: the client launches the server as a local process and talks to it over stdin and stdout. In this project Claude Code is the MCP client. The server is one file, `server.py`, on the MCP Python SDK 2.x:

```python
mcp = MCPServer(RIG_NAME)
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)
```

Every tool carries one of the three annotations, so a client can tell a status check from a deploy from a delete.

The tools do no AWS work themselves. They call `sm.py`, which runs each request as an `aws` CLI subprocess:

```python
def aws(*args: str, region: str | None = REGION, parse: bool = True) -> Any:
    """Run `aws <args> --output json` and return the parsed result."""
    cmd = ["aws", *args, "--output", "json"]
```

The CLI renews an `aws login` session on its own, so a server that stays up for hours keeps working credentials. `sm.py` also drops any `AWS_SESSION_TOKEN` the server inherits from its parent process, because a static token expires inside a long-running server and outranks the login session.

---

#### Running the Python Code

The project can be linted:

```shell
make lint
```

```plaintext
All checks passed!
16 files already formatted
```

and tested:

```shell
make test
```

```plaintext
----------------------------------------------------------------------
Ran 16 tests in 0.010s

OK
```

The tests replace the `aws` subprocess with a fake, so they run offline with no credentials. One of them compares the registered tool set and annotations against a fixed list: a tool that failed to register, or a delete that lost its destructive flag, fails the suite. ✅

---

#### Test the Protocol by Hand

A client speaks JSON-RPC over stdio. Hold stdin open with `sleep`, or the server sees end-of-input and exits before it answers:

```shell
{ printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'; sleep 3; } \
  | python3 server.py 2>/dev/null
```

Summarised:

```plaintext
1 sagemaker-gemma
2 ['check_quotas', 'delete_endpoint', 'deploy_endpoint', 'find_vllm_image', 'get_deployment_config', 'get_endpoint_logs', 'get_endpoint_status', 'get_help', 'list_endpoints', 'query_model', 'verify_model_health']
```

🟢 The server answers the handshake and lists 11 tools.

---

#### Claude Code .mcp.json

Claude Code reads `.mcp.json` in the project directory and launches the server with the system `python3`:

```json
{
  "mcpServers": {
    "sagemaker-gemma": {
      "command": "python3",
      "args": ["/home/xbill/sagemaker-gemma/server.py"]
    }
  }
}
```

Gemini CLI reads the same entry from `.gemini/settings.json`.

---

#### Validation with Claude Code

```shell
claude mcp get sagemaker-gemma
```

```plaintext
sagemaker-gemma:
  Scope: Project config (shared via .mcp.json)
  Status: ✔ Connected
```

From inside Claude Code, `get_help` returns the resolved settings and the order of work:

```markdown
### sagemaker-gemma

Serving `google/gemma-4-E2B-it` with vLLM on a **SageMaker real-time endpoint**, managed
entirely through the aws CLI.

| Setting | Value |
| --- | --- |
| Region | `us-east-2` |
| Endpoint | `gemma-4-e2b` |
| Instance types (priority order) | `ml.g6.xlarge`, `ml.g6.2xlarge`, `ml.g6.4xlarge` |
| Max model length | `8192` |
| Image | `newest SageMaker vLLM image (find_vllm_image)` |
| Execution role | `sagemaker-gemma-execution-role` |

**Order of work:** check_quotas → deploy_endpoint → get_endpoint_status until
`InService` (about 10 minutes once an instance is placed) → verify_model_health
→ query_model → delete_endpoint.
```

The steps below follow that order. Each one shows the raw CLI command; `deploy_endpoint` runs Steps 2 to 6 in one call.

```shell
export AWS_REGION=us-east-2 AWS_PAGER=
export NAME=gemma-4-e2b
export MODEL_ID=google/gemma-4-E2B-it
```

---

#### Step 1 — Check the Endpoint Quota

SageMaker endpoint quotas are per instance type, per region, and count instances.

```shell
aws service-quotas list-service-quotas --service-code sagemaker \
  --query "Quotas[?QuotaName=='ml.g6.xlarge for endpoint usage'].Value"
```

```plaintext
[
    1.0
]
```

The `check_quotas` tool reads the same quota for each fallback type across the US regions:

```plaintext
| Instance | us-east-1 | us-east-2 | us-west-1 | us-west-2 |
| --- | ---: | ---: | ---: | ---: |
| `ml.g6.xlarge` | 1 | 1 | - | 1 |
| `ml.g6.2xlarge` | 1 | 1 | - | 1 |
| `ml.g6.4xlarge` | 1 | 1 | - | 1 |

Regions with quota for at least one of these types: us-east-1, us-east-2, us-west-2.
```

A `-` means the type is not offered in that region. A `0` means a quota increase request first.

---

#### 🔎 Tip: One Quota Call per Region

`list-service-quotas` pages through every SageMaker quota in the region, and the Service Quotas API is rate limited per account. Twelve calls at once, one per type per region, came back as:

```plaintext
An error occurred (TooManyRequestsException) when calling the ListServiceQuotas operation (reached max retries: 2)
```

`check_quotas` makes one call per region, one region at a time, and filters the types from that one result. `sm.py` also sets `AWS_RETRY_MODE=adaptive` so the CLI backs off and retries.

---

#### Step 2 — Find the vLLM Container

The AWS vLLM repository holds pinned release tags such as `0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1`, floating aliases such as `0.30-gpu-py312`, and `-soci` index tags. Older release lines receive patch rebuilds, so the most recent push can carry an older vLLM. Sort the pinned tags by version:

```shell
TAG=$(aws ecr describe-images --registry-id 763104351884 --repository-name vllm \
  --query "imageDetails[].imageTags[]" --output text | tr '\t' '\n' \
  | grep -E -- '^[0-9]+\.[0-9]+\.[0-9]+-.*-sagemaker-v[0-9]+\.[0-9]+$' | sort -V | tail -1)
export IMAGE=763104351884.dkr.ecr.$AWS_REGION.amazonaws.com/vllm:$TAG
echo $IMAGE
```

```plaintext
763104351884.dkr.ecr.us-east-2.amazonaws.com/vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1
```

The `find_vllm_image` tool applies the same rule. The image runs vLLM 0.30.0.

---

#### Step 3 — Create the Execution Role

SageMaker assumes this role to pull the image and write logs. It is created once per account.

```shell
aws iam create-role --role-name sagemaker-gemma-execution-role \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam attach-role-policy --role-name sagemaker-gemma-execution-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
export ROLE=$(aws iam get-role --role-name sagemaker-gemma-execution-role --query Role.Arn --output text)
```

On a second run `create-role` reports `EntityAlreadyExists`; `get-role` still sets `ROLE`. `deploy_endpoint` creates the role only when it is missing.

---

#### Step 4 — Create the Model

A SageMaker model pairs an image with its settings. The container turns each `SM_VLLM_` variable into the matching vLLM flag: `SM_VLLM_MAX_MODEL_LEN` becomes `--max-model-len`.

```shell
aws sagemaker create-model --model-name $NAME --execution-role-arn $ROLE \
  --primary-container "{\"Image\":\"$IMAGE\",\"Environment\":{
    \"SM_VLLM_MODEL\":\"$MODEL_ID\",
    \"SM_VLLM_MAX_MODEL_LEN\":\"8192\",
    \"SM_VLLM_GPU_MEMORY_UTILIZATION\":\"0.9\"}}"
```

```json
{
    "ModelArn": "arn:aws:sagemaker:us-east-2:<account-id>:model/gemma-4-e2b"
}
```

---

#### Step 5 — Create the Endpoint Config With Fallback Instance Types

The endpoint config says where the model runs. `InstancePools` lists up to five instance types in priority order, and SageMaker places the first one with a free instance. All three types below carry one L4, so the model sees the same GPU whichever is placed.

```shell
aws sagemaker create-endpoint-config --endpoint-config-name $NAME \
  --production-variants "[{\"VariantName\":\"AllTraffic\",\"ModelName\":\"$NAME\",
    \"InitialInstanceCount\":1,
    \"InstancePools\":[{\"InstanceType\":\"ml.g6.xlarge\",\"Priority\":1},
                       {\"InstanceType\":\"ml.g6.2xlarge\",\"Priority\":2},
                       {\"InstanceType\":\"ml.g6.4xlarge\",\"Priority\":3}],
    \"ContainerStartupHealthCheckTimeoutInSeconds\":1800,
    \"ModelDataDownloadTimeoutInSeconds\":1800}]"
```

```json
{
    "EndpointConfigArn": "arn:aws:sagemaker:us-east-2:<account-id>:endpoint-config/gemma-4-e2b"
}
```

The two 1800-second timeouts give the container time to download the weights and compile before SageMaker's health check gives up.

---

#### 🔎 Tip: A Fallback List Holds Quota for Every Type in It

With one Gemma endpoint running on `ml.g6.xlarge` from the three-type list above, a second endpoint asking for `ml.g6.2xlarge` in the same region was refused:

```plaintext
ResourceLimitExceeded: The account-level service limit 'ml.g6.2xlarge for endpoint usage' is 1 Instances, with current utilization of 1 Instances and a request delta of 1 Instances.
```

With a quota of 1 per type, a second endpoint goes in another region, or uses types outside the first endpoint's list.

---

#### Step 6 — Create the Endpoint

```shell
aws sagemaker create-endpoint --endpoint-name $NAME --endpoint-config-name $NAME
```

```json
{
    "EndpointArn": "arn:aws:sagemaker:us-east-2:<account-id>:endpoint/gemma-4-e2b"
}
```

Billing starts when an instance is placed. `aws sagemaker wait endpoint-in-service --endpoint-name $NAME` blocks until it is ready; `get_endpoint_status` reports the same state and the instance type that was placed:

```plaintext
✅ `gemma-4-e2b` in `us-east-2`: **InService**
- Instance: `ml.g6.xlarge`
```

The container log is in CloudWatch, and `get_endpoint_logs` tails it:

```shell
aws logs tail /aws/sagemaker/Endpoints/$NAME --follow
```

Start-up on the L4, in minutes after `create-endpoint`:

| Phase | Minutes |
| --- | ---: |
| Weights loaded, 9.75 GiB in 82.75 s | 7.4 |
| KV cache sized, 723,484 tokens | 9.2 |
| `InService` | 9.9 |

---

#### 🔎 Tip: A Creating Endpoint Cannot Be Deleted

`create-endpoint` cannot be undone until the endpoint settles. A `delete-endpoint` sent while it is `Creating` is refused:

```plaintext
aws: [ERROR]: An error occurred (ValidationException) when calling the DeleteEndpoint operation: Cannot update in-progress endpoint "arn:aws:sagemaker:us-east-2:<account-id>:endpoint/gemma-4-e2b".
```

The endpoint finishes starting, bills from the moment its instance is placed, and can be deleted once it reaches `InService` or `Failed`. Check the model ID and instance types before Step 6.

---

#### 🔎 Tip: Capacity Is Separate From Quota

A quota of 1 lets you request one instance; the region still has to have one free. In `us-east-1`, two requests for L4 instances each stayed in `Creating` for about 30 minutes and then failed:

```plaintext
Unable to provision requested ML compute capacity due to InsufficientInstanceCapacity error. Please retry using a different ML instance type or after some time.
```

The same request in `us-east-2` placed an instance at once. While SageMaker waits for capacity, no container starts and the CloudWatch log group never appears, which is how `get_endpoint_logs` tells a capacity wait from a slow model load. No charge accrues during the wait. When it fails, repeat Steps 4 to 6 in another region where Step 1 shows the same quota.

---

#### Step 7 — Cross Check the Deployed Model

The vLLM container accepts an OpenAI chat body on `invoke-endpoint`:

```shell
echo '{"messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":256}' > req.json
aws sagemaker-runtime invoke-endpoint --endpoint-name $NAME \
  --content-type application/json --body fileb://req.json out.json
jq -r '.choices[0].message.content' out.json | head -1
jq -c '{model,usage}' out.json
```

```plaintext
{
    "ContentType": "application/json",
    "InvokedProductionVariant": "AllTraffic"
}
The sky is blue due to a phenomenon called **Rayleigh scattering**. This process is caused by how sunlight interacts with the Earth's atmosphere.
{"model":"google/gemma-4-E2B-it","usage":{"prompt_tokens":15,"total_tokens":271,"completion_tokens":256,"prompt_tokens_details":null,"completion_tokens_details":null}}
```

`verify_model_health` sends one short request and checks for a reply:

```plaintext
✅ model=`google/gemma-4-E2B-it` tokens=2 wall=0.749s reply='ok'
```

---

#### Step 8 — Teardown

The endpoint bills by the hour until it is deleted. Deleting the config and the model as well leaves nothing behind. `delete_endpoint` runs all three:

```shell
aws sagemaker delete-endpoint --endpoint-name $NAME
aws sagemaker delete-endpoint-config --endpoint-config-name $NAME
aws sagemaker delete-model --model-name $NAME
```

Each command prints nothing and exits 0. `list_endpoints` confirms the account is clear:

```plaintext
### 0 endpoint(s) matching `gemma`
```

---

#### Summary

The goal of this article was to deploy Gemma 4 E2B to an Amazon SageMaker real-time endpoint with the AWS CLI and manage it from an MCP server. The key to the solution was the AWS vLLM SageMaker container, which turns the deployment into three `create-` calls and makes the endpoint answer OpenAI-style chat requests. The deployment results were:

- 🟢 Eight CLI steps take an account from quota check to a serving endpoint and back to nothing
- 🟢 The endpoint reached `InService` 9.9 minutes after `create-endpoint`, with the weights using 9.75 GiB of the L4's 24 GB
- 🟢 The MCP server runs every step from Claude Code, marks each tool read-only, write or destructive, and passes its tests offline
- 🟢 `InstancePools` gives one endpoint config several fallback instance types
- ⚠️ L4 capacity varied by region: `us-east-1` refused twice, about 30 minutes each, while `us-east-2` placed an instance at once
- ⚠️ A fallback list holds quota for every type in it

Scope: one account, `ml.g6.xlarge` with one NVIDIA L4, vLLM 0.30.0 from the AWS container, `google/gemma-4-E2B-it` at full precision, deployed in `us-east-2` on 2026-09-25. Start-up times are from a single deployment.

The strategy for using MCP for SageMaker deployment was validated with an incremental step by step approach.

---

#### References

- Repository: https://github.com/xbill9/sagemaker-gemma
- Gemma 4 E2B on Hugging Face: https://huggingface.co/google/gemma-4-E2B-it
- AWS Deep Learning Containers: https://github.com/aws/deep-learning-containers
- SageMaker real-time inference: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html
- CreateEndpointConfig API: https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateEndpointConfig.html
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- vLLM: https://docs.vllm.ai/

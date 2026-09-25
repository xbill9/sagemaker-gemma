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
| `IMAGE_URI` | empty → newest SageMaker vLLM image |

"""SageMaker real-time endpoint lifecycle and inference MCP server for Gemma 4.

Every AWS call goes through the aws CLI (`sm.aws`), not boto3: the CLI renews
`aws login` sessions on its own, so a server that stays up for hours never holds
expired credentials. `sm.py` is the engine and doubles as a command line
(`python3 sm.py status`); this file wraps it as MCP tools.

What shapes this server, measured 2026-09-25 (docs/runs/2026-09-25-e2b-bf16):

- GPU capacity decides how long a deploy takes. us-east-1 had no L4 for two
  attempts; SageMaker reported `InsufficientInstanceCapacity` about 30 minutes
  after each request. us-east-2 placed one at once and served in 9.9 minutes.
  Deploys therefore take a list of same-GPU instance types (`InstancePools`).
- A capacity wait has no container and no CloudWatch log group. Use
  get_endpoint_logs to tell a capacity wait from a slow model load.
- The AWS vLLM container takes the OpenAI chat body on `invoke-endpoint`.
"""

import asyncio
import logging

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

import sm

RIG_NAME = "sagemaker-gemma"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(RIG_NAME)

mcp = MCPServer(RIG_NAME)
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

# L4 sizes with one GPU: a fallback list of these keeps benchmarks comparable
# whichever one SageMaker places.
L4_SINGLE_GPU = ["ml.g6.xlarge", "ml.g6.2xlarge", "ml.g6.4xlarge"]
US_REGIONS = ["us-east-1", "us-east-2", "us-west-1", "us-west-2"]


async def _call(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


def _error(exc: Exception) -> str:
    if isinstance(exc, sm.AwsError):
        return f"❌ aws CLI: {exc}"
    return f"❌ {exc}"


def _pools() -> list[str]:
    return sm.INSTANCE_POOLS or [sm.INSTANCE_TYPE]


@mcp.tool(title="Help and configuration", annotations=READ_ONLY)
async def get_help() -> str:
    """Show the resolved configuration and how a deploy proceeds."""
    return f"""### {RIG_NAME}

Serving `{sm.MODEL_ID}` with vLLM on a **SageMaker real-time endpoint**, managed
entirely through the aws CLI.

| Setting | Value |
| --- | --- |
| Region | `{sm.REGION}` |
| Endpoint | `{sm.ENDPOINT_NAME}` |
| Instance types (priority order) | {", ".join(f"`{t}`" for t in _pools())} |
| Max model length | `{sm.MAX_MODEL_LEN}` |
| Image | `{sm.IMAGE_URI or "newest SageMaker vLLM image (find_vllm_image)"}` |
| Execution role | `{sm.ROLE_NAME}` |

**Order of work:** check_quotas → deploy_endpoint → get_endpoint_status until
`InService` (about 10 minutes once an instance is placed) → verify_model_health
→ query_model → delete_endpoint.

**Capacity.** A quota of 1 lets you request one instance; the region still has
to have one free. SageMaker reports `InsufficientInstanceCapacity` about 30
minutes after the request. If that happens, deploy in another region where
check_quotas shows the same quota.

Settings come from the environment, then `.env` (gitignored), then defaults.
"""


@mcp.tool(title="Generate aws CLI deployment commands", annotations=READ_ONLY)
async def get_deployment_config(
    model_id: str = sm.MODEL_ID,
    endpoint_name: str = sm.ENDPOINT_NAME,
    region: str = sm.REGION,
) -> str:
    """Return the aws CLI commands that deploy the endpoint, without changing AWS."""
    pools = ",\n".join(f'      {{"InstanceType":"{t}","Priority":{i + 1}}}' for i, t in enumerate(_pools()))
    return f"""### Deploy `{model_id}` as `{endpoint_name}` in `{region}`

```bash
export AWS_REGION={region} NAME={endpoint_name}
TAG=$(aws ecr describe-images --registry-id {sm.DLC_ACCOUNT} --repository-name vllm \\
  --query "imageDetails[].imageTags[]" --output text | tr '\\t' '\\n' \\
  | grep -E -- '^[0-9]+\\.[0-9]+\\.[0-9]+-.*-sagemaker-v[0-9]+\\.[0-9]+$' | sort -V | tail -1)
IMAGE={sm.DLC_ACCOUNT}.dkr.ecr.$AWS_REGION.amazonaws.com/vllm:$TAG
ROLE=$(aws iam get-role --role-name {sm.ROLE_NAME} --query Role.Arn --output text)

aws sagemaker create-model --model-name $NAME --execution-role-arn $ROLE \\
  --primary-container "{{\\"Image\\":\\"$IMAGE\\",\\"Environment\\":{{
    \\"SM_VLLM_MODEL\\":\\"{model_id}\\",
    \\"SM_VLLM_MAX_MODEL_LEN\\":\\"{sm.MAX_MODEL_LEN}\\",
    \\"SM_VLLM_GPU_MEMORY_UTILIZATION\\":\\"0.9\\"}}}}"

aws sagemaker create-endpoint-config --endpoint-config-name $NAME --production-variants '[{{
    "VariantName":"AllTraffic","ModelName":"'$NAME'","InitialInstanceCount":1,
    "InstancePools":[
{pools}],
    "ContainerStartupHealthCheckTimeoutInSeconds":1800,
    "ModelDataDownloadTimeoutInSeconds":1800}}]'

aws sagemaker create-endpoint --endpoint-name $NAME --endpoint-config-name $NAME
aws sagemaker wait endpoint-in-service --endpoint-name $NAME
```

The full walkthrough, including the execution role, is `docs/DEPLOY.md`.
"""


@mcp.tool(title="Find the vLLM SageMaker image", annotations=READ_ONLY)
async def find_vllm_image(region: str = sm.REGION) -> str:
    """Highest pinned version of the AWS vLLM SageMaker container in the region."""
    try:
        return f"📦 `{await _call(sm.latest_vllm_image, region)}`"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check endpoint quotas", annotations=READ_ONLY)
async def check_quotas(instance_types: list[str] | None = None) -> str:
    """Endpoint quota for each instance type in each US region.

    A value of 0 means request an increase first; '-' means the type is not
    offered in that region. Quota is permission to request, and says nothing
    about whether the region has a free instance right now.
    """
    types = instance_types or _pools()
    try:
        # One region at a time: each call pages through every SageMaker quota.
        per_region = [await _call(sm.endpoint_quotas, types, r) for r in US_REGIONS]
        grid = {(t, r): {"value": q[t]} for r, q in zip(US_REGIONS, per_region, strict=True) for t in types}
        lines = [
            "| Instance | " + " | ".join(US_REGIONS) + " |",
            "| --- |" + " ---: |" * len(US_REGIONS),
        ]
        for t in types:
            cells = []
            for r in US_REGIONS:
                value = grid[(t, r)]["value"]
                cells.append("-" if value is None else str(int(value)))
            lines.append(f"| `{t}` | " + " | ".join(cells) + " |")
        usable = [r for r in US_REGIONS if any(grid[(t, r)]["value"] for t in types)]
        lines.append("")
        lines.append(f"Regions with quota for at least one of these types: {', '.join(usable) or 'none'}.")
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Deploy endpoint", annotations=WRITE)
async def deploy_endpoint(
    model_id: str = sm.MODEL_ID,
    endpoint_name: str = sm.ENDPOINT_NAME,
    region: str = sm.REGION,
    instance_types: list[str] | None = None,
) -> str:
    """Create model, endpoint config and endpoint. Starts a billed GPU instance
    that runs until delete_endpoint. Returns at once; poll get_endpoint_status.
    An endpoint cannot be deleted while it is Creating, so a deploy cannot be
    cancelled until it reaches InService or Failed.

    `instance_types` is a priority-ordered fallback list (InstancePools);
    SageMaker places the first type that has capacity.
    """
    pools = instance_types or _pools()
    try:
        result = await _call(
            sm.deploy,
            endpoint_name=endpoint_name,
            model_id=model_id,
            instance_pools=pools,
            region=region,
        )
        logger.info("deploy submitted: %s", result)
        return (
            f"✅ `{endpoint_name}` is Creating in `{region}`.\n\n"
            f"- Model: `{model_id}`\n"
            f"- Instance types: {', '.join(f'`{t}`' for t in pools)}\n"
            f"- Image: `{result['image']}`\n\n"
            "Poll get_endpoint_status. No log group after ~10 minutes means SageMaker "
            "is still waiting for a free instance."
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get endpoint status", annotations=READ_ONLY)
async def get_endpoint_status(endpoint_name: str = sm.ENDPOINT_NAME, region: str = sm.REGION) -> str:
    """Creating, InService, Failed (with reason) or NotFound, plus the placed instance type."""
    try:
        s = await _call(sm.status, endpoint_name, region)
        icon = {"InService": "✅", "Failed": "❌", "NotFound": "⚪"}.get(s["status"], "⏳")
        lines = [f"{icon} `{endpoint_name}` in `{region}`: **{s['status']}**"]
        if s.get("instance_types"):
            lines.append(f"- Instance: {', '.join(f'`{t}`' for t in s['instance_types'])}")
        if s.get("created"):
            lines.append(f"- Created: {s['created']}; last change: {s['last_modified']}")
        if s.get("failure_reason"):
            lines.append(f"- Reason: {s['failure_reason']}")
        if s.get("image_digest"):
            lines.append(f"- Image: `{s['image_digest']}`")
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="List Gemma endpoints", annotations=READ_ONLY)
async def list_endpoints(name_contains: str = "gemma") -> str:
    """Endpoints whose name contains `name_contains`, across the US regions, with counts."""
    try:
        results = await asyncio.gather(*(_call(sm.list_endpoints, name_contains, r) for r in US_REGIONS))
        total = sum(r["count"] for r in results)
        lines = [f"### {total} endpoint(s) matching `{name_contains}`", ""]
        for region, r in zip(US_REGIONS, results, strict=True):
            for e in r["endpoints"]:
                lines.append(f"- `{e['name']}` in `{region}`: {e['status']}")
        return "\n".join(lines)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Get endpoint logs", annotations=READ_ONLY)
async def get_endpoint_logs(
    endpoint_name: str = sm.ENDPOINT_NAME, region: str = sm.REGION, tail: int = 50
) -> str:
    """Last `tail` container log lines from the past hour (model download, vLLM start-up, errors)."""
    try:
        return await _call(sm.logs, endpoint_name, tail, region)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Verify model health", annotations=READ_ONLY)
async def verify_model_health(endpoint_name: str = sm.ENDPOINT_NAME, region: str = sm.REGION) -> str:
    """Confirm the endpoint answers a chat request with a non-empty reply."""
    try:
        r = await _call(
            sm.invoke,
            "Reply with the single word: ok",
            max_tokens=16,
            temperature=0.0,
            endpoint_name=endpoint_name,
            region=region,
        )
        status = "✅" if r["text"].strip() else "❌"
        return (
            f"{status} model=`{r['model']}` tokens={r['completion_tokens']} "
            f"wall={r['wall_seconds']}s reply={r['text']!r}"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Query model", annotations=READ_ONLY)
async def query_model(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    system: str | None = None,
    endpoint_name: str = sm.ENDPOINT_NAME,
    region: str = sm.REGION,
) -> str:
    """Send a chat completion to the endpoint; returns the reply and its token counts."""
    try:
        r = await _call(
            sm.invoke,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            endpoint_name=endpoint_name,
            region=region,
        )
        return (
            f"{r['text']}\n\n---\n"
            f"`{r['model']}` · {r['prompt_tokens']} in / {r['completion_tokens']} out · "
            f"{r['wall_seconds']} s · finish={r['finish_reason']}"
        )
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Delete endpoint", annotations=DESTRUCTIVE)
async def delete_endpoint(endpoint_name: str = sm.ENDPOINT_NAME, region: str = sm.REGION) -> str:
    """Delete the endpoint, its endpoint config and its model. Stops billing."""
    try:
        r = await _call(sm.destroy, endpoint_name, region)
        return "\n".join(
            [f"🗑️ `{endpoint_name}` in `{region}`"] + [f"- {k}: {v}" for k, v in r["results"].items()]
        )
    except Exception as exc:
        return _error(exc)


if __name__ == "__main__":
    mcp.run()

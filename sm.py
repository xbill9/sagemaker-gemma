# sm.py
"""Gemma on a SageMaker real-time endpoint, driven entirely through the aws CLI.

Every AWS call in this project goes through `aws(...)`, which runs the aws CLI
as a subprocess. There is no boto3: the CLI handles `aws login` session renewal
itself, so a long-running MCP server never holds stale credentials.

Deployment is three CLI calls (create-model, create-endpoint-config,
create-endpoint) against the AWS vLLM SageMaker container, which reads its
vLLM flags from SM_VLLM_* environment variables and serves the OpenAI chat
API behind SageMaker's /invocations route.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

# --- configuration (override in .env or the environment) ---------------------


def _load_dotenv(path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Read KEY=VALUE lines from .env (gitignored). The environment wins."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E2B-it")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.g6.xlarge")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "gemma-4-e2b")
ROLE_NAME = os.environ.get("ROLE_NAME", "sagemaker-gemma-execution-role")
# Empty means "newest SageMaker vLLM image in the region" (see latest_vllm_image).
IMAGE_URI = os.environ.get("IMAGE_URI", "")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "8192")
# Optional comma-separated fallback list, highest priority first. SageMaker tries
# each type in turn when one has no capacity (InsufficientInstanceCapacity).
INSTANCE_POOLS = [t.strip() for t in os.environ.get("INSTANCE_POOLS", "").split(",") if t.strip()]

# AWS Deep Learning Containers registry (same account id in most commercial regions).
DLC_ACCOUNT = "763104351884"

# Inherited static session keys expire inside a long-running server and outrank
# the `aws login` session, so they are dropped and the CLI resolves its own.
_STALE_ENV = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


class AwsError(RuntimeError):
    """An aws CLI call failed; the message is the CLI's own stderr."""


def aws(*args: str, region: str | None = REGION, parse: bool = True) -> Any:
    """Run `aws <args> --output json` and return the parsed result."""
    cmd = ["aws", *args, "--output", "json"]
    if region:
        cmd += ["--region", region]
    env = {k: v for k, v in os.environ.items() if k not in _STALE_ENV}
    env["AWS_PAGER"] = ""
    # Paged list calls (service-quotas, logs) hit per-account rate limits; let the
    # CLI back off and retry instead of failing on the first throttle.
    env.setdefault("AWS_RETRY_MODE", "adaptive")
    env.setdefault("AWS_MAX_ATTEMPTS", "8")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        err = proc.stderr.strip()
        if "aws login" in err or "expired" in err.lower() or "Unable to locate credentials" in err:
            err += "\n-> Run `aws login` and try again."
        raise AwsError(err)
    if not parse or not proc.stdout.strip():
        return proc.stdout
    return json.loads(proc.stdout)


# --- discovery ----------------------------------------------------------------


def account_id() -> str:
    return aws("sts", "get-caller-identity", region=None)["Account"]


def latest_vllm_image(region: str = REGION) -> str:
    """Newest SageMaker-flavoured tag in the AWS vLLM DLC repository."""
    images = aws(
        "ecr",
        "describe-images",
        "--registry-id",
        DLC_ACCOUNT,
        "--repository-name",
        "vllm",
        "--query",
        "imageDetails[].{tags: imageTags, pushed: imagePushedAt}",
        region=region,
    )
    # Pinned release tags look like 0.30.0-gpu-...-sagemaker-v1.1. Older lines get
    # patch rebuilds, so push date is no guide: pick the highest version.
    pattern = re.compile(r"^(\d+\.\d+\.\d+)-gpu-.*-sagemaker-v(\d+\.\d+)$")

    def version(tag: str) -> tuple:
        m = pattern.match(tag)
        return tuple(int(x) for x in m.group(1).split(".")) + tuple(int(x) for x in m.group(2).split("."))

    tags = [t for img in images for t in (img["tags"] or []) if pattern.match(t)]
    if not tags:
        raise AwsError(f"No SageMaker vLLM image found in {DLC_ACCOUNT}/vllm in {region}.")
    tag = max(tags, key=version)
    return f"{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/vllm:{tag}"


def endpoint_quotas(instance_types: list[str], region: str = REGION) -> dict[str, float | None]:
    """Applied 'for endpoint usage' quota per instance type, from ONE list call.

    list-service-quotas pages through every SageMaker quota, so one call per
    region (not per type) keeps clear of its rate limit. None means the type is
    not offered in the region.
    """
    names = {f"{t} for endpoint usage": t for t in instance_types}
    found = aws(
        "service-quotas",
        "list-service-quotas",
        "--service-code",
        "sagemaker",
        "--query",
        "Quotas[?ends_with(QuotaName, 'for endpoint usage')].[QuotaName, Value]",
        region=region,
    )
    values = {names[n]: v for n, v in found if n in names}
    return {t: values.get(t) for t in instance_types}


def endpoint_quota(instance_type: str = INSTANCE_TYPE, region: str = REGION) -> dict:
    """The account's applied 'for endpoint usage' quota for one instance type."""
    value = endpoint_quotas([instance_type], region)[instance_type]
    return {"quota": f"{instance_type} for endpoint usage", "value": value}


# --- IAM ---------------------------------------------------------------------

_TRUST = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def ensure_role(role_name: str = ROLE_NAME) -> str:
    """Return the execution role ARN, creating the role if it does not exist."""
    try:
        return aws("iam", "get-role", "--role-name", role_name, region=None)["Role"]["Arn"]
    except AwsError as e:
        if "NoSuchEntity" not in str(e):
            raise
    arn = aws(
        "iam",
        "create-role",
        "--role-name",
        role_name,
        "--assume-role-policy-document",
        json.dumps(_TRUST),
        region=None,
    )["Role"]["Arn"]
    aws(
        "iam",
        "attach-role-policy",
        "--role-name",
        role_name,
        "--policy-arn",
        "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
        region=None,
        parse=False,
    )
    # A new role takes a few seconds to become assumable by SageMaker.
    time.sleep(10)
    return arn


# --- lifecycle ----------------------------------------------------------------


def deploy(
    endpoint_name: str = ENDPOINT_NAME,
    model_id: str = MODEL_ID,
    instance_type: str = INSTANCE_TYPE,
    image_uri: str = IMAGE_URI,
    max_model_len: str = MAX_MODEL_LEN,
    instance_pools: list[str] | None = None,
    region: str = REGION,
) -> dict:
    """Create model + endpoint config + endpoint. Returns at once; the endpoint
    takes several minutes to reach InService (poll `status`)."""
    image = image_uri or latest_vllm_image(region)
    role = ensure_role()
    container_env = {
        "SM_VLLM_MODEL": model_id,
        "SM_VLLM_MAX_MODEL_LEN": max_model_len,
        "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.9",
        "SM_VLLM_SERVED_MODEL_NAME": model_id,
    }
    aws(
        "sagemaker",
        "create-model",
        "--model-name",
        endpoint_name,
        "--execution-role-arn",
        role,
        "--primary-container",
        json.dumps({"Image": image, "Environment": container_env}),
        region=region,
    )
    pools = INSTANCE_POOLS if instance_pools is None else instance_pools
    variant = {
        "VariantName": "AllTraffic",
        "ModelName": endpoint_name,
        "InitialInstanceCount": 1,
        # Model download + vLLM start-up exceeds the default health-check window.
        "ContainerStartupHealthCheckTimeoutInSeconds": 1800,
        "ModelDataDownloadTimeoutInSeconds": 1800,
    }
    if pools:
        variant["InstancePools"] = [{"InstanceType": t, "Priority": i + 1} for i, t in enumerate(pools)]
    else:
        variant["InstanceType"] = instance_type
    aws(
        "sagemaker",
        "create-endpoint-config",
        "--endpoint-config-name",
        endpoint_name,
        "--production-variants",
        json.dumps([variant]),
        region=region,
    )
    aws(
        "sagemaker",
        "create-endpoint",
        "--endpoint-name",
        endpoint_name,
        "--endpoint-config-name",
        endpoint_name,
        region=region,
    )
    return {
        "endpoint": endpoint_name,
        "model_id": model_id,
        "instance_type": instance_type if not pools else None,
        "instance_pools": pools or None,
        "image": image,
        "role": role,
        "status": "Creating",
    }


def status(endpoint_name: str = ENDPOINT_NAME, region: str = REGION) -> dict:
    try:
        d = aws(
            "sagemaker",
            "describe-endpoint",
            "--endpoint-name",
            endpoint_name,
            region=region,
        )
    except AwsError as e:
        if "Could not find endpoint" in str(e):
            return {"endpoint": endpoint_name, "status": "NotFound"}
        raise
    out = {
        "endpoint": endpoint_name,
        "status": d["EndpointStatus"],
        "created": d["CreationTime"],
        "last_modified": d["LastModifiedTime"],
    }
    if d.get("FailureReason"):
        out["failure_reason"] = d["FailureReason"]
    variants = d.get("ProductionVariants") or []
    if variants:
        v = variants[0]
        out["instances"] = v.get("CurrentInstanceCount")
        # With InstancePools, the placed type is the pool with a running instance.
        placed = [p["InstanceType"] for p in v.get("InstancePools") or [] if p.get("CurrentInstanceCount")]
        out["instance_types"] = placed or None
        images = v.get("DeployedImages") or []
        if images:
            out["image_digest"] = images[0].get("ResolvedImage")
    return out


def list_endpoints(name_contains: str = "gemma", region: str = REGION) -> dict:
    """Endpoints whose name contains `name_contains`, with the count computed here."""
    eps = aws(
        "sagemaker",
        "list-endpoints",
        "--name-contains",
        name_contains,
        "--query",
        "Endpoints[].{name: EndpointName, status: EndpointStatus}",
        region=region,
    )
    by_status: dict[str, int] = {}
    for e in eps:
        by_status[e["status"]] = by_status.get(e["status"], 0) + 1
    return {
        "filter": {"name_contains": name_contains, "region": region},
        "count": len(eps),
        "by_status": by_status,
        "endpoints": eps,
    }


def invoke(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    system: str | None = None,
    endpoint_name: str = ENDPOINT_NAME,
    region: str = REGION,
    extra: dict | None = None,
) -> dict:
    """One chat completion via `aws sagemaker-runtime invoke-endpoint`.

    `extra` merges vLLM sampling fields into the body, e.g. {"ignore_eos": True}
    to force exactly `max_tokens` output tokens.
    """
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, **(extra or {})}
    with tempfile.TemporaryDirectory() as tmp:
        req = os.path.join(tmp, "request.json")
        resp = os.path.join(tmp, "response.json")
        with open(req, "w") as f:
            json.dump(body, f)
        start = time.perf_counter()
        aws(
            "sagemaker-runtime",
            "invoke-endpoint",
            "--endpoint-name",
            endpoint_name,
            "--content-type",
            "application/json",
            "--accept",
            "application/json",
            "--body",
            f"fileb://{req}",
            resp,
            region=region,
        )
        elapsed = time.perf_counter() - start
        with open(resp) as f:
            data = json.load(f)
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    return {
        "text": data["choices"][0]["message"]["content"],
        "finish_reason": data["choices"][0].get("finish_reason"),
        "model": data.get("model"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        # Wall clock includes the CLI's own start-up and the network round trip.
        "wall_seconds": round(elapsed, 3),
        "tokens_per_second": (
            round(completion_tokens / elapsed, 1) if completion_tokens and elapsed else None
        ),
    }


def destroy(endpoint_name: str = ENDPOINT_NAME, region: str = REGION) -> dict:
    """Delete endpoint, endpoint config and model. Missing pieces are skipped."""
    results = {}
    for kind, flag in (
        ("endpoint", "--endpoint-name"),
        ("endpoint-config", "--endpoint-config-name"),
        ("model", "--model-name"),
    ):
        try:
            aws(
                "sagemaker",
                f"delete-{kind}",
                flag,
                endpoint_name,
                region=region,
                parse=False,
            )
            results[kind] = "deleted"
        except AwsError as e:
            msg = str(e)
            if "Could not find" in msg:
                results[kind] = "not found"
            elif "Cannot update in-progress endpoint" in msg:
                # SageMaker refuses to delete an endpoint while it is Creating; the
                # config and model it uses cannot go either, so stop here.
                results[kind] = (
                    "refused: endpoint is still Creating; delete it once it is InService or Failed"
                )
                break
            else:
                results[kind] = f"error: {msg}"
    return {"endpoint": endpoint_name, "results": results}


def logs(
    endpoint_name: str = ENDPOINT_NAME, limit: int = 50, region: str = REGION, since_seconds: int = 3600
) -> str:
    """Most recent container log lines from CloudWatch for the endpoint."""
    group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    try:
        events = aws(
            "logs",
            "filter-log-events",
            "--log-group-name",
            group,
            "--start-time",
            str(int((time.time() - since_seconds) * 1000)),
            "--query",
            "events[].message",
            region=region,
        )
    except AwsError as e:
        if "ResourceNotFoundException" in str(e):
            return f"No log group {group} yet (the container has not started)."
        raise
    return "\n".join(events[-limit:])


# --- command line ---------------------------------------------------------------


def _main(argv: list[str]) -> int:
    commands = {
        "image": lambda: latest_vllm_image(),
        "quota": lambda: endpoint_quota(),
        "role": lambda: ensure_role(),
        "deploy": lambda: deploy(),
        "status": lambda: status(),
        "list": lambda: list_endpoints(),
        "invoke": lambda: invoke(" ".join(argv[1:]) or "Say hello in one sentence."),
        "logs": lambda: logs(),
        "destroy": lambda: destroy(),
    }
    if not argv or argv[0] not in commands:
        print(f"usage: python sm.py {{{','.join(commands)}}} [prompt]", file=sys.stderr)
        return 2
    try:
        result = commands[argv[0]]()
    except AwsError as e:
        print(e, file=sys.stderr)
        return 1
    print(result if isinstance(result, str) else json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))

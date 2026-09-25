# bench.py
"""Run a fixed prompt set against the endpoint and write computed results.

Same prompts, same sampling for every model, so a full-size run and a QAT run
are directly comparable. All statistics are computed here; the output files
are what a write-up quotes.

    python bench.py docs/runs/<run-dir> [repeats]
"""

import json
import re
import statistics
import sys
import time
from datetime import UTC, datetime, timedelta

import sm

PROMPTS = [
    ("short-fact", "What is the capital of Australia? Answer in one word."),
    ("arithmetic", "What is 17 * 23? Reply with the number only."),
    (
        "explain",
        "Explain what an Amazon SageMaker real-time endpoint is in three sentences.",
    ),
    (
        "code",
        ("Write a Python function that returns the n-th Fibonacci number iteratively. Code only."),
    ),
    (
        "long",
        (
            "Write a 300-word overview of the trade-offs between quantized and full-precision "
            "language models for inference."
        ),
    ),
]
CHECKS = {"short-fact": r"\bcanberra\b", "arithmetic": r"\b391\b"}
MAX_TOKENS = 512
TEMPERATURE = 0.0


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def cloudwatch_latency(start: datetime, end: datetime) -> dict:
    """Server-side ModelLatency and OverheadLatency (microseconds → seconds)."""
    out = {}
    for metric in ("ModelLatency", "OverheadLatency", "Invocations"):
        stat = "Sum" if metric == "Invocations" else "Average"
        data = sm.aws(
            "cloudwatch",
            "get-metric-statistics",
            "--namespace",
            "AWS/SageMaker",
            "--metric-name",
            metric,
            "--dimensions",
            f"Name=EndpointName,Value={sm.ENDPOINT_NAME}",
            "Name=VariantName,Value=AllTraffic",
            "--start-time",
            start.isoformat(),
            "--end-time",
            end.isoformat(),
            "--period",
            "60",
            "--statistics",
            stat,
            "--query",
            "Datapoints",
        )
        out[metric] = data
    return out


def load_facts() -> dict:
    """vLLM start-up facts parsed from the container log (load time, memory, KV size)."""
    text = sm.logs(limit=100000)
    patterns = {
        "vllm_version": r"vLLM API server version ([\w.+-]+)",
        "weights_gib": r"Model loading took ([\d.]+) GiB",
        "load_seconds": r"Model loading took [\d.]+ GiB.*?and ([\d.]+) second",
        "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
        "max_concurrency": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
    }
    facts = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        facts[key] = m.group(1) if m else None
    return facts


def main(run_dir: str, repeats: int = 3) -> None:
    start = datetime.now(UTC)
    calls = []
    for name, prompt in PROMPTS:
        for i in range(repeats):
            r = sm.invoke(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
            r.update(prompt_name=name, repeat=i)
            if name in CHECKS:
                r["correct"] = bool(re.search(CHECKS[name], r["text"], re.IGNORECASE))
            calls.append(r)
            print(
                f"{name}#{i}: {r['completion_tokens']} tok in {r['wall_seconds']}s",
                flush=True,
            )
    end = datetime.now(UTC)

    per_prompt = {}
    for name, _ in PROMPTS:
        rows = [c for c in calls if c["prompt_name"] == name]
        per_prompt[name] = {
            "wall_seconds": summarize([c["wall_seconds"] for c in rows]),
            "completion_tokens": summarize([c["completion_tokens"] for c in rows]),
            "tokens_per_second": summarize([c["tokens_per_second"] for c in rows]),
            "identical_outputs": len({c["text"] for c in rows}) == 1,
        }
        if name in CHECKS:
            per_prompt[name]["correct"] = sum(c["correct"] for c in rows)

    # CloudWatch publishes per-minute metrics a minute or two late.
    time.sleep(150)
    result = {
        "endpoint": sm.ENDPOINT_NAME,
        "model_id": sm.MODEL_ID,
        "instance_type": sm.INSTANCE_TYPE,
        "sampling": {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "repeats": repeats,
        },
        "window": [start.isoformat(), end.isoformat()],
        "total_calls": len(calls),
        "per_prompt": per_prompt,
        "load_facts": load_facts(),
        "cloudwatch": cloudwatch_latency(start - timedelta(minutes=1), end + timedelta(minutes=2)),
    }
    with open(f"{run_dir}/bench.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(f"{run_dir}/bench-calls.json", "w") as f:
        json.dump(calls, f, indent=2)
    print(json.dumps({k: result[k] for k in ("per_prompt", "load_facts")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)

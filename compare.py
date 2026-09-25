"""Compare two Gemma endpoints (e.g. full size vs QAT) on the same measurements.

    python3 compare.py measure <run-dir> <name>@<region> [<name>@<region>]
    python3 compare.py combine <out.json> <measure-a.json> <measure-b.json>

With two live endpoints, measure alternates between them so drift over time
lands on both. With one, run measure per endpoint and combine the files. All
statistics are computed here; a write-up quotes the JSON files.

1. decode   Fixed-length outputs (ignore_eos) of SHORT and LONG tokens. The
            decode rate is (LONG - SHORT) / (wall_long - wall_short), which
            cancels the per-call cost of the aws CLI and the network.
2. load     C parallel requests of LOAD_TOKENS each; aggregate tokens/s is
            C * LOAD_TOKENS / wall of the whole batch. Cross-checked against
            the "Avg generation throughput" lines vLLM writes to CloudWatch.
3. quality  Fixed questions with exact answers, temperature 0, scored by regex;
            plus how often the two endpoints give byte-identical answers.
"""

import json
import random
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import sm

SHORT, LONG, DECODE_REPEATS = 16, 512, 5
LOAD_TOKENS, CONCURRENCY, LOAD_REPEATS = 256, [1, 4, 16], 2
STORY = "Write a long, detailed story about a lighthouse keeper on a remote island."


def questions() -> list[tuple[str, str, str]]:
    """(id, prompt, answer regex). Seeded so every run asks the same questions."""
    rng = random.Random(20260925)
    qs = []
    for i in range(15):
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        qs.append((f"mul-{i}", f"What is {a} * {b}? Reply with the number only.", rf"\b{a * b}\b"))
    for i in range(15):
        a, b, c = rng.randint(100, 999), rng.randint(100, 999), rng.randint(100, 999)
        qs.append(
            (f"add-{i}", f"What is {a} + {b} - {c}? Reply with the number only.", rf"(?<![\d-]){a + b - c}\b")
        )
    capitals = {
        "Australia": "Canberra",
        "Canada": "Ottawa",
        "Brazil": "Bras[ií]lia",
        "Turkey": "Ankara",
        "Nigeria": "Abuja",
        "Switzerland": "Bern",
        "Pakistan": "Islamabad",
        "New Zealand": "Wellington",
        "Morocco": "Rabat",
        "Vietnam": "Hanoi",
    }
    for country, city in capitals.items():
        qs.append(
            (f"cap-{country}", f"What is the capital of {country}? Answer in one word.", rf"(?i)\b{city}\b")
        )
    return qs


def invoke(ep: dict, prompt: str, max_tokens: int, fixed: bool = False) -> dict:
    extra = {"ignore_eos": True} if fixed else None
    return sm.invoke(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        endpoint_name=ep["name"],
        region=ep["region"],
        extra=extra,
    )


def median(xs: list[float]) -> float:
    return round(statistics.median(xs), 3)


def decode(eps: list[dict]) -> dict:
    walls = {ep["name"]: {SHORT: [], LONG: []} for ep in eps}
    for rep in range(DECODE_REPEATS):
        for n in (SHORT, LONG):
            for ep in eps if rep % 2 == 0 else eps[::-1]:
                r = invoke(ep, STORY, n, fixed=True)
                if r["completion_tokens"] != n:
                    raise RuntimeError(f"{ep['name']}: asked for {n} tokens, got {r['completion_tokens']}")
                walls[ep["name"]][n].append(r["wall_seconds"])
                print(f"decode {ep['name']} n={n} rep={rep}: {r['wall_seconds']}s", flush=True)
    out = {}
    for name, w in walls.items():
        ws, wl = median(w[SHORT]), median(w[LONG])
        out[name] = {
            f"wall_{SHORT}_median": ws,
            f"wall_{LONG}_median": wl,
            f"wall_{LONG}_min": min(w[LONG]),
            f"wall_{LONG}_max": max(w[LONG]),
            "decode_tokens_per_second": round((LONG - SHORT) / (wl - ws), 1),
            "per_call_fixed_cost_seconds": round(ws - SHORT * (wl - ws) / (LONG - SHORT), 3),
            "samples_per_length": DECODE_REPEATS,
        }
    return out


def load(eps: list[dict]) -> dict:
    out = {ep["name"]: {} for ep in eps}
    for c in CONCURRENCY:
        for rep in range(LOAD_REPEATS):
            for ep in eps if rep % 2 == 0 else eps[::-1]:
                start = time.perf_counter()
                with ThreadPoolExecutor(max_workers=c) as pool:
                    results = list(
                        pool.map(lambda _, ep=ep: invoke(ep, STORY, LOAD_TOKENS, fixed=True), range(c))
                    )
                wall = time.perf_counter() - start
                tokens = sum(r["completion_tokens"] for r in results)
                out[ep["name"]].setdefault(str(c), []).append(
                    {"wall": round(wall, 3), "tokens": tokens, "tokens_per_second": round(tokens / wall, 1)}
                )
                print(f"load {ep['name']} c={c} rep={rep}: {tokens} tok in {wall:.2f}s", flush=True)
            time.sleep(12)  # let a vLLM stats line (every 10 s) close on this batch
    summary = {}
    for name, by_c in out.items():
        summary[name] = {
            c: {
                "aggregate_tokens_per_second_median": median([b["tokens_per_second"] for b in batches]),
                "batch_wall_seconds_median": median([b["wall"] for b in batches]),
                "batches": batches,
            }
            for c, batches in by_c.items()
        }
    return summary


def quality(eps: list[dict]) -> dict:
    qs = questions()
    answers = {ep["name"]: {} for ep in eps}
    for i, (qid, prompt, _) in enumerate(qs):
        for ep in eps if i % 2 == 0 else eps[::-1]:
            answers[ep["name"]][qid] = invoke(ep, prompt, 64)["text"].strip()
        print(f"quality {qid}", flush=True)
    out = {}
    for name, ans in answers.items():
        correct = {qid for qid, _, pat in qs if re.search(pat, ans[qid])}
        by_kind = {}
        for qid, _, _ in qs:
            kind = qid.split("-")[0]
            k = by_kind.setdefault(kind, {"correct": 0, "total": 0})
            k["total"] += 1
            k["correct"] += qid in correct
        out[name] = {"correct": len(correct), "total": len(qs), "by_kind": by_kind, "answers": ans}
    return out


def server_throughput(ep: dict, start_ms: int, end_ms: int) -> dict:
    """Peak and per-line vLLM 'Avg generation throughput' from the endpoint's CloudWatch logs."""
    proc = subprocess.run(
        [
            "aws",
            "logs",
            "filter-log-events",
            "--region",
            ep["region"],
            "--log-group-name",
            f"/aws/sagemaker/Endpoints/{ep['name']}",
            "--start-time",
            str(start_ms),
            "--end-time",
            str(end_ms),
            "--filter-pattern",
            '"Avg generation throughput"',
            "--query",
            "events[].message",
            "--output",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    values = [
        float(m.group(1))
        for line in json.loads(proc.stdout)
        if (m := re.search(r"Avg generation throughput: ([\d.]+)", line))
    ]
    return {"lines": len(values), "peak_tokens_per_second": max(values) if values else None}


def load_facts(ep: dict) -> dict:
    text = sm.logs(ep["name"], limit=1_000_000, region=ep["region"], since_seconds=12 * 3600)
    patterns = {
        "vllm_version": r"version (\d+\.\d+\.\d+)",
        "weights_gib": r"Model loading took ([\d.]+) GiB",
        "load_seconds": r"Model loading took [\d.]+ GiB memory and ([\d.]+) second",
        "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
        "max_concurrency_at_max_len": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
        "quantization": r"quantization=([\w-]+)",
    }
    # A reused endpoint name shares one log group across deployments: take the
    # newest start-up, which is the last match.
    return {k: (found[-1] if (found := re.findall(p, text)) else None) for k, p in patterns.items()}


def measure(run_dir: str, specs: list[str]) -> None:
    """Measure one or more live endpoints; writes <run-dir>/measure-<name>.json per endpoint."""
    eps = []
    for spec in specs:
        name, region = spec.split("@")
        st = sm.status(name, region)
        if st["status"] != "InService":
            raise RuntimeError(f"{name} in {region} is {st['status']}")
        eps.append(
            {
                "name": name,
                "region": region,
                "instance_types": st.get("instance_types"),
                "image_digest": st.get("image_digest"),
            }
        )
    started = datetime.now(UTC).isoformat()
    facts = {ep["name"]: load_facts(ep) for ep in eps}
    dec = decode(eps)
    t0 = int(time.time() * 1000)
    ld = load(eps)
    time.sleep(15)
    t1 = int(time.time() * 1000)
    srv = {ep["name"]: server_throughput(ep, t0, t1) for ep in eps}
    qual = quality(eps)
    settings = {
        "short": SHORT,
        "long": LONG,
        "decode_repeats": DECODE_REPEATS,
        "load_tokens": LOAD_TOKENS,
        "concurrency": CONCURRENCY,
        "load_repeats": LOAD_REPEATS,
        "temperature": 0.0,
    }
    for ep in eps:
        n = ep["name"]
        result = {
            "endpoint": ep,
            "started": started,
            "finished": datetime.now(UTC).isoformat(),
            "settings": settings,
            "load_facts": facts[n],
            "decode": dec[n],
            "load": ld[n],
            "server_throughput_during_load": srv[n],
            "quality": qual[n],
        }
        path = f"{run_dir}/measure-{n}.json"
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"wrote {path}")
        print(json.dumps(brief(result), indent=2))


def brief(r: dict) -> dict:
    return {
        "load_facts": r["load_facts"],
        "decode_tokens_per_second": r["decode"]["decode_tokens_per_second"],
        "per_call_fixed_cost_seconds": r["decode"]["per_call_fixed_cost_seconds"],
        "load_tokens_per_second": {c: v["aggregate_tokens_per_second_median"] for c, v in r["load"].items()},
        "server_peak": r["server_throughput_during_load"],
        "quality": f"{r['quality']['correct']}/{r['quality']['total']}",
    }


def combine(out_path: str, a_path: str, b_path: str) -> None:
    """Compare two measure-*.json files; ratios are b / a, computed here."""
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    na, nb = a["endpoint"]["name"], b["endpoint"]["name"]

    def ratio(x, y):
        return round(y / x, 2) if x else None

    wa, wb = float(a["load_facts"]["weights_gib"]), float(b["load_facts"]["weights_gib"])
    ka = int(a["load_facts"]["kv_cache_tokens"].replace(",", ""))
    kb = int(b["load_facts"]["kv_cache_tokens"].replace(",", ""))
    rows = {
        "weights_gib": [wa, wb, ratio(wa, wb)],
        "kv_cache_tokens": [ka, kb, ratio(ka, kb)],
        "load_seconds": [
            float(a["load_facts"]["load_seconds"]),
            float(b["load_facts"]["load_seconds"]),
            None,
        ],
        "decode_tokens_per_second": [
            a["decode"]["decode_tokens_per_second"],
            b["decode"]["decode_tokens_per_second"],
            ratio(a["decode"]["decode_tokens_per_second"], b["decode"]["decode_tokens_per_second"]),
        ],
    }
    for c in a["load"]:
        x = a["load"][c]["aggregate_tokens_per_second_median"]
        y = b["load"][c]["aggregate_tokens_per_second_median"]
        rows[f"load_c{c}_tokens_per_second"] = [x, y, ratio(x, y)]
    qa, qb = a["quality"], b["quality"]
    rows["quality_correct"] = [qa["correct"], qb["correct"], None]
    ids = list(qa["answers"])
    same = [i for i in ids if qa["answers"][i] == qb["answers"].get(i)]
    result = {
        "a": {"name": na, "model": a["load_facts"], "endpoint": a["endpoint"], "window": a["started"]},
        "b": {"name": nb, "model": b["load_facts"], "endpoint": b["endpoint"], "window": b["started"]},
        "columns": [na, nb, f"{nb} / {na}"],
        "rows": rows,
        "identical_answers": {"count": len(same), "total": len(ids)},
        "disagreements": [
            {"id": i, na: qa["answers"][i], nb: qb["answers"].get(i)} for i in ids if i not in same
        ],
        "quality_by_kind": {na: qa["by_kind"], nb: qb["by_kind"]},
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    width = max(len(k) for k in rows)
    print(f"{'':<{width}}  {na:>14}  {nb:>18}  ratio")
    for k, (x, y, r) in rows.items():
        print(f"{k:<{width}}  {x!s:>14}  {y!s:>18}  {'' if r is None else r}")
    print(f"identical answers: {len(same)}/{len(ids)}")


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "combine":
        combine(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit("usage: compare.py measure <run-dir> <name>@<region> ... | combine <out> <a.json> <b.json>")

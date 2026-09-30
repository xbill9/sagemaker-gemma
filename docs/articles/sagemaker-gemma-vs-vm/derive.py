"""Every ratio and $/M-token figure in the article, computed from the evidence files.

python3 derive.py > evidence/derived.txt
"""

import json
from datetime import datetime
from pathlib import Path

E = Path(__file__).resolve().parent / "evidence"
PRICE = {  # $/h, evidence/prices.md
    "sm_t4": 0.7360,
    "sm_l4": 1.1267,
    "ec2_t4": 0.5260,
    "ec2_l4": 0.8048,
    "gce_t4": 0.5241,
    "g2_4": 0.7045,
    "g2_8": 0.8508,
    "cloudrun_l4": 1.4209,
}
RUNS = {  # (SageMaker, EC2) measure files, same checkpoint per GPU
    "T4": (
        "runs/measure-gemma-4-e2b-emb4-t4.json",
        "ec2-t4/measure-gemma-4-e2b-emb4-g4dn.json",
        "sm_t4",
        "ec2_t4",
    ),
    "L4": (
        "ec2-l4/measure-gemma-4-e2b-qat.json",
        "ec2-l4/measure-gemma-4-e2b-qat-g6.json",
        "sm_l4",
        "ec2_l4",
    ),
}


def load(rel):
    return json.loads((E / rel).read_text())


def rate(d, n):
    return d["load"][str(n)]["aggregate_tokens_per_second_median"]


def per_mtok(price, tps):
    return price / (tps * 3600) * 1e6


def minutes(a, b):
    f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))  # noqa: E731
    return (f(b) - f(a)).total_seconds() / 60


print("# derived by derive.py from evidence/ (measure JSONs, timelines) and evidence/prices.md")
for gpu, (sm_f, ec2_f, sm_p, ec2_p) in RUNS.items():
    s, e = load(sm_f), load(ec2_f)
    same = sum(e["quality"]["answers"].get(k) == v for k, v in s["quality"]["answers"].items())
    print(
        f"{gpu}: decode SM {s['decode']['decode_tokens_per_second']} EC2 {e['decode']['decode_tokens_per_second']} "
        f"ratio {e['decode']['decode_tokens_per_second'] / s['decode']['decode_tokens_per_second']:.2f}x; "
        f"per-call SM {s['decode']['per_call_fixed_cost_seconds']} EC2 {e['decode']['per_call_fixed_cost_seconds']}; answers identical {same}/40"
    )
    for n in (1, 4, 16):
        print(f"  {n}-par SM {rate(s, n)} EC2 {rate(e, n)} ratio {rate(e, n) / rate(s, n):.2f}x")
    print(f"  hourly SM/EC2 {PRICE[sm_p] / PRICE[ec2_p]:.2f}x, EC2/SM {PRICE[ec2_p] / PRICE[sm_p]:.2f}x")
    a, b = per_mtok(PRICE[sm_p], rate(s, 16)), per_mtok(PRICE[ec2_p], rate(e, 16))
    c = per_mtok(PRICE[ec2_p], rate(s, 16))
    print(
        f"  $/Mtok at 16-par, each on its own rate: SM ${a:.3f} EC2 ${b:.3f} EC2/SM {b / a:.2f}x; "
        f"EC2 at SageMaker's rate ${c:.3f} ({c / a:.2f}x)"
    )

t4 = [line.split()[0] for line in (E / "ec2-t4/timeline.txt").read_text().splitlines()]
ready = next(
    line.split()[0] for line in (E / "ec2-t4/timeline.txt").read_text().splitlines() if "READY" in line
)
print(f"EC2 T4 launch -> serving: {minutes(t4[0], ready):.1f} min")
print(
    f"GCE T4 / EC2 T4 hourly {PRICE['gce_t4'] / PRICE['ec2_t4']:.3f}x; g2-standard-4 / EC2 g6 {PRICE['g2_4'] / PRICE['ec2_l4']:.3f}x"
)
cr = PRICE["cloudrun_l4"] / PRICE["g2_8"]
print(f"Cloud Run L4 / g2-standard-8: {cr:.2f}x; break-even uptime {1 / cr:.0%} = {24 / cr:.1f} h/day")

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
    "g2_4": 0.7045,
    "g2_8": 0.8508,
    "cloudrun_l4": 1.4209,
}


def load(name):
    return json.loads((E / "runs" / f"measure-gemma-4-{name}.json").read_text())


def dec(d):
    return d["decode"]["decode_tokens_per_second"]


def par16(d):
    return d["load"]["16"]["aggregate_tokens_per_second_median"]


def per_mtok(price, tps):
    return price / (tps * 3600) * 1e6


print("# derived by derive.py from evidence/runs/*.json and evidence/prices.md")
print(
    f"SageMaker/EC2 hourly: T4 {PRICE['sm_t4'] / PRICE['ec2_t4']:.2f}x  L4 {PRICE['sm_l4'] / PRICE['ec2_l4']:.2f}x"
)
print(
    f"T4/L4 hourly: SageMaker {PRICE['sm_t4'] / PRICE['sm_l4']:.2f}x  EC2 {PRICE['ec2_t4'] / PRICE['ec2_l4']:.2f}x"
)
print(f"EC2 T4 / SageMaker L4 hourly: {PRICE['ec2_t4'] / PRICE['sm_l4']:.2f}x")
print(
    f"EC2 T4 / SageMaker T4 hourly: {PRICE['ec2_t4'] / PRICE['sm_t4']:.2f}x  EC2 L4 / SageMaker L4: {PRICE['ec2_l4'] / PRICE['sm_l4']:.2f}x"
)
cr8 = PRICE["cloudrun_l4"] / PRICE["g2_8"]
print(f"Cloud Run L4 / g2-standard-8: {cr8:.2f}x; break-even uptime {1 / cr8:.0%} = {24 / cr8:.1f} h/day")
for m in ("e2b", "e4b", "12b"):
    t, g = load(f"{m}-emb4-t4"), load(f"{m}-emb4-l4")
    same = sum(t["quality"]["answers"].get(k) == v for k, v in g["quality"]["answers"].items())
    print(
        f"{m}: decode T4 {dec(t)} L4 {dec(g)} ratio {dec(t) / dec(g):.2f}x; "
        f"16-par T4 {par16(t)} L4 {par16(g)} ratio {par16(t) / par16(g):.2f}x; "
        f"answers identical {same}/{len(g['quality']['answers'])}"
    )
    print(
        f"  $/Mtok at 16-par: SageMaker L4 ${per_mtok(PRICE['sm_l4'], par16(g)):.3f}  "
        f"SageMaker T4 ${per_mtok(PRICE['sm_t4'], par16(t)):.3f}  "
        f"EC2 L4 ${per_mtok(PRICE['ec2_l4'], par16(g)):.3f}  EC2 T4 ${per_mtok(PRICE['ec2_t4'], par16(t)):.3f}"
    )

print("# part four")
for m in ("e2b", "e4b", "12b"):
    t, g = load(f"{m}-emb4-t4"), load(f"{m}-emb4-l4")
    tt = per_mtok(PRICE["sm_t4"], par16(t))
    gg = per_mtok(PRICE["sm_l4"], par16(g))
    kt = int(t["load_facts"]["kv_cache_tokens"].replace(",", ""))
    kg = int(g["load_facts"]["kv_cache_tokens"].replace(",", ""))
    print(f"{m}: SageMaker $/Mtok T4/L4 {tt / gg:.2f}x (+{tt / gg - 1:.0%}); KV L4/T4 {kg / kt:.2f}x")
ts = [line.split()[0] for line in (E / "runs" / "timeline-26b-t4.txt").read_text().splitlines()]
d = datetime.fromisoformat(ts[1].replace("Z", "+00:00")) - datetime.fromisoformat(
    ts[0].replace("Z", "+00:00")
)
print(f"26b Creating -> Failed: {d.total_seconds() / 60:.1f} min")

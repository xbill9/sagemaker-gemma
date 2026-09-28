#!/bin/bash
# Gemma 4 12B: bf16 then QAT, one endpoint at a time, on ml.g6e.xlarge (1x L40S,
# 48 GB): 12B bf16 weights (22.4 GiB) leave no KV cache room on a 24 GB L4.
# Same container tag, settings and compare.py as the E2B and E4B runs.
# A region is abandoned only on a capacity failure; any other failure stops.
cd /home/xbill/sagemaker-gemma
C=docs/runs/2026-09-28-12b-qat-vs-bf16
TAG=vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1
REGIONS="us-east-2 us-west-2 us-east-1"
# INSTANCE_POOLS needs 2+ types; empty means InstanceType alone.
export INSTANCE_TYPE=ml.g6e.xlarge INSTANCE_POOLS=
run_one() {  # <endpoint> <model_id> <run-dir>
  export ENDPOINT_NAME=$1 MODEL_ID=$2; R=$3; placed=""
  mkdir -p "$R"; env | grep -E '^(INSTANCE_|ENDPOINT_NAME|MODEL_ID|MAX_MODEL_LEN)' | sort > "$R/run-env.txt"
  for region in $REGIONS; do
    export IMAGE_URI=763104351884.dkr.ecr.$region.amazonaws.com/$TAG
    if ./failover.sh "$R" "$region"; then placed=$region; break; fi
    last=$(ls -d "$R"/attempt-* | sort -V | tail -1)
    reason=$(python3 -c "import json;print(json.load(open('$last/status-final.json')).get('failure_reason',''))" 2>/dev/null)
    echo "$reason" | grep -qi capacity || { echo "STOP $1: non-capacity failure: ${reason:0:200}"; return 1; }
  done
  [ -z "$placed" ] && { echo "NO CAPACITY for $1"; return 1; }
  export AWS_REGION=$placed
  python3 sm.py status > "$R/status-inservice.json"
  echo "$(date -u +%T) measuring $1@$placed"
  python3 compare.py measure "$C" "$1@$placed" > "$C/measure-$1.log" 2>&1 && echo "measured $1" || echo "MEASURE FAILED $1: $(tail -3 "$C/measure-$1.log")"
  python3 sm.py logs > "$R/logs-tail.txt" 2>&1
  python3 sm.py destroy > "$R/destroy.json" 2>&1; date -u +%FT%TZ > "$R/deleted-at.txt"
  until [ "$(python3 sm.py status | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')" = NotFound ]; do sleep 10; done
  echo "$(date -u +%T) deleted $1"
}
date -u +%FT%TZ > $C/start.txt; aws --version > $C/meta.txt 2>&1
run_one gemma-4-12b google/gemma-4-12B-it docs/runs/2026-09-28-12b-bf16
run_one gemma-4-12b-qat google/gemma-4-12B-it-qat-w4a16-ct docs/runs/2026-09-28-12b-qat
[ -f $C/measure-gemma-4-12b.json ] && [ -f $C/measure-gemma-4-12b-qat.json ] && \
  python3 compare.py combine $C/compare.json $C/measure-gemma-4-12b.json $C/measure-gemma-4-12b-qat.json
echo "ALL DONE"

#!/bin/bash
# Gemma 4 E4B: bf16 then QAT, one endpoint at a time, same container and
# settings as the 2026-09-25 E2B runs. For each: deploy (failover by region),
# measure with compare.py, delete. Then combine.
cd /home/xbill/sagemaker-gemma
C=docs/runs/2026-09-27-e4b-qat-vs-bf16
TAG=vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1
REGIONS="us-east-2 us-west-2 us-east-1"
run_one() {  # <endpoint> <model_id> <run-dir>
  export ENDPOINT_NAME=$1 MODEL_ID=$2; R=$3; placed=""
  for region in $REGIONS; do
    export IMAGE_URI=763104351884.dkr.ecr.$region.amazonaws.com/$TAG
    if ./failover.sh "$R" "$region"; then placed=$region; break; fi
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
date -u +%FT%TZ > $C/start.txt; aws --version > $C/meta.txt
run_one gemma-4-e4b google/gemma-4-E4B-it docs/runs/2026-09-27-e4b-bf16
run_one gemma-4-e4b-qat google/gemma-4-E4B-it-qat-w4a16-ct docs/runs/2026-09-27-e4b-qat
[ -f $C/measure-gemma-4-e4b.json ] && [ -f $C/measure-gemma-4-e4b-qat.json ] && \
  python3 compare.py combine $C/compare.json $C/measure-gemma-4-e4b.json $C/measure-gemma-4-e4b-qat.json
echo "ALL DONE"

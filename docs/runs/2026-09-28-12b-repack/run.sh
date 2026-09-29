#!/bin/bash
# xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct (a W4A16 repack of Google's
# -qat-q4_0-unquantized 12B) against Google's own -qat-w4a16-ct, on the two
# instances Google's build was measured on today: ml.g6.xlarge (L4) and
# ml.g6e.xlarge (L40S). Standard settings; same container tag and compare.py.
# Waits until the repo is readable without a token (the container is anonymous).
cd /home/xbill/sagemaker-gemma
C=docs/runs/2026-09-28-12b-repack
M=xbill9/gemma-4-12B-it-qat-q4_0-w4a16-ct
until curl -sf -o /dev/null "https://huggingface.co/api/models/$M"; do sleep 30; done
echo "$(date -u +%T) $M is public"
TAG=vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1
REGIONS="us-east-2 us-west-2 us-east-1"
export INSTANCE_POOLS= TENSOR_PARALLEL_SIZE= MAX_MODEL_LEN=8192
run_one() {  # <endpoint> <instance_type> <run-dir>
  export ENDPOINT_NAME=$1 MODEL_ID=$M INSTANCE_TYPE=$2; R=$3; placed=""
  mkdir -p "$R"; env | grep -E '^(INSTANCE_|ENDPOINT_NAME|MODEL_ID|MAX_MODEL_LEN|TENSOR_PARALLEL_SIZE|SM_VLLM_)' | sort > "$R/run-env.txt"
  for region in $REGIONS; do
    export IMAGE_URI=763104351884.dkr.ecr.$region.amazonaws.com/$TAG
    if ./failover.sh "$R" "$region"; then placed=$region; break; fi
    last=$(ls -d "$R"/attempt-* | sort -V | tail -1)
    reason=$(python3 -c "import json;print(json.load(open('$last/status-final.json')).get('failure_reason',''))" 2>/dev/null)
    echo "$reason" | grep -qi capacity || { echo "STOP $1: non-capacity failure: ${reason:0:200}"; aws logs filter-log-events --region $region --log-group-name /aws/sagemaker/Endpoints/$1 --query 'events[].message' --output text 2>&1 | tr '\t' '\n' | grep -v '#015' > "$R/logs-failed.txt"; return 1; }
  done
  [ -z "$placed" ] && { echo "NO CAPACITY for $1"; return 1; }
  export AWS_REGION=$placed
  python3 sm.py status > "$R/status-inservice.json"
  echo "$(date -u +%T) measuring $1@$placed"
  python3 compare.py measure "$C" "$1@$placed" > "$C/measure-$1.log" 2>&1 && echo "measured $1" || echo "MEASURE FAILED $1: $(tail -3 "$C/measure-$1.log")"
  aws logs filter-log-events --region $placed --log-group-name /aws/sagemaker/Endpoints/$1 --query 'events[].message' --output text 2>&1 | tr '\t' '\n' | grep -v '#015' > "$R/logs-full.txt"
  python3 sm.py destroy > "$R/destroy.json" 2>&1; date -u +%FT%TZ > "$R/deleted-at.txt"
  until [ "$(python3 sm.py status | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')" = NotFound ]; do sleep 10; done
  echo "$(date -u +%T) deleted $1"
}
date -u +%FT%TZ > $C/start.txt; aws --version > $C/meta.txt 2>&1
run_one gemma-4-12b-repack-l4 ml.g6.xlarge docs/runs/2026-09-28-12b-repack-l4
run_one gemma-4-12b-repack ml.g6e.xlarge docs/runs/2026-09-28-12b-repack-l40s
G4=docs/runs/2026-09-28-12b-qat-l4/measure-gemma-4-12b-qat-l4.json
G40=docs/runs/2026-09-28-12b-qat-vs-bf16/measure-gemma-4-12b-qat.json
[ -f $C/measure-gemma-4-12b-repack-l4.json ] && python3 compare.py combine $C/compare-l4.json $G4 $C/measure-gemma-4-12b-repack-l4.json
[ -f $C/measure-gemma-4-12b-repack.json ] && python3 compare.py combine $C/compare-l40s.json $G40 $C/measure-gemma-4-12b-repack.json
echo "ALL DONE"

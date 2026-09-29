#!/bin/bash
# xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text (revision 315aec0): the E2B W4A16
# repack with the vision and audio towers removed, Gemma4ForCausalLM. On
# ml.g6.xlarge (1x L4) against the full E2B repack (same day) and Google E2B QAT.
# Standard settings; same container tag and compare.py.
cd /home/xbill/sagemaker-gemma
C=docs/runs/2026-09-29-e2b-text
TAG=vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1
REGIONS="us-east-2 us-west-2 us-east-1"
export INSTANCE_TYPE=ml.g6.xlarge INSTANCE_POOLS= TENSOR_PARALLEL_SIZE= MAX_MODEL_LEN=8192
run_one() {  # <endpoint> <model_id> <run-dir>
  export ENDPOINT_NAME=$1 MODEL_ID=$2; R=$3; placed=""
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
run_one gemma-4-e2b-text xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text docs/runs/2026-09-29-e2b-text-l4
F=docs/runs/2026-09-29-e2b-e4b-repack/measure-gemma-4-e2b-repack.json
G=docs/runs/2026-09-25-qat-vs-bf16/measure-gemma-4-e2b-qat.json
T=$C/measure-gemma-4-e2b-text.json
[ -f $T ] && python3 compare.py combine $C/compare-vs-repack.json $F $T
[ -f $T ] && python3 compare.py combine $C/compare-vs-google.json $G $T
echo "ALL DONE"

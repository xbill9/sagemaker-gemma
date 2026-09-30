#!/bin/bash
# L40S sweep for the repack write-up: every W4A16 variant not yet measured on
# ml.g6e.xlarge (1x L40S), one endpoint at a time, same day. Google -qat-w4a16-ct,
# xbill9 repack and xbill9 text-only for E2B and E4B; text-only for 26B, 31B and
# 12B (12B last: its gemma4_unified_text config type is new to these runs).
# Same container tag, standard settings and compare.py as every other run.
# A non-capacity failure is recorded and the queue moves on.
cd /home/xbill/sagemaker-gemma
C=docs/runs/2026-09-29-l40s-repack-sweep
TAG=vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1
REGIONS="us-east-2 us-west-2 us-east-1"
export INSTANCE_TYPE=ml.g6e.xlarge INSTANCE_POOLS= TENSOR_PARALLEL_SIZE= MAX_MODEL_LEN=8192
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
X=xbill9/gemma-4
run_one gemma-4-e2b-qat-l40s  google/gemma-4-E2B-it-qat-w4a16-ct       docs/runs/2026-09-29-l40s-e2b-qat
run_one gemma-4-e2b-rep-l40s  $X-E2B-it-qat-q4_0-w4a16-ct               docs/runs/2026-09-29-l40s-e2b-repack
run_one gemma-4-e2b-text-l40s $X-E2B-it-qat-q4_0-w4a16-ct-text          docs/runs/2026-09-29-l40s-e2b-text
run_one gemma-4-e4b-qat-l40s  google/gemma-4-E4B-it-qat-w4a16-ct       docs/runs/2026-09-29-l40s-e4b-qat
run_one gemma-4-e4b-rep-l40s  $X-E4B-it-qat-q4_0-w4a16-ct               docs/runs/2026-09-29-l40s-e4b-repack
run_one gemma-4-e4b-text-l40s $X-E4B-it-qat-q4_0-w4a16-ct-text          docs/runs/2026-09-29-l40s-e4b-text
run_one gemma-4-26b-text-l40s $X-26B-A4B-it-qat-q4_0-w4a16-ct-text      docs/runs/2026-09-29-l40s-26b-text
run_one gemma-4-31b-text-l40s $X-31B-it-qat-q4_0-w4a16-ct-text          docs/runs/2026-09-29-l40s-31b-text
run_one gemma-4-12b-text-l40s $X-12B-it-qat-q4_0-w4a16-ct-text          docs/runs/2026-09-29-l40s-12b-text
M() { echo $C/measure-$1.json; }
cmb() { [ -f "$2" ] && [ -f "$3" ] && python3 compare.py combine $C/compare-$1.json "$2" "$3"; }
cmb e2b-repack-vs-qat $(M gemma-4-e2b-qat-l40s) $(M gemma-4-e2b-rep-l40s)
cmb e2b-text-vs-repack $(M gemma-4-e2b-rep-l40s) $(M gemma-4-e2b-text-l40s)
cmb e4b-repack-vs-qat $(M gemma-4-e4b-qat-l40s) $(M gemma-4-e4b-rep-l40s)
cmb e4b-text-vs-repack $(M gemma-4-e4b-rep-l40s) $(M gemma-4-e4b-text-l40s)
cmb 26b-text-vs-repack docs/runs/2026-09-28-31b-26b-l40s/measure-gemma-4-26b-w4a16.json $(M gemma-4-26b-text-l40s)
cmb 31b-text-vs-qat docs/runs/2026-09-28-31b-26b-l40s/measure-gemma-4-31b-qat.json $(M gemma-4-31b-text-l40s)
cmb 12b-text-vs-repack docs/runs/2026-09-28-12b-repack/measure-gemma-4-12b-repack.json $(M gemma-4-12b-text-l40s)
echo "ALL DONE"

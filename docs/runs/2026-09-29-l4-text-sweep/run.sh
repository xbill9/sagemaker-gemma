#!/bin/bash
# L4 sweep for the repack write-up: the four xbill9 text-only builds not yet
# measured on the L4 (E4B, 26B, 31B, 12B), each against the sibling already
# measured there. ml.g6.xlarge (1x L4); 31B on ml.g6.2xlarge (the same L4, 32 GiB
# host RAM) with the reduced settings 31B QAT needed (memory 0.97, 4 sequences,
# 1024 batched tokens, 1024 context), minus the image-input limit: text-only has
# no vision part. 12B last: its gemma4_unified_text config type is new here.
# Same container tag, standard settings and compare.py as every other run.
# A non-capacity failure is recorded and the queue moves on.
cd /home/xbill/sagemaker-gemma
C=docs/runs/2026-09-29-l4-text-sweep
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
X=xbill9/gemma-4
run_one gemma-4-e4b-text-l4 $X-E4B-it-qat-q4_0-w4a16-ct-text docs/runs/2026-09-29-l4-e4b-text
run_one gemma-4-26b-text-l4 $X-26B-A4B-it-qat-q4_0-w4a16-ct-text docs/runs/2026-09-29-l4-26b-text
( export INSTANCE_TYPE=ml.g6.2xlarge MAX_MODEL_LEN=1024 SM_VLLM_GPU_MEMORY_UTILIZATION=0.97 SM_VLLM_MAX_NUM_SEQS=4 SM_VLLM_MAX_NUM_BATCHED_TOKENS=1024
  run_one gemma-4-31b-text-l4 $X-31B-it-qat-q4_0-w4a16-ct-text docs/runs/2026-09-29-l4-31b-text )
run_one gemma-4-12b-text-l4 $X-12B-it-qat-q4_0-w4a16-ct-text docs/runs/2026-09-29-l4-12b-text
M() { echo $C/measure-$1.json; }
cmb() { [ -f "$2" ] && [ -f "$3" ] && python3 compare.py combine $C/compare-$1.json "$2" "$3"; }
cmb e4b-text-vs-repack docs/runs/2026-09-29-e2b-e4b-repack/measure-gemma-4-e4b-repack.json $(M gemma-4-e4b-text-l4)
cmb 26b-text-vs-repack docs/runs/2026-09-28-l4-26b-31b/measure-gemma-4-26b-w4a16-l4.json $(M gemma-4-26b-text-l4)
cmb 31b-text-vs-qat docs/runs/2026-09-28-l4-26b-31b/measure-gemma-4-31b-qat-l4-2xl.json $(M gemma-4-31b-text-l4)
cmb 12b-text-vs-repack docs/runs/2026-09-28-12b-repack/measure-gemma-4-12b-repack-l4.json $(M gemma-4-12b-text-l4)
echo "ALL DONE"

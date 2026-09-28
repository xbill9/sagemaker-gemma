#!/bin/bash
# Gemma 4 31B three ways and the 26B A4B W4A16 repack, one endpoint at a time.
# Same container tag, settings and compare.py as the E2B, E4B and 12B runs.
# 26B repack, 31B QAT and 31B NVFP4 on ml.g6e.xlarge (1x L40S, the 12B
# instance); 31B bf16 (62.5 GB) only fits across 4 GPUs, so it runs last on
# ml.g6e.12xlarge (4x L40S) at tensor parallel 4.
# A region is abandoned only on a capacity failure; any other failure stops.
cd /home/xbill/sagemaker-gemma
C=docs/runs/2026-09-28-31b-26b-l40s
TAG=vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1
REGIONS="us-east-2 us-west-2 us-east-1"
export INSTANCE_POOLS=   # a single InstanceType; pools need 2+ entries
run_one() {  # <endpoint> <model_id> <instance_type> <tensor_parallel_size or ""> <run-dir>
  export ENDPOINT_NAME=$1 MODEL_ID=$2 INSTANCE_TYPE=$3 TENSOR_PARALLEL_SIZE=$4; R=$5; placed=""
  mkdir -p "$R"; env | grep -E '^(INSTANCE_|ENDPOINT_NAME|MODEL_ID|MAX_MODEL_LEN|TENSOR_PARALLEL_SIZE)' | sort > "$R/run-env.txt"
  for region in $REGIONS; do
    export IMAGE_URI=763104351884.dkr.ecr.$region.amazonaws.com/$TAG
    if ./failover.sh "$R" "$region"; then placed=$region; break; fi
    last=$(ls -d "$R"/attempt-* | sort -V | tail -1)
    reason=$(python3 -c "import json;print(json.load(open('$last/status-final.json')).get('failure_reason',''))" 2>/dev/null)
    echo "$reason" | grep -qi capacity || { echo "STOP $1: non-capacity failure: ${reason:0:200}"; AWS_REGION=$region python3 sm.py logs > "$R/logs-failed.txt" 2>&1; return 1; }
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
run_one gemma-4-26b-w4a16 xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct ml.g6e.xlarge "" docs/runs/2026-09-28-26b-w4a16
run_one gemma-4-31b-qat google/gemma-4-31B-it-qat-w4a16-ct ml.g6e.xlarge "" docs/runs/2026-09-28-31b-qat
run_one gemma-4-31b-nvfp4 nvidia/Gemma-4-31B-IT-NVFP4 ml.g6e.xlarge "" docs/runs/2026-09-28-31b-nvfp4
[ -f $C/measure-gemma-4-31b-qat.json ] && [ -f $C/measure-gemma-4-31b-nvfp4.json ] && \
  python3 compare.py combine $C/compare-31b-nvfp4-vs-qat.json $C/measure-gemma-4-31b-qat.json $C/measure-gemma-4-31b-nvfp4.json
run_one gemma-4-31b google/gemma-4-31B-it ml.g6e.12xlarge 4 docs/runs/2026-09-28-31b-bf16
[ -f $C/measure-gemma-4-31b.json ] && [ -f $C/measure-gemma-4-31b-qat.json ] && \
  python3 compare.py combine $C/compare-31b-qat-vs-bf16-tp4.json $C/measure-gemma-4-31b.json $C/measure-gemma-4-31b-qat.json
echo "ALL DONE"

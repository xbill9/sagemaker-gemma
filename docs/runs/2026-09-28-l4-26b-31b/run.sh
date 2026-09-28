#!/bin/bash
# The 26B A4B W4A16 repack and 31B QAT on ml.g6.xlarge (1x L4, 24 GB), the
# E2B/E4B/12B-QAT instance. Same container tag, settings and compare.py.
# 31B QAT tries max_model_len 8192 first; if vLLM reports the KV cache cannot
# hold one 8192-token request, it redeploys once at the maximum length vLLM
# estimates, and that one setting differs from the other rows.
cd /home/xbill/sagemaker-gemma
C=docs/runs/2026-09-28-l4-26b-31b
TAG=vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1
REGIONS="us-east-2 us-west-2 us-east-1"
export INSTANCE_TYPE=ml.g6.xlarge INSTANCE_POOLS= TENSOR_PARALLEL_SIZE=
# run_one <endpoint> <model_id> <run-dir> <max_model_len>; returns 2 on a non-capacity failure
run_one() {
  export ENDPOINT_NAME=$1 MODEL_ID=$2 MAX_MODEL_LEN=$4; R=$3; placed=""
  mkdir -p "$R"; env | grep -E '^(INSTANCE_|ENDPOINT_NAME|MODEL_ID|MAX_MODEL_LEN|TENSOR_PARALLEL_SIZE)' | sort > "$R/run-env.txt"
  for region in $REGIONS; do
    export IMAGE_URI=763104351884.dkr.ecr.$region.amazonaws.com/$TAG
    if ./failover.sh "$R" "$region"; then placed=$region; break; fi
    last=$(ls -d "$R"/attempt-* | sort -V | tail -1)
    reason=$(python3 -c "import json;print(json.load(open('$last/status-final.json')).get('failure_reason',''))" 2>/dev/null)
    echo "$reason" | grep -qi capacity || { echo "STOP $1: non-capacity failure: ${reason:0:200}"; AWS_REGION=$region python3 sm.py logs > "$R/logs-failed.txt" 2>&1; return 2; }
  done
  [ -z "$placed" ] && { echo "NO CAPACITY for $1"; return 1; }
  export AWS_REGION=$placed
  python3 sm.py status > "$R/status-inservice.json"
  echo "$(date -u +%T) measuring $1@$placed max_model_len=$4"
  python3 compare.py measure "$C" "$1@$placed" > "$C/measure-$1.log" 2>&1 && echo "measured $1" || echo "MEASURE FAILED $1: $(tail -3 "$C/measure-$1.log")"
  python3 sm.py logs > "$R/logs-tail.txt" 2>&1
  python3 sm.py destroy > "$R/destroy.json" 2>&1; date -u +%FT%TZ > "$R/deleted-at.txt"
  until [ "$(python3 sm.py status | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')" = NotFound ]; do sleep 10; done
  echo "$(date -u +%T) deleted $1"
}
date -u +%FT%TZ > $C/start.txt; aws --version > $C/meta.txt 2>&1
run_one gemma-4-26b-w4a16-l4 xbill9/gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct docs/runs/2026-09-28-l4-26b-w4a16 8192
R=docs/runs/2026-09-28-l4-31b-qat
run_one gemma-4-31b-qat-l4 google/gemma-4-31B-it-qat-w4a16-ct $R 8192
if [ $? -eq 2 ]; then
  est=$(grep -oE "estimated maximum model length is [0-9]+" $R/logs-failed.txt | grep -oE "[0-9]+$" | tail -1)
  if [ -n "$est" ] && [ "$est" -ge 2048 ]; then
    echo "31B QAT: vLLM estimates max length $est on the L4; redeploying at $est"
    run_one gemma-4-31b-qat-l4 google/gemma-4-31B-it-qat-w4a16-ct ${R}-len$est $est
  else
    echo "31B QAT: does not serve on one L4 (estimated max length: ${est:-none}); see $R/logs-failed.txt"
  fi
fi
echo "ALL DONE"

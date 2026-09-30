#!/bin/bash
# L4 format sweep: int4 embeddings (emb4, ple4), FP8 W8A8 and int8 W8A8 builds of the
# QAT weights across sizes, one endpoint at a time on ml.g6.xlarge (1x L4); 31B on
# ml.g6.2xlarge with the reduced settings. Each is compared with the text-only W4A16
# build of its size measured on the same instance today. A run waits until its
# repo's upload has finished (docs/runs/2026-09-29-publish/publish.log).
cd /home/xbill/sagemaker-gemma
C=docs/runs/2026-09-29-l4-format-sweep
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
PUB=docs/runs/2026-09-29-publish/publish.log
ready() { grep -q "xbill9/$1 anonymous HTTP 200" $PUB 2>/dev/null || [ "$2" = published ]; }
go() {  # <endpoint> <repo-name> <run-dir> [published]
  until ready "$2" "${4:-}"; do sleep 30; done
  run_one "$1" "xbill9/$2" "$3"
}
while pgrep -f "2026-09-29-e2b-w8a8-emb4/run.sh" >/dev/null; do sleep 30; done
date -u +%FT%TZ > $C/start.txt; aws --version > $C/meta.txt 2>&1
R=docs/runs/2026-09-29-l4
go gemma-4-e2b-ple4-l4    gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-ple4        $R-e2b-ple4
go gemma-4-e2b-fp8emb4-l4 gemma-4-E2B-it-qat-q4_0-fp8-text-emb4             $R-e2b-fp8emb4
go gemma-4-e4b-emb4-l4    gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-emb4        $R-e4b-emb4 published
go gemma-4-e4b-fp8-l4     gemma-4-E4B-it-qat-q4_0-fp8-text                  $R-e4b-fp8 published
go gemma-4-e4b-int8-l4    gemma-4-E4B-it-qat-w8a8-int8                      $R-e4b-int8
go gemma-4-12b-emb4-l4    gemma-4-12B-it-qat-q4_0-w4a16-ct-text-emb4        $R-12b-emb4
go gemma-4-12b-fp8-l4     gemma-4-12B-it-qat-q4_0-fp8-text                  $R-12b-fp8
go gemma-4-12b-int8-l4    gemma-4-12B-it-qat-w8a8-int8                      $R-12b-int8
go gemma-4-26b-emb4-l4    gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text-emb4    $R-26b-emb4
( export INSTANCE_TYPE=ml.g6.2xlarge MAX_MODEL_LEN=1024 SM_VLLM_GPU_MEMORY_UTILIZATION=0.97 SM_VLLM_MAX_NUM_SEQS=4 SM_VLLM_MAX_NUM_BATCHED_TOKENS=1024
  go gemma-4-31b-emb4-l4  gemma-4-31B-it-qat-q4_0-w4a16-ct-text-emb4        $R-31b-emb4 )
M() { echo $C/measure-$1.json; }
cmb() { [ -f "$2" ] && [ -f "$3" ] && python3 compare.py combine $C/compare-$1.json "$2" "$3"; }
TS=docs/runs/2026-09-29-l4-text-sweep
cmb e2b-ple4-vs-text   docs/runs/2026-09-29-e2b-text/measure-gemma-4-e2b-text.json $(M gemma-4-e2b-ple4-l4)
cmb e2b-ple4-vs-emb4   docs/runs/2026-09-29-e2b-emb4/measure-gemma-4-e2b-emb4-l4.json $(M gemma-4-e2b-ple4-l4)
cmb e2b-fp8emb4-vs-fp8 docs/runs/2026-09-29-e2b-8bit/measure-gemma-4-e2b-fp8-l4.json $(M gemma-4-e2b-fp8emb4-l4)
for f in emb4 fp8 int8; do cmb e4b-$f-vs-text $TS/measure-gemma-4-e4b-text-l4.json $(M gemma-4-e4b-$f-l4); done
for f in emb4 fp8 int8; do cmb 12b-$f-vs-text $TS/measure-gemma-4-12b-text-l4.json $(M gemma-4-12b-$f-l4); done
cmb 26b-emb4-vs-text $TS/measure-gemma-4-26b-text-l4.json $(M gemma-4-26b-emb4-l4)
cmb 31b-emb4-vs-text $TS/measure-gemma-4-31b-text-l4.json $(M gemma-4-31b-emb4-l4)
echo "ALL DONE"

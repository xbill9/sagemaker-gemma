#!/bin/bash
# vllm-project/vllm#39018 (head f7b6f506) on a T4: the same SageMaker sweep as
# 2026-09-30-t4-emb4-sweep, E2B and 12B only, with the image built from
# turing/pr39018 (the PR's two files on the AWS vLLM 0.30.0 image, none of
# patch_triton_turing.py). Every other setting is unchanged, so the comparison
# with the clamped runs isolates the fix.
cd /home/xbill/sagemaker-gemma
C=docs/runs/2026-09-30-t4-pr39018
REGIONS="us-east-2"
export IMAGE_URI=106059658660.dkr.ecr.us-east-2.amazonaws.com/sagemaker-gemma-vllm:0.30.0-sagemaker-v1.3-pr39018
export INSTANCE_TYPE=ml.g4dn.xlarge INSTANCE_POOLS=ml.g4dn.xlarge,ml.g4dn.2xlarge TENSOR_PARALLEL_SIZE= MAX_MODEL_LEN=8192
export SM_VLLM_DTYPE=float16
# The cu130 image needs NVIDIA driver 580. Without this the first E2B deploy
# (e2b/attempt-1) ended CannotStartContainerError with no container log.
export INFERENCE_AMI_VERSION=al2023-ami-sagemaker-inference-gpu-4-1
run_one() {  # <endpoint> <model_id> <run-dir>
  export ENDPOINT_NAME=$1 MODEL_ID=$2; R=$3; placed=""
  mkdir -p "$R"; env | grep -E '^(INSTANCE_|ENDPOINT_NAME|MODEL_ID|MAX_MODEL_LEN|TENSOR_PARALLEL_SIZE|IMAGE_URI|INFERENCE_AMI_VERSION|SM_VLLM_)' | sort > "$R/run-env.txt"
  for region in $REGIONS; do
    if ./failover.sh "$R" "$region"; then placed=$region; break; fi
    last=$(ls -d "$R"/attempt-* | sort -V | tail -1)
    reason=$(python3 -c "import json;print(json.load(open('$last/status-final.json')).get('failure_reason',''))" 2>/dev/null)
    aws logs filter-log-events --region $region --log-group-name /aws/sagemaker/Endpoints/$1 --query 'events[].message' --output text 2>&1 | tr '\t' '\n' | grep -v '#015' > "$R/logs-failed.txt"
    python3 sm.py destroy > "$R/destroy.json" 2>&1
    echo "$reason" | grep -qi capacity || { echo "STOP $1: non-capacity failure: ${reason:0:200}"; return 1; }
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
run_one gemma-4-e2b-emb4-t4-pr $X-E2B-it-qat-q4_0-w4a16-ct-text-emb4 $C/e2b
run_one gemma-4-12b-emb4-t4-pr $X-12B-it-qat-q4_0-w4a16-ct-text-emb4 $C/12b
P=docs/runs/2026-09-30-t4-emb4-sweep
cmb() { [ -f "$2" ] && [ -f "$3" ] && python3 compare.py combine $C/compare-$1.json "$2" "$3"; }
cmb e2b-clamp-vs-pr $P/measure-gemma-4-e2b-emb4-t4.json $C/measure-gemma-4-e2b-emb4-t4-pr.json
cmb 12b-clamp-vs-pr $P/measure-gemma-4-12b-emb4-t4.json $C/measure-gemma-4-12b-emb4-t4-pr.json
echo "ALL DONE"

#!/bin/bash
# 31B QAT on one L4 with the memory settings squeezed. At the standard settings
# the 19.77 GiB of weights exceed vLLM's 0.9 memory cap on the L4's 21.96 GiB and
# start-up runs out of memory (../2026-09-28-l4-31b-qat/logs-oom.txt). Here:
# memory cap 0.97, at most 4 sequences and 1024 batched tokens per step, and
# max_model_len 1024 (the measurement's prompts are ~30 tokens, replies <= 512),
# and image, video and audio input off: the measurement is text only, and with
# them on vLLM needs 2496 batched tokens for one image and refuses 1024.
# On ml.g6.2xlarge: the same single L4 with 32 GiB of host RAM instead of 16;
# on ml.g6.xlarge mapping the 23.3 GB checkpoint failed with ENOMEM
# (../2026-09-28-l4-31b-qat-textonly/).
cd /home/xbill/sagemaker-gemma
while pgrep -f "2026-09-28-l4-26b-31b/run-31b-textonly.sh" >/dev/null; do sleep 30; done
C=docs/runs/2026-09-28-l4-26b-31b
TAG=vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1
export ENDPOINT_NAME=gemma-4-31b-qat-l4-2xl MODEL_ID=google/gemma-4-31B-it-qat-w4a16-ct INSTANCE_TYPE=ml.g6.2xlarge INSTANCE_POOLS= TENSOR_PARALLEL_SIZE= MAX_MODEL_LEN=1024
export SM_VLLM_GPU_MEMORY_UTILIZATION=0.97 SM_VLLM_MAX_NUM_SEQS=4 SM_VLLM_MAX_NUM_BATCHED_TOKENS=1024 SM_VLLM_LIMIT_MM_PER_PROMPT='{"image":0,"video":0,"audio":0}'
R=docs/runs/2026-09-28-l4-31b-qat-2xl; mkdir -p $R; placed=""
env | grep -E '^(INSTANCE_|ENDPOINT_NAME|MODEL_ID|MAX_MODEL_LEN|TENSOR_PARALLEL_SIZE|SM_VLLM_)' | sort > "$R/run-env.txt"
for region in us-east-2 us-west-2 us-east-1; do
  export IMAGE_URI=763104351884.dkr.ecr.$region.amazonaws.com/$TAG
  if ./failover.sh "$R" "$region"; then placed=$region; break; fi
  last=$(ls -d "$R"/attempt-* | sort -V | tail -1)
  reason=$(python3 -c "import json;print(json.load(open('$last/status-final.json')).get('failure_reason',''))" 2>/dev/null)
  echo "$reason" | grep -qi capacity || { echo "STOP: non-capacity failure: ${reason:0:200}"; aws logs filter-log-events --region $region --log-group-name /aws/sagemaker/Endpoints/$ENDPOINT_NAME --query 'events[].message' --output text 2>&1 | tr '\t' '\n' | grep -v '#015' > "$R/logs-failed.txt"; break; }
done
if [ -n "$placed" ]; then
  export AWS_REGION=$placed
  python3 sm.py status > "$R/status-inservice.json"
  echo "$(date -u +%T) measuring $ENDPOINT_NAME@$placed"
  python3 compare.py measure "$C" "$ENDPOINT_NAME@$placed" > "$C/measure-$ENDPOINT_NAME.log" 2>&1 && echo "measured $ENDPOINT_NAME" || echo "MEASURE FAILED: $(tail -3 "$C/measure-$ENDPOINT_NAME.log")"
  aws logs filter-log-events --region $placed --log-group-name /aws/sagemaker/Endpoints/$ENDPOINT_NAME --query 'events[].message' --output text 2>&1 | tr '\t' '\n' | grep -v '#015' > "$R/logs-full.txt"
fi
for region in us-east-2 us-west-2 us-east-1; do
  echo "$(date -u +%FT%TZ) cleanup $region: $(AWS_REGION=$region python3 sm.py destroy 2>&1 | tr -d '\n' | cut -c1-200)"
done
date -u +%FT%TZ > "$R/deleted-at.txt"
echo "ALL DONE"

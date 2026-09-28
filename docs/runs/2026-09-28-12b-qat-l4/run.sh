#!/bin/bash
# Gemma 4 12B QAT on ml.g6.xlarge (1x L4), the E2B/E4B instance, to set beside
# the 2026-09-28 L40S run of the same checkpoint. Starts after the 26B/31B
# queue finishes. Same container tag, settings and compare.py.
cd /home/xbill/sagemaker-gemma
while pgrep -f "2026-09-28-31b-26b-l40s/run.sh" >/dev/null; do sleep 30; done
C=docs/runs/2026-09-28-12b-qat-l4
TAG=vllm:0.30.0-gpu-py312-cu130-ubuntu24.04-sagemaker-v1.1
export ENDPOINT_NAME=gemma-4-12b-qat-l4 MODEL_ID=google/gemma-4-12B-it-qat-w4a16-ct INSTANCE_TYPE=ml.g6.xlarge INSTANCE_POOLS= TENSOR_PARALLEL_SIZE=
R=$C/deploy; mkdir -p $R; placed=""
env | grep -E '^(INSTANCE_|ENDPOINT_NAME|MODEL_ID|MAX_MODEL_LEN|TENSOR_PARALLEL_SIZE)' | sort > "$R/run-env.txt"
date -u +%FT%TZ > $C/start.txt
for region in us-east-2 us-west-2 us-east-1; do
  export IMAGE_URI=763104351884.dkr.ecr.$region.amazonaws.com/$TAG
  if ./failover.sh "$R" "$region"; then placed=$region; break; fi
  last=$(ls -d "$R"/attempt-* | sort -V | tail -1)
  reason=$(python3 -c "import json;print(json.load(open('$last/status-final.json')).get('failure_reason',''))" 2>/dev/null)
  echo "$reason" | grep -qi capacity || { echo "STOP: non-capacity failure: ${reason:0:200}"; AWS_REGION=$region python3 sm.py logs > "$R/logs-failed.txt" 2>&1; break; }
done
if [ -n "$placed" ]; then
  export AWS_REGION=$placed
  python3 sm.py status > "$R/status-inservice.json"
  echo "$(date -u +%T) measuring $ENDPOINT_NAME@$placed"
  python3 compare.py measure "$C" "$ENDPOINT_NAME@$placed" > "$C/measure-$ENDPOINT_NAME.log" 2>&1 && echo "measured $ENDPOINT_NAME" || echo "MEASURE FAILED: $(tail -3 "$C/measure-$ENDPOINT_NAME.log")"
  python3 sm.py logs > "$R/logs-tail.txt" 2>&1
fi
for region in us-east-2 us-west-2 us-east-1; do
  echo "$(date -u +%FT%TZ) cleanup $region: $(AWS_REGION=$region python3 sm.py destroy 2>&1 | tr -d '\n' | cut -c1-200)"
done
date -u +%FT%TZ > "$R/deleted-at.txt"
L=docs/runs/2026-09-28-12b-qat-vs-bf16/measure-gemma-4-12b-qat.json
[ -f $C/measure-$ENDPOINT_NAME.json ] && python3 compare.py combine $C/compare-l4-vs-l40s.json $L $C/measure-$ENDPOINT_NAME.json
echo "ALL DONE"

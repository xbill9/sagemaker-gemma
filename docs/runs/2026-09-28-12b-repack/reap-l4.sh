#!/bin/bash
# The queue was stopped after the L4 container failed on a config error; this
# waits for the Creating endpoint to reach a terminal state, saves the log, and
# deletes it (SageMaker refuses to delete an endpoint while it is Creating).
cd /home/xbill/sagemaker-gemma
export AWS_REGION=us-east-2 ENDPOINT_NAME=gemma-4-12b-repack-l4
R=docs/runs/2026-09-28-12b-repack-l4
while :; do
  st=$(python3 sm.py status | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')
  case $st in Creating) sleep 30;; *) break;; esac
done
echo "$(date -u +%FT%TZ) $st" >> $R/attempt-1/timeline.txt
aws logs filter-log-events --region us-east-2 --log-group-name /aws/sagemaker/Endpoints/$ENDPOINT_NAME --query 'events[].message' --output text 2>&1 | tr '\t' '\n' | grep -v '#015' > $R/logs-failed.txt
for region in us-east-2 us-west-2 us-east-1; do
  for ep in gemma-4-12b-repack-l4 gemma-4-12b-repack; do
    echo "$(date -u +%FT%TZ) $region $ep: $(AWS_REGION=$region ENDPOINT_NAME=$ep python3 sm.py destroy 2>&1 | tr -d '\n' | cut -c1-160)"
  done
done
date -u +%FT%TZ > $R/deleted-at.txt
echo "REAPED"

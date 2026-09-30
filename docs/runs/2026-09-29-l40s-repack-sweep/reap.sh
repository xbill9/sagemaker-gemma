#!/bin/bash
# The sweep was stopped at the user's request (L40, not L40S). Wait for the one
# endpoint it created to leave Creating, then delete everything the sweep could
# have created, in every region.
cd /home/xbill/sagemaker-gemma
export AWS_REGION=us-east-2 ENDPOINT_NAME=gemma-4-e2b-qat-l40s
while [ "$(python3 sm.py status | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')" = Creating ]; do sleep 30; done
for region in us-east-2 us-west-2 us-east-1; do
  for ep in gemma-4-e2b-qat-l40s gemma-4-e2b-rep-l40s gemma-4-e2b-text-l40s gemma-4-e4b-qat-l40s gemma-4-e4b-rep-l40s gemma-4-e4b-text-l40s gemma-4-26b-text-l40s gemma-4-31b-text-l40s gemma-4-12b-text-l40s; do
    echo "$(date -u +%FT%TZ) $region $ep: $(AWS_REGION=$region ENDPOINT_NAME=$ep python3 sm.py destroy 2>&1 | tr -d '\n' | cut -c1-160)"
  done
done
echo "REAPED"

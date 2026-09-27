#!/bin/bash
# If run.sh exits (or is killed) with the QAT endpoint still up, delete it.
cd /home/xbill/sagemaker-gemma
export AWS_REGION=us-east-2 ENDPOINT_NAME=gemma-4-e4b-qat
while kill -0 20173 2>/dev/null; do sleep 30; done
st=$(python3 sm.py status | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')
echo "$(date -u +%FT%TZ) run.sh gone, endpoint $st"
[ "$st" != NotFound ] && python3 sm.py destroy && echo "$(date -u +%FT%TZ) watchdog deleted $ENDPOINT_NAME"

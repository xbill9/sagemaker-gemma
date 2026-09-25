#!/bin/bash
# Wait for the re-deployed full-size endpoint, measure it, delete it.
cd /home/xbill/sagemaker-gemma
R=docs/runs/2026-09-25-qat-vs-bf16/rerun-bf16
export AWS_REGION=us-east-2 ENDPOINT_NAME=gemma-4-e2b
last=""
while :; do
  st=$(python3 sm.py status | python3 -c "import json,sys;print(json.load(sys.stdin).get('status'))" 2>/dev/null || echo ERR)
  [ "$st" != "$last" ] && { echo "$(date -u +%FT%TZ) $st" | tee -a $R/timeline.txt; last=$st; }
  case $st in InService|Failed|NotFound) break;; esac
  sleep 30
done
python3 sm.py status > $R/status-final.json
if [ "$st" = InService ]; then
  echo "measuring"
  python3 compare.py measure $R gemma-4-e2b@us-east-2 > $R/measure.log 2>&1 && echo "measured" || echo "MEASURE FAILED: $(tail -3 $R/measure.log)"
fi
python3 sm.py destroy > $R/destroy.json 2>&1; date -u +%FT%TZ > $R/deleted-at.txt
echo "DONE $st, endpoint deleted"

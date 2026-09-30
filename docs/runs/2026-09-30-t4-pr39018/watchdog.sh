#!/bin/bash
# When run.sh exits (or is killed), delete anything it left. An endpoint still
# Creating cannot be deleted, so retry each one until it is gone.
cd /home/xbill/sagemaker-gemma
while pgrep -f "2026-09-30-t4-pr39018/run.sh" >/dev/null; do sleep 30; done
for ep in gemma-4-e2b-emb4-t4-pr gemma-4-12b-emb4-t4-pr; do
  export AWS_REGION=us-east-2 ENDPOINT_NAME=$ep
  until [ "$(python3 sm.py status | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')" = NotFound ]; do
    echo "$(date -u +%FT%TZ) $ep: $(python3 sm.py destroy 2>&1 | tr -d '\n' | cut -c1-160)"; sleep 30
  done
  echo "$(date -u +%FT%TZ) $ep: gone"
done
echo "$(date -u +%FT%TZ) watchdog done"

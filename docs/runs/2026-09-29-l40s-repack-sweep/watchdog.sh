#!/bin/bash
# When run.sh exits (or is killed), delete anything it left, in every region.
cd /home/xbill/sagemaker-gemma
while pgrep -f "2026-09-29-l40s-repack-sweep/run.sh" >/dev/null; do sleep 30; done
for region in us-east-2 us-west-2 us-east-1; do
  for ep in gemma-4-e2b-qat-l40s gemma-4-e2b-rep-l40s gemma-4-e2b-text-l40s gemma-4-e4b-qat-l40s gemma-4-e4b-rep-l40s gemma-4-e4b-text-l40s gemma-4-26b-text-l40s gemma-4-31b-text-l40s gemma-4-12b-text-l40s; do
    echo "$(date -u +%FT%TZ) $region $ep: $(AWS_REGION=$region ENDPOINT_NAME=$ep python3 sm.py destroy 2>&1 | tr -d '\n' | cut -c1-160)"
  done
done
echo "$(date -u +%FT%TZ) watchdog done"

#!/bin/bash
# When run.sh exits (or is killed), delete any 12B endpoint it left up, in every region.
cd /home/xbill/sagemaker-gemma
while pgrep -f "2026-09-28-12b-qat-vs-bf16/run.sh" >/dev/null; do sleep 30; done
for region in us-east-2 us-west-2 us-east-1; do for ep in gemma-4-12b gemma-4-12b-qat; do
  # destroy also removes a model or endpoint config left without an endpoint.
  echo "$(date -u +%FT%TZ) $region $ep: $(AWS_REGION=$region ENDPOINT_NAME=$ep python3 sm.py destroy 2>&1 | tr -d '\n' | cut -c1-200)"
done; done
echo "$(date -u +%FT%TZ) watchdog done"

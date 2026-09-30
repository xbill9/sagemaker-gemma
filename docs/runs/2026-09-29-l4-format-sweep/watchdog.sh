#!/bin/bash
# When run.sh exits (or is killed), delete anything it left, in every region.
cd /home/xbill/sagemaker-gemma
while pgrep -f "2026-09-29-l4-format-sweep/run.sh" >/dev/null; do sleep 30; done
for region in us-east-2 us-west-2 us-east-1; do
  for ep in gemma-4-e2b-ple4-l4 gemma-4-e2b-fp8emb4-l4 gemma-4-e4b-emb4-l4 gemma-4-e4b-fp8-l4 gemma-4-e4b-int8-l4 gemma-4-12b-emb4-l4 gemma-4-12b-fp8-l4 gemma-4-12b-int8-l4 gemma-4-26b-emb4-l4 gemma-4-31b-emb4-l4; do
    echo "$(date -u +%FT%TZ) $region $ep: $(AWS_REGION=$region ENDPOINT_NAME=$ep python3 sm.py destroy 2>&1 | tr -d '\n' | cut -c1-160)"
  done
done
echo "$(date -u +%FT%TZ) watchdog done"

#!/bin/bash
# After reap-l4.sh deletes the endpoint that failed on the quantization ignore
# list, archive that attempt and rerun run.sh (L4 then L40S) on the fixed config
# (Hugging Face commit 2a96908).
cd /home/xbill/sagemaker-gemma
C=docs/runs/2026-09-28-12b-repack
until grep -q REAPED $C/reap-l4.log 2>/dev/null; do sleep 20; done
mv docs/runs/2026-09-28-12b-repack-l4 docs/runs/2026-09-28-12b-repack-l4-ignore-error
mkdir -p $C/ignore-error && mv $C/run.log $C/watchdog.log $C/reap-l4.log $C/start.txt $C/ignore-error/
nohup $C/watchdog.sh > $C/watchdog.log 2>&1 &
$C/run.sh > $C/run.log 2>&1

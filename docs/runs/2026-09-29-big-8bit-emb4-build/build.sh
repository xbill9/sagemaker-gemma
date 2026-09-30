#!/bin/bash
# Build the 12B/26B/31B variants for the L4 sweep: emb4 at all three sizes, and
# 12B int8 and FP8 (26B and 31B at 8 bits do not fit one L4).
set -u
M=~/models; S=/home/xbill/sagemaker-gemma/docs/runs/2026-09-29-l40s-repack-sweep; E=~/gemma4-dev/gpu-vllm-t4-2b-w4a16/repack/embed_int4.py
cd /home/xbill/sagemaker-gemma
for s in 12B 26B-A4B 31B; do
  echo "$(date -u +%FT%TZ) emb4 $s start"
  python3 $E $M/gemma-4-$s-it-qat-q4_0-w4a16-ct-text $M/gemma-4-$s-it-qat-q4_0-w4a16-ct-text-emb4 --embed-tokens > $B_DIR/emb4-$s.log 2>&1; echo "$(date -u +%FT%TZ) emb4 $s exit $?"
done
echo "$(date -u +%FT%TZ) int8 12B start"
python3 ~/gemma4-dev/jev-tpu-31b/w8a8_from_qat.py $M/gemma-4-12B-it-qat-q4_0-unquantized $M/gemma-4-12B-it-qat-w8a8-int8 > $B_DIR/int8-12B.log 2>&1; echo "$(date -u +%FT%TZ) int8 12B exit $?"
echo "$(date -u +%FT%TZ) fp8 12B start"
python3 tools/fp8_text.py build $M/gemma-4-12B-it-qat-q4_0-unquantized $S/config-text-12B.json $S/index-text-12B.json $M/gemma-4-12B-it-qat-q4_0-fp8-text > $B_DIR/fp8-12B.log 2>&1 && \
python3 tools/fp8_text.py verify $M/gemma-4-12B-it-qat-q4_0-unquantized $M/gemma-4-12B-it-qat-q4_0-fp8-text >> $B_DIR/fp8-12B.log 2>&1; echo "$(date -u +%FT%TZ) fp8 12B exit $?"
df -h ~ | tail -1
echo "BUILDS DONE"

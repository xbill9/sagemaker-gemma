#!/bin/bash
# Build the three E4B text-only variants for the L4 sweep, one after another.
set -u
M=~/models; S=/home/xbill/sagemaker-gemma/docs/runs/2026-09-29-l40s-repack-sweep
cd /home/xbill/sagemaker-gemma
echo "$(date -u +%FT%TZ) emb4 start"
python3 ~/gemma4-dev/gpu-vllm-t4-2b-w4a16/repack/embed_int4.py $M/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text $M/gemma-4-E4B-it-qat-q4_0-w4a16-ct-text-emb4 --embed-tokens > $B_DIR/emb4.log 2>&1; echo "$(date -u +%FT%TZ) emb4 exit $?"
echo "$(date -u +%FT%TZ) int8 start"
python3 ~/gemma4-dev/jev-tpu-31b/w8a8_from_qat.py $M/gemma-4-E4B-it-qat-q4_0-unquantized $M/gemma-4-E4B-it-qat-w8a8-int8 > $B_DIR/int8.log 2>&1; echo "$(date -u +%FT%TZ) int8 exit $?"
echo "$(date -u +%FT%TZ) fp8 start"
python3 tools/fp8_text.py build $M/gemma-4-E4B-it-qat-q4_0-unquantized $S/config-text-E4B.json $S/index-text-E4B.json $M/gemma-4-E4B-it-qat-q4_0-fp8-text > $B_DIR/fp8.log 2>&1 && \
python3 tools/fp8_text.py verify $M/gemma-4-E4B-it-qat-q4_0-unquantized $M/gemma-4-E4B-it-qat-q4_0-fp8-text >> $B_DIR/fp8.log 2>&1; echo "$(date -u +%FT%TZ) fp8 exit $?"
echo "BUILDS DONE"

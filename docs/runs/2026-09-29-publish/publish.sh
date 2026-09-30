#!/bin/bash
# Publish the eight builds for the L4 sweep (public), smallest first.
for d in gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-ple4 gemma-4-E2B-it-qat-q4_0-fp8-text-emb4 gemma-4-12B-it-qat-q4_0-w4a16-ct-text-emb4 gemma-4-E4B-it-qat-w8a8-int8 gemma-4-12B-it-qat-w8a8-int8 gemma-4-12B-it-qat-q4_0-fp8-text gemma-4-26B-A4B-it-qat-q4_0-w4a16-ct-text-emb4 gemma-4-31B-it-qat-q4_0-w4a16-ct-text-emb4; do
  R=xbill9/$d
  echo "$(date -u +%FT%TZ) $R create: $(hf repo create $R --repo-type model 2>&1 | tail -1)"
  echo "$(date -u +%FT%TZ) $R upload: $(hf upload $R ~/models/$d . --commit-message "$(head -12 ~/models/$d/README.md | grep '^# ' | sed 's/^# //')" 2>&1 | tail -1)"
  echo "$(date -u +%FT%TZ) $R anonymous HTTP $(curl -s -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/$R)"
done
echo "PUBLISH DONE"

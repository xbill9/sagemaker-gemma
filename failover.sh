#!/bin/bash
# Deploy to the first region with capacity. For each region in turn: deploy,
# wait for InService or Failed, and on Failed tear down and move on.
#
#   ./failover.sh <run-dir> [--watch-first] region1 region2 ...
#
# --watch-first: an endpoint is already Creating in region1; watch it instead
# of deploying. Each attempt gets <run-dir>/attempt-N/ with a timeline.
# Prints one line per status change and vLLM start-up milestone.
cd "$(dirname "$0")"
# sm.py reads .env itself.
RUN=$1; shift
WATCH_FIRST=0; [ "$1" = --watch-first ] && { WATCH_FIRST=1; shift; }
n=$(ls -d "$RUN"/attempt-* 2>/dev/null | wc -l)
[ $WATCH_FIRST = 1 ] || n=$((n + 1))
py=python3

for region in "$@"; do
  export AWS_REGION=$region
  A="$RUN/attempt-$n"; mkdir -p "$A"; echo "region=$region" > "$A/region.txt"
  if [ $WATCH_FIRST = 1 ]; then
    WATCH_FIRST=0
  else
    grep -v '^#' .env | sed "s/^AWS_REGION=.*/AWS_REGION=$region/" > "$A/env.txt"
    date -u +%FT%TZ > "$A/deploy-start.txt"
    $py sm.py deploy > "$A/deploy.json" 2>&1 || { echo "attempt-$n $region deploy call failed: $(tail -1 "$A/deploy.json")"; n=$((n + 1)); continue; }
    echo "attempt-$n $region deploy submitted"
  fi
  last=""; seen=""
  while :; do
    s=$($py sm.py status 2>&1)
    st=$(echo "$s" | python3 -c "import json,sys;print(json.load(sys.stdin).get('status'))" 2>/dev/null || echo ERR)
    if [ "$st" != "$last" ]; then echo "$(date -u +%FT%TZ) $st" >> "$A/timeline.txt"; echo "attempt-$n $region $(date -u +%T) $st"; last=$st; fi
    out=$($py sm.py logs 2>&1 | grep -E "vLLM API server version|Model loading took|KV cache size|Maximum concurrency|Application startup complete|Traceback|CUDA out of memory|ERROR" | cut -c1-240 || true)
    new=$(comm -13 <(echo "$seen" | sort -u) <(echo "$out" | sort -u)); [ -n "$new" ] && echo "$new" | sed "s/^/attempt-$n log: /"
    seen="$seen
$out"
    case $st in InService|Failed|NotFound) break;; esac
    sleep 30
  done
  echo "$s" > "$A/status-final.json"
  if [ "$st" = InService ]; then
    sed -i "s/^AWS_REGION=.*/AWS_REGION=$region/" .env
    echo "READY attempt-$n $region (.env AWS_REGION updated)"
    exit 0
  fi
  echo "attempt-$n $region ended $st: $(echo "$s" | python3 -c "import json,sys;print(json.load(sys.stdin).get('failure_reason','')[:160])" 2>/dev/null)"
  $py sm.py destroy > "$A/destroy.json" 2>&1
  until [ "$($py sm.py status | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')" = NotFound ]; do sleep 10; done
  n=$((n + 1))
done
echo "NO CAPACITY in: $*"
exit 1

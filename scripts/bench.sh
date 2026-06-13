#!/bin/bash
# Curio benchmark protocol: fixed 6-game set, 3 seeds, equal budgets.
# Usage: scripts/bench.sh [LABEL] [STEPS]
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL="${1:-current}"
STEPS="${2:-4000}"
GAMES="lp85,vc33,ls20,ft09,tr87,cn04"
echo "=== $LABEL | games=$GAMES | steps=$STEPS ==="
TOTAL=0
for S in 0 7 42; do
  OUT=$(CURIO_SEED=$S make play-local GAME=$GAMES STEPS=$STEPS 2>&1 | grep -A 8 "SUMMARY")
  LV=$(echo "$OUT" | awk '/levels=/{gsub(/levels=/,"");sum+=$2} END{print sum+0}')
  WINS=$(echo "$OUT" | awk '/levels=/{gsub(/levels=/,""); if($2>0) printf "%s:%s ", $1, $2}')
  echo "seed=$S levels=$LV  [${WINS:-none}]"
  TOTAL=$((TOTAL + LV))
done
echo "--- mean levels/seed: $(echo "scale=2; $TOTAL/3" | bc)"

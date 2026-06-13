#!/usr/bin/env bash
# bench_held.sh — honest scoreboard over the HELD-18 proxy set.
#
# Runs the Curio agent across all 18 held-out games (never tuned on) for
# three seeds, printing each game's per-seed level count and the aggregate
# scorecard score per seed plus a mean. The held set is the honest proxy
# for generalization; the FIT set is reverse-engineered and excluded here.
#
# Usage:
#   scripts/bench_held.sh                 # full agent, seeds 0 1 2
#   CURIO_GENERIC_ONLY=1 scripts/bench_held.sh   # generic-core ablation
#   STEPS=2000 SEEDS="0 1" scripts/bench_held.sh # override budget/seeds
#
# Honors CURIO_GENERIC_ONLY (passed straight through to the agent) so the
# same harness measures both the full agent and the ablated generic core.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

HELD="tu93 ar25 re86 su15 m0r0 sc25 sp80 ka59 g50t sb26 lf52 bp35 s5i5 r11l sk48 wa30 cd82 tn36"
CSV="$(echo "$HELD" | tr ' ' ',')"

STEPS="${STEPS:-4000}"
SEEDS="${SEEDS:-0 1 2}"

echo "================================================================"
echo "HELD-18 honest scoreboard"
echo "  games:  $HELD"
echo "  steps:  $STEPS   seeds: $SEEDS"
echo "  CURIO_GENERIC_ONLY=${CURIO_GENERIC_ONLY:-<unset>}"
echo "================================================================"

agg_sum=0
agg_n=0

for s in $SEEDS; do
  echo
  echo "---- seed $s ----------------------------------------------------"
  out="$(CURIO_SEED="$s" make play-local GAME="$CSV" STEPS="$STEPS" 2>&1)"

  # per-game levels lines look like:  ar25     levels=  3  actions=...
  echo "$out" | grep -E "levels=" || true

  agg="$(echo "$out" | grep -E "Aggregate scorecard score" \
         | grep -oE '[0-9]+(\.[0-9]+)?' | tail -1)"
  echo "  seed $s aggregate scorecard: ${agg:-NA}"

  if [[ -n "${agg:-}" ]]; then
    agg_sum="$(python3 -c "print($agg_sum + $agg)")"
    agg_n=$((agg_n + 1))
  fi
done

echo
echo "================================================================"
if [[ "$agg_n" -gt 0 ]]; then
  mean="$(python3 -c "print(round($agg_sum / $agg_n, 4))")"
  echo "HELD-18 mean aggregate scorecard over $agg_n seed(s): $mean"
else
  echo "HELD-18 mean aggregate scorecard: NA (no aggregate lines parsed)"
fi
echo "================================================================"

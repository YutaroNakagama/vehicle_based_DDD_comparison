#!/bin/bash
# Auto-resubmit Lstm CPU-hedge jobs as queue slots free up.
# Default LIMIT=50 to gate scaling until we measure CPU runtime.
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SUBMITTER="$PROJECT_ROOT/scripts/hpc/launchers/submit_lstm_cpu_hedge.sh"
cd "$PROJECT_ROOT"

LIMIT="${1:-50}"
INTERVAL_SEC=900
MAX_PASSES=200

pass=0
while true; do
    pass=$((pass + 1))
    active=$(qstat -u s2240011 2>/dev/null | awk 'NR>5 && $10 != "C"' | wc -l)
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pass $pass | active=$active | hedge_limit=$LIMIT"
    bash "$SUBMITTER" "$LIMIT" 2>&1 | tail -3

    if [[ "$pass" -ge "$MAX_PASSES" ]]; then break; fi
    sleep "$INTERVAL_SEC"
done

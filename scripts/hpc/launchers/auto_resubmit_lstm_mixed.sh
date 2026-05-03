#!/bin/bash
# Auto-resubmit Lstm mixed-domain GPU jobs as queue slots free up.
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SUBMITTER="$PROJECT_ROOT/scripts/hpc/launchers/submit_lstm_mixed_seeds.sh"
cd "$PROJECT_ROOT"

INTERVAL_SEC=900
MAX_PASSES=200

pass=0
while true; do
    pass=$((pass + 1))
    active=$(qstat -u s2240011 2>/dev/null | awk 'NR>5 && $10 != "C"' | wc -l)

    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pass $pass | active=$active"

    n_sub=$(bash "$SUBMITTER" 2>&1 | tee /dev/stderr | grep -c "^\[SUB\]" || true)
    echo "[INFO] Submitted $n_sub new Lstm-mixed jobs this pass."

    if [[ "$pass" -ge "$MAX_PASSES" ]]; then break; fi
    sleep "$INTERVAL_SEC"
done

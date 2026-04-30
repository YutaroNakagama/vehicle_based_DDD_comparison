#!/bin/bash
# Auto-resubmit Lstm mixed-domain CPU jobs (Lm_*) as queue slots free up.
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SUBMITTER="$PROJECT_ROOT/scripts/hpc/launchers/submit_lstm_mixed_cpu.sh"
cd "$PROJECT_ROOT"

INTERVAL_SEC=900
MAX_PASSES=200

pass=0
while true; do
    pass=$((pass + 1))
    active=$(qstat -u s2240011 2>/dev/null | awk 'NR>5 && $10 != "C"' | wc -l)
    remaining=$(bash "$SUBMITTER" --dry-run 2>/dev/null | grep -c "^\[DRY\]")

    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pass $pass | active=$active | lstm_mixed_cpu_remaining=$remaining"

    if [[ "$remaining" -eq 0 ]]; then
        echo "[INFO] No Lstm-mixed-CPU work remaining — exiting."
        break
    fi
    bash "$SUBMITTER" 2>&1 | tail -3

    if [[ "$pass" -ge "$MAX_PASSES" ]]; then break; fi
    sleep "$INTERVAL_SEC"
done

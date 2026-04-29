#!/bin/bash
# ============================================================
# Auto-resubmit SvmW mixed-domain jobs as queue slots free up.
# Runs in parallel with auto_resubmit_exp3.sh (different submitter).
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SUBMITTER="$PROJECT_ROOT/scripts/hpc/launchers/submit_svmw_mixed_seeds.sh"
cd "$PROJECT_ROOT"

ONCE=false
[[ "${1:-}" == "--once" ]] && ONCE=true

INTERVAL_SEC=900
MAX_PASSES=200

pass=0
while true; do
    pass=$((pass + 1))
    active=$(qstat -u s2240011 2>/dev/null | awk 'NR>5 && $10 != "C"' | wc -l)
    # Count remaining SvmW-mixed work via the submitter's dry-run
    remaining=$(bash "$SUBMITTER" --dry-run 2>/dev/null | grep -c "^\[DRY\]")

    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pass $pass | active=$active | svmw_mixed_remaining=$remaining"

    if [[ "$remaining" -eq 0 ]]; then
        echo "[INFO] No SvmW-mixed work remaining — exiting."
        break
    fi

    bash "$SUBMITTER" 2>&1 | tail -3

    $ONCE && { echo "[INFO] --once specified; exiting."; break; }
    if [[ "$pass" -ge "$MAX_PASSES" ]]; then
        echo "[INFO] Reached MAX_PASSES=$MAX_PASSES; exiting."
        break
    fi
    sleep "$INTERVAL_SEC"
done

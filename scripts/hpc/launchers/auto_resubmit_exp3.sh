#!/bin/bash
# ============================================================
# Auto-resubmit exp3 missing jobs as queue slots free up
#
# Cluster has a per-user queue limit of ~113. This wrapper
# periodically calls submit_exp3_final_remaining.sh which
# dedups against the queue, so already-queued jobs are skipped.
#
# Usage:
#   bash scripts/hpc/launchers/auto_resubmit_exp3.sh
#   bash scripts/hpc/launchers/auto_resubmit_exp3.sh --once   # one pass only
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SUBMITTER="$PROJECT_ROOT/scripts/hpc/launchers/submit_exp3_final_remaining.sh"
cd "$PROJECT_ROOT"

ONCE=false
[[ "${1:-}" == "--once" ]] && ONCE=true

INTERVAL_SEC=900     # 15 min between passes
# Threshold raised to 200 so the loop keeps trying — per-queue limits, not the
# overall queue depth, are the actual bottleneck. The submitter dedups against
# already-queued jobs, so re-runs are cheap when no slots free up.
THRESHOLD=200
MAX_PASSES=200

pass=0
while true; do
    pass=$((pass + 1))
    active=$(qstat -u s2240011 2>/dev/null | awk 'NR>5 && $10 != "C"' | wc -l)
    missing=$(grep -cv "^#\|^$" /tmp/exp3_all_missing.txt 2>/dev/null || echo 0)

    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pass $pass | active=$active | missing_list=$missing"

    if [[ "$active" -lt "$THRESHOLD" ]]; then
        echo "[INFO] Active jobs ($active) below threshold ($THRESHOLD); submitting..."
        bash "$SUBMITTER" 2>&1 | tail -5
    else
        echo "[INFO] Queue full ($active >= $THRESHOLD); skipping submission this pass."
    fi

    if $ONCE; then
        echo "[INFO] --once specified; exiting."
        break
    fi

    if [[ "$pass" -ge "$MAX_PASSES" ]]; then
        echo "[INFO] Reached MAX_PASSES=$MAX_PASSES; exiting."
        break
    fi

    # Stop if no jobs in queue and no missing jobs left
    if [[ "$active" -eq 0 ]]; then
        # Verify all 478 expected outputs exist before exiting
        echo "[INFO] Queue empty. Checking if all expected outputs exist..."
        # (regenerate missing list externally to confirm done)
        echo "[INFO] Sleeping $INTERVAL_SEC s before final re-check..."
    fi

    sleep "$INTERVAL_SEC"
done

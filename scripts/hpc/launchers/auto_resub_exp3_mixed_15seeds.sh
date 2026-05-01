#!/bin/bash
# ============================================================
# Periodic auto-resubmitter for exp3 mixed-domain (15-seed scope).
#
# Wraps submit_exp3_mixed_15seeds.sh in a loop and waits for QOS
# capacity to free before retrying. Stops once the missing list is
# fully covered (no pending tuples) or after MAX_ITER iterations.
#
# Tunables via env:
#   MISSING_FILE      Path to missing tuples (default: /tmp/exp3_mixed_missing.txt)
#   SLEEP_SECS        Seconds between iterations (default: 600)
#   MAX_ITER          Stop after N iterations (default: 200, ~33h with 10min sleep)
#   ACTIVE_THRESHOLD  Skip iteration when active jobs >= this (default: 280)
#
# Usage:
#   nohup bash scripts/hpc/launchers/auto_resub_exp3_mixed_15seeds.sh \
#         > /tmp/auto_resub_exp3_mixed.log 2>&1 &
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SUBMITTER="$PROJECT_ROOT/scripts/hpc/launchers/submit_exp3_mixed_15seeds.sh"
MISSING_FILE="${MISSING_FILE:-/tmp/exp3_mixed_missing.txt}"
SLEEP_SECS="${SLEEP_SECS:-600}"
MAX_ITER="${MAX_ITER:-200}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-280}"

cd "$PROJECT_ROOT"

iter=0
while (( iter < MAX_ITER )); do
    iter=$((iter+1))
    ts=$(date '+%F %T')
    active=$(squeue -h -u "$USER" 2>/dev/null | wc -l)
    echo "[$ts] iter=$iter active=$active"

    if (( active >= ACTIVE_THRESHOLD )); then
        echo "[$ts]   active>=${ACTIVE_THRESHOLD}, sleeping ${SLEEP_SECS}s"
        sleep "$SLEEP_SECS"
        continue
    fi

    MISSING_FILE="$MISSING_FILE" bash "$SUBMITTER" 2>&1 | tail -3

    sleep "$SLEEP_SECS"
done
echo "[done] reached MAX_ITER=$MAX_ITER"

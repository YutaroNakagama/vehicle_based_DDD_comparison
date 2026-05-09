#!/bin/bash
# Auto-resubmit daemon for exp2 remaining jobs.
# Calls submit_exp2_multiqueue.sh every 30 min until all 1512 jobs are submitted.

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
MULTI_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/submit_exp2_multiqueue.sh"
SUBMITTED_FILE="$PROJECT_ROOT/scripts/hpc/logs/domain/submitted_exp2_v2.txt"
DAEMON_LOG="$PROJECT_ROOT/scripts/hpc/logs/domain/auto_resubmit_exp2_v2.log"
SLEEP_INTERVAL=1800  # 30 min

echo "[$(date)] auto_resubmit_exp2_v2 started (PID=$$)" | tee -a "$DAEMON_LOG"

while true; do
    TOTAL=$(grep -c . "$SUBMITTED_FILE" 2>/dev/null || echo 0)
    if [[ $TOTAL -ge 1512 ]]; then
        echo "[$(date)] All 1512 jobs submitted. Daemon exiting." | tee -a "$DAEMON_LOG"
        exit 0
    fi
    echo "[$(date)] $TOTAL/1512 submitted, running multiqueue submission..." | tee -a "$DAEMON_LOG"
    bash "$MULTI_SCRIPT" >> "$DAEMON_LOG" 2>&1
    echo "[$(date)] Sleeping ${SLEEP_INTERVAL}s..." | tee -a "$DAEMON_LOG"
    sleep "$SLEEP_INTERVAL"
done

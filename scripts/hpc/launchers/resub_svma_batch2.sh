#!/bin/bash
# =============================================================
# Daemon: Wait for Batch 1 to finish, then submit Batch 2
# SvmA domain_train remaining (indices 125-249)
# =============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

TASK_FILE="scripts/hpc/logs/train/task_files/array_svma_domain_train_remaining.txt"
PBS_SCRIPT="scripts/hpc/jobs/train/pbs_array_svma.sh"
LOG="scripts/hpc/logs/train/resub_svma_batch2_$(date +%Y%m%d_%H%M%S).log"
POLL_INTERVAL=300  # 5 minutes

# Batch 1 job IDs (array base IDs)
BATCH1_JOBS=("14965886" "14965887" "14965888" "14965889")

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "Daemon started. Waiting for Batch 1 to complete..."
log "Batch 1 jobs: ${BATCH1_JOBS[*]}"
log "Poll interval: ${POLL_INTERVAL}s"

while true; do
    # Check if any batch 1 jobs are still running/queued
    STILL_RUNNING=0
    for jid in "${BATCH1_JOBS[@]}"; do
        if qstat "$jid" 2>/dev/null | grep -q "$jid"; then
            STILL_RUNNING=$((STILL_RUNNING + 1))
        fi
    done

    if [[ $STILL_RUNNING -eq 0 ]]; then
        log "All Batch 1 jobs completed!"
        break
    fi

    log "Batch 1: $STILL_RUNNING jobs still active. Waiting ${POLL_INTERVAL}s..."
    sleep $POLL_INTERVAL
done

# Submit Batch 2
log "=== Submitting Batch 2 (indices 125-249) ==="

log "SINGLE: 125-164 (40 tasks)"
JOB1=$(qsub -J 125-164 -q SINGLE -v TASK_FILE="$TASK_FILE" "$PBS_SCRIPT")
log "  Submitted: $JOB1"

log "DEFAULT: 165-204 (40 tasks)"
JOB2=$(qsub -J 165-204 -q DEFAULT -v TASK_FILE="$TASK_FILE" "$PBS_SCRIPT")
log "  Submitted: $JOB2"

log "SMALL: 205-234 (30 tasks)"
JOB3=$(qsub -J 205-234 -q SMALL -v TASK_FILE="$TASK_FILE" "$PBS_SCRIPT")
log "  Submitted: $JOB3"

log "LONG: 235-249 (15 tasks)"
JOB4=$(qsub -J 235-249 -q LONG -v TASK_FILE="$TASK_FILE" "$PBS_SCRIPT")
log "  Submitted: $JOB4"

log "=== Batch 2 submitted (125 tasks). All 250 tasks now submitted. ==="
log "Daemon finished."

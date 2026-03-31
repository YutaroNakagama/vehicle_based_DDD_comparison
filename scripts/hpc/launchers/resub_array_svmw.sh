#!/bin/bash
# ============================================================
# Array Job Resubmitter: submits remaining SvmW array ranges
# as queue slots become available.
#
# Usage:
#   nohup bash scripts/hpc/launchers/resub_array_svmw.sh \
#     > scripts/hpc/logs/train/resub_array_output.log 2>&1 &
# ============================================================
set -o pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
TASK_FILE="$PROJECT_ROOT/scripts/hpc/logs/train/task_files/array_svmw_tasks.txt"
PBS_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_array_svmw.sh"

TOTAL_TASKS=$(wc -l < "$TASK_FILE")
MAX_INDEX=$((TOTAL_TASKS - 1))
POLL_INTERVAL=120  # seconds

# Track where we left off (first unsubmitted index)
NEXT_INDEX=${1:-81}   # pass as argument or default to 81

# Per-user queue limits (approximate - will retry on failure)
BATCH_SIZE=10         # elements per array job
QUEUES=(SINGLE DEFAULT SMALL LONG)

LOG_FILE="$PROJECT_ROOT/scripts/hpc/logs/train/resub_array_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "============================================================"
log "Array resubmitter: indices $NEXT_INDEX → $MAX_INDEX ($((MAX_INDEX - NEXT_INDEX + 1)) tasks)"
log "Batch size: $BATCH_SIZE per qsub"
log "============================================================"

while [[ $NEXT_INDEX -le $MAX_INDEX ]]; do
    submitted_any=false

    for q in "${QUEUES[@]}"; do
        [[ $NEXT_INDEX -gt $MAX_INDEX ]] && break

        END_INDEX=$((NEXT_INDEX + BATCH_SIZE - 1))
        [[ $END_INDEX -gt $MAX_INDEX ]] && END_INDEX=$MAX_INDEX

        # Need at least 2 elements for -J
        if [[ $END_INDEX -le $NEXT_INDEX ]]; then
            # Single element: use non-array submission
            RESULT=$(qsub -q "$q" \
                -v "TASK_FILE=$TASK_FILE,PBS_ARRAY_INDEX=$NEXT_INDEX" \
                "$PBS_SCRIPT" 2>&1)
            if [[ $? -eq 0 ]]; then
                log "[SUBMIT] index $NEXT_INDEX → $q → $RESULT"
                NEXT_INDEX=$((NEXT_INDEX + 1))
                submitted_any=true
                sleep 0.5
            else
                log "[SKIP] index $NEXT_INDEX → $q rejected: $RESULT"
            fi
        else
            RESULT=$(qsub -J "${NEXT_INDEX}-${END_INDEX}" -q "$q" \
                -v "TASK_FILE=$TASK_FILE" \
                "$PBS_SCRIPT" 2>&1)
            if [[ $? -eq 0 ]]; then
                COUNT=$((END_INDEX - NEXT_INDEX + 1))
                log "[SUBMIT] indices ${NEXT_INDEX}-${END_INDEX} ($COUNT tasks) → $q → $RESULT"
                NEXT_INDEX=$((END_INDEX + 1))
                submitted_any=true
                sleep 0.5
            else
                log "[SKIP] ${NEXT_INDEX}-${END_INDEX} → $q rejected"
            fi
        fi
    done

    if [[ $NEXT_INDEX -gt $MAX_INDEX ]]; then
        log "[DONE] All $TOTAL_TASKS tasks submitted!"
        break
    fi

    if [[ "$submitted_any" == "false" ]]; then
        log "[WAIT] All queues full. Next: index $NEXT_INDEX. Waiting ${POLL_INTERVAL}s..."
    fi

    sleep "$POLL_INTERVAL"
done

log "============================================================"
log "Resubmitter finished. Total tasks: $TOTAL_TASKS"
log "============================================================"

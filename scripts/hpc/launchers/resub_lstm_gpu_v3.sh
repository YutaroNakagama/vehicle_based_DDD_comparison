#!/bin/bash
# =============================================================
# Daemon: Continuously submit Lstm GPU jobs (v3 task file)
# Regenerated 2026-04-09 with 438 remaining tasks
# =============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

TASK_FILE="scripts/hpc/logs/train/task_files/array_lstm_domain_train_remaining_v3.txt"
PBS_SCRIPT="scripts/hpc/jobs/train/pbs_array_lstm_gpu.sh"
TOTAL_TASKS=$(wc -l < "$TASK_FILE")
LOG="scripts/hpc/logs/train/resub_lstm_gpu_v3_$(date +%Y%m%d_%H%M%S).log"
POLL_INTERVAL=180   # 3 minutes

declare -A Q_LIMIT=( [GPU-1]=4 [GPU-1A]=2 [GPU-S]=2 )
QUEUES=(GPU-1 GPU-1A GPU-S)

IDX=0

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

count_my_jobs() {
    local Q=$1
    local cnt
    cnt=$(qstat -u s2240011 2>/dev/null | awk -v q="$Q" '$3 == q {n++} END {print n+0}')
    echo "$cnt"
}

log "Daemon started. Task file: $TASK_FILE ($TOTAL_TASKS tasks)"
log "Queues: ${QUEUES[*]}"

while [[ $IDX -lt $TOTAL_TASKS ]]; do
    SUBMITTED_THIS_ROUND=0

    for Q in "${QUEUES[@]}"; do
        [[ $IDX -ge $TOTAL_TASKS ]] && break

        LIMIT=${Q_LIMIT[$Q]}
        MY_COUNT=$(count_my_jobs "$Q")

        AVAIL=$((LIMIT - MY_COUNT))
        if [[ $AVAIL -le 0 ]]; then
            continue
        fi

        # Submit up to AVAIL tasks one at a time
        SUBMITTED_Q=0
        while [[ $SUBMITTED_Q -lt $AVAIL && $IDX -lt $TOTAL_TASKS ]]; do
            LINE=$(sed -n "$((IDX + 1))p" "$TASK_FILE")
            if [[ -z "$LINE" ]]; then
                IDX=$((IDX + 1))
                continue
            fi

            WALL="20:00:00"
            JOB=$(qsub -q "$Q" \
                -l "select=1:ncpus=8:ngpus=1:mem=8gb" -l "walltime=$WALL" \
                -v "TASK_FILE=$TASK_FILE,PBS_ARRAY_INDEX=$IDX" \
                "$PBS_SCRIPT" 2>&1) || true

            if [[ "$JOB" =~ ^[0-9]+ ]]; then
                log "[OK] idx=$IDX -> $Q : $JOB"
                SUBMITTED_Q=$((SUBMITTED_Q + 1))
                SUBMITTED_THIS_ROUND=$((SUBMITTED_THIS_ROUND + 1))
            else
                log "[FAIL] idx=$IDX -> $Q : $JOB"
            fi
            IDX=$((IDX + 1))
        done
    done

    if [[ $SUBMITTED_THIS_ROUND -eq 0 ]]; then
        log "[WAIT] All queues full. Next idx=$IDX/$TOTAL_TASKS. Waiting ${POLL_INTERVAL}s..."
        sleep $POLL_INTERVAL
    else
        log "[ROUND] Submitted $SUBMITTED_THIS_ROUND tasks. Next idx=$IDX/$TOTAL_TASKS"
        sleep 2
    fi
done

log "=== All $TOTAL_TASKS task indices processed. Daemon finished. ==="

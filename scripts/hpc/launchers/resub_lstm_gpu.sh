#!/bin/bash
# =============================================================
# Daemon: Continuously submit Lstm GPU array jobs across queues
# Distributes 461 tasks across GPU-1 (4), GPU-1A (2), GPU-S (2)
# GPU-L and GPU-LA excluded (8-GPU nodes, overkill for 1-GPU LSTM)
# =============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

TASK_FILE="scripts/hpc/logs/train/task_files/array_lstm_domain_train_remaining.txt"
PBS_SCRIPT="scripts/hpc/jobs/train/pbs_array_lstm_gpu.sh"
TOTAL_TASKS=$(wc -l < "$TASK_FILE")
LOG="scripts/hpc/logs/train/resub_lstm_gpu_$(date +%Y%m%d_%H%M%S).log"
POLL_INTERVAL=180   # 3 minutes

# Queue config: name, user_limit, select
declare -A Q_LIMIT=( [GPU-1]=4 [GPU-1A]=2 [GPU-S]=2 )
declare -A Q_SELECT=(
    [GPU-1]="select=1:ncpus=8:ngpus=1:mem=8gb"
    [GPU-1A]="select=1:ncpus=8:ngpus=1:mem=8gb"
    [GPU-S]="select=1:ncpus=8:ngpus=1:mem=8gb"
)
declare -A Q_WALL=(
    [GPU-1]="20:00:00"
    [GPU-1A]="20:00:00"
    [GPU-S]="20:00:00"
)
QUEUES=(GPU-1 GPU-1A GPU-S)

IDX=10  # Next task index to submit (0-9 already submitted)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "Daemon started. Tasks: $TOTAL_TASKS (starting from index $IDX)"
log "Task file: $TASK_FILE"
log "Queues: ${QUEUES[*]}"

while [[ $IDX -lt $TOTAL_TASKS ]]; do
    SUBMITTED_THIS_ROUND=0

    for Q in "${QUEUES[@]}"; do
        [[ $IDX -ge $TOTAL_TASKS ]] && break

        LIMIT=${Q_LIMIT[$Q]}
        SELECT=${Q_SELECT[$Q]}
        WALL=${Q_WALL[$Q]}

        # Count my running+queued jobs in this queue (exact match with word boundary)
        MY_COUNT=$(qstat -u s2240011 2>/dev/null | awk -v q="$Q" '$3 == q' | grep -c s2240011 || echo 0)

        AVAIL=$((LIMIT - MY_COUNT))
        if [[ $AVAIL -le 0 ]]; then
            log "[DT] $Q full ($MY_COUNT/$LIMIT). Skip."
            continue
        fi

        # Submit up to AVAIL tasks as one array job
        END_IDX=$((IDX + AVAIL - 1))
        if [[ $END_IDX -ge $TOTAL_TASKS ]]; then
            END_IDX=$((TOTAL_TASKS - 1))
        fi

        RANGE="${IDX}-${END_IDX}"
        ACTUAL=$((END_IDX - IDX + 1))

        log "[DT] Submitting $RANGE ($ACTUAL tasks) → $Q"
        if [[ $ACTUAL -eq 1 ]]; then
            # Single task: use -v PBS_ARRAY_INDEX to simulate array
            JOB=$(qsub -q "$Q" \
                -l "select=1:ncpus=8:ngpus=1:mem=8gb" -l "walltime=$WALL" \
                -v "TASK_FILE=$TASK_FILE,PBS_ARRAY_INDEX=$IDX" \
                "$PBS_SCRIPT" 2>&1) || true
        else
            JOB=$(qsub -J "$RANGE" -q "$Q" \
                -l "select=1:ncpus=8:ngpus=1:mem=8gb" -l "walltime=$WALL" \
                -v "TASK_FILE=$TASK_FILE" \
                "$PBS_SCRIPT" 2>&1) || true
        fi

        if [[ "$JOB" =~ ^[0-9]+ ]]; then
            log "[DT] OK: $JOB"
            IDX=$((END_IDX + 1))
            SUBMITTED_THIS_ROUND=$((SUBMITTED_THIS_ROUND + ACTUAL))
        else
            log "[DT] FAILED: $JOB"
        fi
    done

    if [[ $SUBMITTED_THIS_ROUND -eq 0 ]]; then
        log "[DT] All queues full. Next: $IDX/$TOTAL_TASKS. Waiting ${POLL_INTERVAL}s..."
        sleep $POLL_INTERVAL
    else
        log "[DT] Submitted $SUBMITTED_THIS_ROUND tasks this round. Next: $IDX/$TOTAL_TASKS"
        # Brief pause before trying to fill more
        sleep 2
    fi
done

log "=== All $TOTAL_TASKS tasks submitted. Daemon finished. ==="

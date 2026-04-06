#!/bin/bash
# =============================================================
# Daemon: Continuously submit Lstm GPU jobs, skip completed tasks
# Uses v2 task file (445 lines), checks completion before submitting
# =============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

TASK_FILE="scripts/hpc/logs/train/task_files/array_lstm_domain_train_remaining_v2.txt"
PBS_SCRIPT="scripts/hpc/jobs/train/pbs_array_lstm_gpu.sh"
TOTAL_TASKS=$(wc -l < "$TASK_FILE")
LOG="scripts/hpc/logs/train/resub_lstm_gpu_v2_$(date +%Y%m%d_%H%M%S).log"
POLL_INTERVAL=180   # 3 minutes

# Queue config
declare -A Q_LIMIT=( [GPU-1]=4 [GPU-1A]=2 [GPU-S]=2 )
QUEUES=(GPU-1 GPU-1A GPU-S)

IDX=0  # Start from beginning of v2 file

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Function to check if a task is already completed by looking for train CSV
is_task_done() {
    local LINE=$1
    IFS='|' read -r MODEL CONDITION MODE DISTANCE DOMAIN RATIO SEED N_TRIALS RANKING RUN_EVAL SCRIPT_TYPE <<< "$LINE"
    local TAG
    case "$CONDITION" in
        baseline) TAG="domain_train_prior_${MODEL}_baseline_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_s${SEED}" ;;
        smote)    TAG="domain_train_prior_${MODEL}_imbalv3_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_subjectwise_ratio${RATIO}_s${SEED}" ;;
        smote_plain) TAG="domain_train_prior_${MODEL}_smote_plain_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_ratio${RATIO}_s${SEED}" ;;
        undersample) TAG="domain_train_prior_${MODEL}_undersample_rus_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_ratio${RATIO}_s${SEED}" ;;
    esac
    local PATTERN="train_results_${MODEL}_${TAG}.csv"
    # Check if any such file exists under results/outputs/training/$MODEL
    if find "results/outputs/training/$MODEL" -name "$PATTERN" -print -quit 2>/dev/null | grep -q .; then
        return 0  # done
    fi
    return 1  # not done
}

log "Daemon started. Task file: $TASK_FILE ($TOTAL_TASKS tasks)"
log "Queues: ${QUEUES[*]}"
log "Starting from index $IDX"

while [[ $IDX -lt $TOTAL_TASKS ]]; do
    SUBMITTED_THIS_ROUND=0

    for Q in "${QUEUES[@]}"; do
        [[ $IDX -ge $TOTAL_TASKS ]] && break

        LIMIT=${Q_LIMIT[$Q]}

        # Count my running+queued jobs in this queue (exact match)
        MY_COUNT=$(qstat -u s2240011 2>/dev/null | awk -v q="$Q" '$3 == q' | grep -c s2240011 || echo 0)

        AVAIL=$((LIMIT - MY_COUNT))
        if [[ $AVAIL -le 0 ]]; then
            continue
        fi

        # Submit up to AVAIL tasks, skipping already-done ones
        SUBMITTED_Q=0
        while [[ $SUBMITTED_Q -lt $AVAIL && $IDX -lt $TOTAL_TASKS ]]; do
            LINE=$(sed -n "$((IDX + 1))p" "$TASK_FILE")
            if [[ -z "$LINE" ]]; then
                IDX=$((IDX + 1))
                continue
            fi

            # Skip if already done
            if is_task_done "$LINE"; then
                log "[SKIP] Index $IDX already completed."
                IDX=$((IDX + 1))
                continue
            fi

            # Submit single task
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

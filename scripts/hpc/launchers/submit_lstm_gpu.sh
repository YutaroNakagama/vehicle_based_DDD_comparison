#!/bin/bash
# Submit all pending Lstm jobs to GPU queues (GPU-1 / GPU-1A round-robin)
# GPU-1: A40, max_run u:4, GPU-1A: A100, max_run u:2
# No max_queued limit → can queue them all at once
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
DOMAIN_TRAIN_JOB="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
N_TRIALS=100
RANKING="knn"

PENDING_FILE="/tmp/pending_lstm_gpu.txt"
LOG="$PROJECT_ROOT/scripts/hpc/logs/train/submit_lstm_gpu_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

QUEUES=("GPU-1" "GPU-1A")
Q_IDX=0
SUBMITTED=0
FAILED=0
TOTAL=$(wc -l < "$PENDING_FILE")

log "Submitting $TOTAL Lstm jobs to GPU queues: ${QUEUES[*]}"

while IFS=: read -r MODEL CONDITION DISTANCE DOMAIN MODE RATIO SEED; do
    QUEUE="${QUEUES[$((Q_IDX % 2))]}"

    # Lstm on GPU: 1 GPU, 8 CPUs, 32GB, 20h walltime
    NCPUS_MEM="ncpus=8:ngpus=1:mem=32gb"
    WALLTIME="20:00:00"

    local_model_short="Ls"
    case "$CONDITION" in
        baseline)     local_cond_short="bs" ;;
        smote_plain)  local_cond_short="sp" ;;
        smote)        local_cond_short="sm" ;;
        undersample)  local_cond_short="un" ;;
    esac

    if [[ "$CONDITION" == "baseline" || "$RATIO" == "none" ]]; then
        JOB_NAME="${local_model_short}_${local_cond_short}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_s${SEED}"
        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
        CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    else
        JOB_NAME="${local_model_short}_${local_cond_short}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_r${RATIO}_s${SEED}"
        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
        CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true,RATIO=$RATIO"
    fi
    CMD="$CMD $DOMAIN_TRAIN_JOB"

    JOB_ID=$(eval "$CMD" 2>&1)
    if [[ $? -eq 0 ]]; then
        log "[OK] #$((SUBMITTED+1))/$TOTAL $MODEL:$CONDITION:$DISTANCE:$DOMAIN:r$RATIO:s$SEED → $JOB_ID ($QUEUE)"
        ((SUBMITTED++))
        Q_IDX=$((Q_IDX + 1))
    else
        log "[FAIL] $QUEUE: $JOB_ID for $MODEL:$CONDITION:$DISTANCE:$DOMAIN:r$RATIO:s$SEED"
        ((FAILED++))
    fi
    sleep 0.1
done < "$PENDING_FILE"

log "=== DONE === Submitted: $SUBMITTED, Failed: $FAILED, Total: $TOTAL"

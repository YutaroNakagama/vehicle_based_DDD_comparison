#!/bin/bash
# ============================================================
# Fast submission - Distribute across queues for parallel execution
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$SCRIPT_DIR"

MAX_RETRIES=200
RETRY_WAIT=30  # Reduced wait time

echo "============================================================"
echo "Fast Submit - Distributed Queues"
echo "============================================================"
echo "Time: $(date)"
echo "============================================================"

ALL_JOBS=""
JOB_COUNT=0

JOB_LOG="$PROJECT_ROOT/scripts/hpc/logs/imbalance/job_ids_fast_$(date +%Y%m%d_%H%M%S).txt"
echo "# Fast job submission: $(date)" > "$JOB_LOG"
echo "# Format: method,seed,ratio,queue,mem,train_jobid,eval_jobid" >> "$JOB_LOG"

# Distribute across queues more evenly
get_queue_and_resources() {
    local method="$1"
    local seed="$2"
    
    case "$method" in
        # Heavy methods stay in SMALL (best for long jobs)
        smote_enn|smote_tomek)
            echo "SMALL 6gb 16:00:00 4" ;;
        # Distribute SMOTE variants across DEFAULT and SINGLE based on seed
        smote|smote_rus)
            if [[ "$seed" == "42" ]]; then
                echo "DEFAULT 4gb 08:00:00 4"
            elif [[ "$seed" == "123" ]]; then
                echo "SINGLE 4gb 08:00:00 4"
            else
                echo "DEFAULT 4gb 08:00:00 4"
            fi
            ;;
        smote_balanced_rf)
            echo "SINGLE 4gb 08:00:00 4" ;;
        # Undersample in DEFAULT
        undersample_rus|undersample_tomek|undersample_enn)
            echo "DEFAULT 4gb 04:00:00 4" ;;
        baseline)
            echo "DEFAULT 4gb 02:00:00 4" ;;
        balanced_rf|easy_ensemble)
            if [[ "$seed" == "42" ]]; then
                echo "DEFAULT 4gb 04:00:00 4"
            else
                echo "SINGLE 4gb 04:00:00 4"
            fi
            ;;
        *)
            echo "SINGLE 4gb 08:00:00 4" ;;
    esac
}

submit_with_retry() {
    local cmd="$1"
    local retries=0
    local result
    
    while [ $retries -lt $MAX_RETRIES ]; do
        result=$(eval "$cmd" 2>&1)
        if echo "$result" | grep -q "would exceed"; then
            retries=$((retries + 1))
            current_jobs=$(qstat -u $USER 2>/dev/null | grep -c "$USER" || echo "0")
            echo "      [WAIT] Queue limit ($current_jobs). ${RETRY_WAIT}s... ($retries)"
            sleep $RETRY_WAIT
        else
            echo "$result"
            return 0
        fi
    done
    return 1
}

submit_job() {
    local METHOD="$1"
    local SEED="$2"
    local RATIO="$3"
    
    if [[ "$METHOD" == "smote_balanced_rf" ]]; then
        MODEL="BalancedRF"
    else
        MODEL="RF"
    fi
    
    if [[ "$RATIO" == "default" ]]; then
        TAG="imbal_v2_${METHOD}_seed${SEED}"
    else
        TAG="imbal_v2_${METHOD}_ratio${RATIO//./_}_seed${SEED}"
    fi
    
    read QUEUE MEM WALLTIME NCPUS <<< $(get_queue_and_resources "$METHOD" "$SEED")
    
    if [[ "$RATIO" == "default" ]]; then
        SCRIPT="pbs_train_${METHOD}.sh"
        TRAIN_JOB=$(submit_with_retry "qsub -q '$QUEUE' -l select=1:ncpus=${NCPUS}:mem=${MEM} -l walltime=${WALLTIME} -v PBS_O_WORKDIR='$PROJECT_ROOT',SEED='$SEED' '$SCRIPT'")
    else
        TRAIN_JOB=$(submit_with_retry "qsub -q '$QUEUE' -l select=1:ncpus=${NCPUS}:mem=${MEM} -l walltime=${WALLTIME} -v PBS_O_WORKDIR='$PROJECT_ROOT',MODEL='$MODEL',RATIO='$RATIO',METHOD='$METHOD',TAG='$TAG',SEED='$SEED' pbs_train_generic_ratio.sh")
    fi
    
    TRAIN_ID="${TRAIN_JOB%%.*}"
    echo "  [$METHOD] s=$SEED r=$RATIO q=$QUEUE -> $TRAIN_ID"
    
    EVAL_JOB=$(submit_with_retry "qsub -W depend=afterok:$TRAIN_JOB -v PBS_O_WORKDIR='$PROJECT_ROOT',MODEL='$MODEL',TAG='$TAG',TRAIN_JOBID='$TRAIN_ID',SEED='$SEED' pbs_evaluate.sh")
    
    ALL_JOBS="$ALL_JOBS $TRAIN_JOB $EVAL_JOB"
    JOB_COUNT=$((JOB_COUNT + 2))
    echo "$METHOD,$SEED,$RATIO,$QUEUE,$TRAIN_JOB" >> "$JOB_LOG"
    
    sleep 0.3
}

# ============================================================
# Submit remaining jobs (check what's NOT in queue yet)
# Already submitted: see job_ids_missing_*.txt
# ============================================================

echo ""
echo "=== Remaining seed=42, ratio=1.0 ==="
# smote_enn already submitted (14618608), smote_rus submitted (14618610)
# Need: smote_balanced_rf, undersample_rus, undersample_tomek, undersample_enn
for METHOD in smote_balanced_rf undersample_rus undersample_tomek undersample_enn; do
    submit_job "$METHOD" "42" "1.0"
done

echo ""
echo "=== seed=123: ALL ==="
for RATIO in 0.1 0.5 1.0; do
    echo "  -- ratio=$RATIO --"
    for METHOD in smote smote_tomek smote_enn smote_rus smote_balanced_rf undersample_rus undersample_tomek undersample_enn; do
        submit_job "$METHOD" "123" "$RATIO"
    done
done

echo ""
echo "=== seed=456: ALL ==="
for RATIO in 0.1 0.5 1.0; do
    echo "  -- ratio=$RATIO --"
    for METHOD in smote smote_tomek smote_enn smote_rus smote_balanced_rf undersample_rus undersample_tomek undersample_enn; do
        submit_job "$METHOD" "456" "$RATIO"
    done
done

echo ""
echo "============================================================"
echo "Done! Total: $JOB_COUNT jobs"
echo "Log: $JOB_LOG"
echo "============================================================"

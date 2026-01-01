#!/bin/bash
# ============================================================
# Priority DEFAULT queue - avoid SINGLE queue limit
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$SCRIPT_DIR"

MAX_RETRIES=200
RETRY_WAIT=30

echo "============================================================"
echo "Priority DEFAULT Queue Submission"
echo "============================================================"
echo "Time: $(date)"
echo "============================================================"

JOB_COUNT=0

JOB_LOG="$PROJECT_ROOT/scripts/hpc/logs/imbalance/job_ids_default_$(date +%Y%m%d_%H%M%S).txt"
echo "# DEFAULT priority submission: $(date)" > "$JOB_LOG"

# Use DEFAULT for everything possible (max walltime 168h, max ncpus 64)
get_queue_and_resources() {
    local method="$1"
    case "$method" in
        smote_enn|smote_tomek)
            # These need more memory, use SMALL
            echo "SMALL 6gb 16:00:00 4" ;;
        *)
            # Everything else goes to DEFAULT
            echo "DEFAULT 4gb 08:00:00 4" ;;
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
            current=$(qstat -u $USER 2>/dev/null | grep -c "$USER" || echo "0")
            echo "      [WAIT] limit ($current). ${RETRY_WAIT}s..."
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
    
    [[ "$METHOD" == "smote_balanced_rf" ]] && MODEL="BalancedRF" || MODEL="RF"
    
    if [[ "$RATIO" == "default" ]]; then
        TAG="imbal_v2_${METHOD}_seed${SEED}"
    else
        TAG="imbal_v2_${METHOD}_ratio${RATIO//./_}_seed${SEED}"
    fi
    
    read QUEUE MEM WALLTIME NCPUS <<< $(get_queue_and_resources "$METHOD")
    
    if [[ "$RATIO" == "default" ]]; then
        SCRIPT="pbs_train_${METHOD}.sh"
        TRAIN_JOB=$(submit_with_retry "qsub -q '$QUEUE' -l select=1:ncpus=${NCPUS}:mem=${MEM} -l walltime=${WALLTIME} -v PBS_O_WORKDIR='$PROJECT_ROOT',SEED='$SEED' '$SCRIPT'")
    else
        TRAIN_JOB=$(submit_with_retry "qsub -q '$QUEUE' -l select=1:ncpus=${NCPUS}:mem=${MEM} -l walltime=${WALLTIME} -v PBS_O_WORKDIR='$PROJECT_ROOT',MODEL='$MODEL',RATIO='$RATIO',METHOD='$METHOD',TAG='$TAG',SEED='$SEED' pbs_train_generic_ratio.sh")
    fi
    
    TRAIN_ID="${TRAIN_JOB%%.*}"
    echo "  [$METHOD] s=$SEED r=$RATIO q=$QUEUE -> $TRAIN_ID"
    
    # Eval also goes to DEFAULT (it's short)
    EVAL_JOB=$(submit_with_retry "qsub -q DEFAULT -l select=1:ncpus=2:mem=4gb -l walltime=02:00:00 -W depend=afterok:$TRAIN_JOB -v PBS_O_WORKDIR='$PROJECT_ROOT',MODEL='$MODEL',TAG='$TAG',TRAIN_JOBID='$TRAIN_ID',SEED='$SEED' pbs_evaluate.sh")
    
    JOB_COUNT=$((JOB_COUNT + 2))
    echo "$METHOD,$SEED,$RATIO,$QUEUE,$TRAIN_JOB" >> "$JOB_LOG"
    
    sleep 0.3
}

# ============================================================
# Submit ALL remaining jobs
# ============================================================

echo ""
echo "=== seed=42, ratio=1.0 remaining ==="
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

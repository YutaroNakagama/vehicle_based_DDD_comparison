#!/bin/bash
# ============================================================
# Continue submitting remaining jobs from launch_multiseed_ratio.sh
# With queue limit handling - waits and retries if limit reached
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$SCRIPT_DIR"

SEEDS="42 123 456"
RATIOS="0.1 0.5 1.0"
MAX_RETRIES=100
RETRY_WAIT=60  # seconds

echo "============================================================"
echo "Continue Multi-Seed + Multi-Ratio Submission"
echo "============================================================"
echo "Time: $(date)"
echo "Will wait and retry if queue limit is reached"
echo "============================================================"

ALL_JOBS=""
JOB_COUNT=0

JOB_LOG="$PROJECT_ROOT/scripts/hpc/logs/imbalance/job_ids_multiseed_ratio_continue_$(date +%Y%m%d_%H%M%S).txt"
echo "# Continue multi-seed + Multi-ratio job submission: $(date)" > "$JOB_LOG"
echo "# Format: method,seed,ratio,queue,mem,train_jobid,eval_jobid" >> "$JOB_LOG"

get_queue_and_resources() {
    local method="$1"
    local queue mem walltime ncpus
    
    case "$method" in
        # Heavy methods: SMOTE+ENN, SMOTE+Tomek (need more time and memory)
        smote_enn|smote_tomek)
            queue="SMALL"
            mem="6gb"
            walltime="16:00:00"
            ncpus=4
            ;;
        # Medium methods: SMOTE variants
        smote|smote_rus|smote_balanced_rf)
            queue="SINGLE"
            mem="4gb"
            walltime="08:00:00"
            ncpus=4
            ;;
        # Light methods: Undersample only - use DEFAULT (TINY has 30min limit!)
        undersample_rus|undersample_tomek|undersample_enn)
            queue="DEFAULT"
            mem="4gb"
            walltime="04:00:00"
            ncpus=4
            ;;
        # Baseline - use DEFAULT
        baseline)
            queue="DEFAULT"
            mem="4gb"
            walltime="02:00:00"
            ncpus=4
            ;;
        # Ensemble methods
        balanced_rf|easy_ensemble)
            queue="SINGLE"
            mem="4gb"
            walltime="04:00:00"
            ncpus=4
            ;;
        *)
            queue="SINGLE"
            mem="4gb"
            walltime="08:00:00"
            ncpus=4
            ;;
    esac
    
    echo "$queue $mem $walltime $ncpus"
}

# Function to submit with retry on queue limit
submit_with_retry() {
    local cmd="$1"
    local retries=0
    local result
    
    while [ $retries -lt $MAX_RETRIES ]; do
        result=$(eval "$cmd" 2>&1)
        if echo "$result" | grep -q "would exceed"; then
            retries=$((retries + 1))
            current_jobs=$(qstat -u $USER 2>/dev/null | grep -c "$USER" || echo "0")
            echo "    [WAIT] Queue limit reached ($current_jobs jobs). Waiting ${RETRY_WAIT}s... (retry $retries/$MAX_RETRIES)"
            sleep $RETRY_WAIT
        else
            echo "$result"
            return 0
        fi
    done
    echo "ERROR: Max retries reached"
    return 1
}

# ============================================================
# Remaining jobs for seed=42, ratio=1.0
# ============================================================
echo ""
echo "=== Remaining seed=42, ratio=1.0 ==="
SEED=42
RATIO=1.0
for METHOD in smote_enn smote_rus smote_balanced_rf undersample_rus undersample_tomek undersample_enn; do
    if [[ "$METHOD" == "smote_balanced_rf" ]]; then
        MODEL="BalancedRF"
    else
        MODEL="RF"
    fi
    
    TAG="imbal_v2_${METHOD}_ratio${RATIO//./_}_seed${SEED}"
    read QUEUE MEM WALLTIME NCPUS <<< $(get_queue_and_resources "$METHOD")
    
    TRAIN_JOB=$(submit_with_retry "qsub -q '$QUEUE' -l select=1:ncpus=${NCPUS}:mem=${MEM} -l walltime=${WALLTIME} -v PBS_O_WORKDIR='$PROJECT_ROOT',MODEL='$MODEL',RATIO='$RATIO',METHOD='$METHOD',TAG='$TAG',SEED='$SEED' pbs_train_generic_ratio.sh")
    TRAIN_ID="${TRAIN_JOB%%.*}"
    echo "  [$METHOD] ratio=$RATIO q=$QUEUE -> $TRAIN_ID"
    
    EVAL_JOB=$(submit_with_retry "qsub -W depend=afterok:$TRAIN_JOB -v PBS_O_WORKDIR='$PROJECT_ROOT',MODEL='$MODEL',TAG='$TAG',TRAIN_JOBID='$TRAIN_ID',SEED='$SEED' pbs_evaluate.sh")
    EVAL_ID="${EVAL_JOB%%.*}"
    echo "    -> Eval: $EVAL_ID"
    
    ALL_JOBS="$ALL_JOBS $TRAIN_JOB $EVAL_JOB"
    JOB_COUNT=$((JOB_COUNT + 2))
    echo "$METHOD,$SEED,$RATIO,$QUEUE,$MEM,$TRAIN_JOB,$EVAL_JOB" >> "$JOB_LOG"
    
    sleep 0.5
done

# ============================================================
# All jobs for seed=123 and seed=456
# ============================================================
for SEED in 123 456; do
    echo ""
    echo "=== SEED=$SEED ==="
    
    for RATIO in $RATIOS; do
        echo "  -- ratio=$RATIO --"
        
        for METHOD in smote smote_tomek smote_enn smote_rus smote_balanced_rf undersample_rus undersample_tomek undersample_enn; do
            if [[ "$METHOD" == "smote_balanced_rf" ]]; then
                MODEL="BalancedRF"
            else
                MODEL="RF"
            fi
            
            TAG="imbal_v2_${METHOD}_ratio${RATIO//./_}_seed${SEED}"
            read QUEUE MEM WALLTIME NCPUS <<< $(get_queue_and_resources "$METHOD")
            
            TRAIN_JOB=$(submit_with_retry "qsub -q '$QUEUE' -l select=1:ncpus=${NCPUS}:mem=${MEM} -l walltime=${WALLTIME} -v PBS_O_WORKDIR='$PROJECT_ROOT',MODEL='$MODEL',RATIO='$RATIO',METHOD='$METHOD',TAG='$TAG',SEED='$SEED' pbs_train_generic_ratio.sh")
            TRAIN_ID="${TRAIN_JOB%%.*}"
            echo "    [$METHOD] q=$QUEUE -> $TRAIN_ID"
            
            EVAL_JOB=$(submit_with_retry "qsub -W depend=afterok:$TRAIN_JOB -v PBS_O_WORKDIR='$PROJECT_ROOT',MODEL='$MODEL',TAG='$TAG',TRAIN_JOBID='$TRAIN_ID',SEED='$SEED' pbs_evaluate.sh")
            EVAL_ID="${EVAL_JOB%%.*}"
            echo "      -> Eval: $EVAL_ID"
            
            ALL_JOBS="$ALL_JOBS $TRAIN_JOB $EVAL_JOB"
            JOB_COUNT=$((JOB_COUNT + 2))
            echo "$METHOD,$SEED,$RATIO,$QUEUE,$MEM,$TRAIN_JOB,$EVAL_JOB" >> "$JOB_LOG"
            
            sleep 0.5
        done
    done
done

echo ""
echo "============================================================"
echo "All remaining jobs submitted!"
echo "Total new jobs: $JOB_COUNT"
echo "Job log: $JOB_LOG"
echo "============================================================"

#!/bin/bash
# ============================================================
# Complete remaining jobs - fixed queue allocation
# Skips already submitted jobs, submits missing ones
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$SCRIPT_DIR"

SEEDS="42 123 456"
RATIOS="0.1 0.5 1.0"
MAX_RETRIES=200
RETRY_WAIT=60

echo "============================================================"
echo "Complete All Remaining Jobs (Fixed Queue)"
echo "============================================================"
echo "Time: $(date)"
echo "Will wait and retry if queue limit is reached"
echo "============================================================"

ALL_JOBS=""
JOB_COUNT=0

JOB_LOG="$PROJECT_ROOT/scripts/hpc/logs/imbalance/job_ids_complete_$(date +%Y%m%d_%H%M%S).txt"
echo "# Complete job submission: $(date)" > "$JOB_LOG"
echo "# Format: method,seed,ratio,queue,mem,train_jobid,eval_jobid" >> "$JOB_LOG"

get_queue_and_resources() {
    local method="$1"
    local queue mem walltime ncpus
    
    case "$method" in
        smote_enn|smote_tomek)
            queue="SMALL"
            mem="6gb"
            walltime="16:00:00"
            ncpus=4
            ;;
        smote|smote_rus|smote_balanced_rf)
            queue="SINGLE"
            mem="4gb"
            walltime="08:00:00"
            ncpus=4
            ;;
        undersample_rus|undersample_tomek|undersample_enn)
            queue="DEFAULT"
            mem="4gb"
            walltime="04:00:00"
            ncpus=4
            ;;
        baseline)
            queue="DEFAULT"
            mem="4gb"
            walltime="02:00:00"
            ncpus=4
            ;;
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

submit_with_retry() {
    local cmd="$1"
    local retries=0
    local result
    
    while [ $retries -lt $MAX_RETRIES ]; do
        result=$(eval "$cmd" 2>&1)
        if echo "$result" | grep -q "would exceed"; then
            retries=$((retries + 1))
            current_jobs=$(qstat -u $USER 2>/dev/null | grep -c "$USER" || echo "0")
            echo "      [WAIT] Queue limit ($current_jobs jobs). Waiting ${RETRY_WAIT}s... ($retries/$MAX_RETRIES)"
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
# Part 1: Fixed-ratio methods (baseline, balanced_rf, easy_ensemble)
# Re-submit baseline for all seeds (was in TINY, now DEFAULT)
# ============================================================
echo ""
echo "=== Part 1: Fixed-ratio methods ==="
FIXED_METHODS="baseline balanced_rf easy_ensemble"

for SEED in $SEEDS; do
    echo ""
    echo "--- SEED=$SEED ---"
    
    for METHOD in $FIXED_METHODS; do
        SCRIPT="pbs_train_${METHOD}.sh"
        
        if [[ ! -f "$SCRIPT" ]]; then
            echo "  [SKIP] $SCRIPT not found"
            continue
        fi
        
        TAG="imbal_v2_${METHOD}_seed${SEED}"
        read QUEUE MEM WALLTIME NCPUS <<< $(get_queue_and_resources "$METHOD")
        
        TRAIN_JOB=$(submit_with_retry "qsub -q '$QUEUE' -l select=1:ncpus=${NCPUS}:mem=${MEM} -l walltime=${WALLTIME} -v PBS_O_WORKDIR='$PROJECT_ROOT',SEED='$SEED' '$SCRIPT'")
        TRAIN_ID="${TRAIN_JOB%%.*}"
        echo "  [$METHOD] seed=$SEED q=$QUEUE -> $TRAIN_ID"
        
        EVAL_JOB=$(submit_with_retry "qsub -W depend=afterok:$TRAIN_JOB -v PBS_O_WORKDIR='$PROJECT_ROOT',MODEL=RF,TAG='$TAG',TRAIN_JOBID='$TRAIN_ID',SEED='$SEED' pbs_evaluate.sh")
        EVAL_ID="${EVAL_JOB%%.*}"
        echo "    -> Eval: $EVAL_ID"
        
        ALL_JOBS="$ALL_JOBS $TRAIN_JOB $EVAL_JOB"
        JOB_COUNT=$((JOB_COUNT + 2))
        echo "$METHOD,$SEED,default,$QUEUE,$MEM,$TRAIN_JOB,$EVAL_JOB" >> "$JOB_LOG"
        
        sleep 0.5
    done
done

# ============================================================
# Part 2: Variable-ratio methods for ALL seeds and ratios
# ============================================================
echo ""
echo "=== Part 2: Variable-ratio methods ==="
VARIABLE_METHODS="smote smote_tomek smote_enn smote_rus smote_balanced_rf undersample_rus undersample_tomek undersample_enn"

for SEED in $SEEDS; do
    echo ""
    echo "=== SEED=$SEED ==="
    
    for RATIO in $RATIOS; do
        echo "  -- ratio=$RATIO --"
        
        for METHOD in $VARIABLE_METHODS; do
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
echo "All jobs submitted!"
echo "Total new jobs: $JOB_COUNT"
echo "Job log: $JOB_LOG"
echo "============================================================"

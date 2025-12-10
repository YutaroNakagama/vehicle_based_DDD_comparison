#!/bin/bash
# ============================================================
# Submit all imbalance methods x target_ratio experiments
# Ratios: 0.1, 0.33 (baseline), 0.5, 1.0
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"

# Methods and their corresponding models
declare -A METHODS=(
    ["baseline"]="RF"
    ["smote"]="RF"
    ["smote_tomek"]="RF"
    ["smote_enn"]="RF"
    ["smote_rus"]="RF"
    ["balanced_rf"]="BalancedRF"
    ["easy_ensemble"]="EasyEnsemble"
    ["smote_balanced_rf"]="BalancedRF"
    ["undersample_rus"]="RF"
    ["undersample_tomek"]="RF"
    ["undersample_enn"]="RF"
)

# Target ratios to test
RATIOS=("0.1" "0.5" "1.0")

# Create log directory
mkdir -p "$PROJECT_ROOT/scripts/hpc/log"

# Job ID tracking file
JOB_LOG="$PROJECT_ROOT/logs/ratio_experiment_jobs_$(date +%Y%m%d_%H%M%S).txt"
echo "# Ratio experiment jobs submitted at $(date)" > "$JOB_LOG"
echo "# Format: method,ratio,model,train_jobid,eval_jobid" >> "$JOB_LOG"

for ratio in "${RATIOS[@]}"; do
    echo "============================================================"
    echo "Submitting jobs for target_ratio=$ratio"
    echo "============================================================"
    
    for method in "${!METHODS[@]}"; do
        model="${METHODS[$method]}"
        tag="imbal_v2_${method}_ratio${ratio//./_}"
        
        echo "  $method ($model) with ratio=$ratio -> tag=$tag"
        
        # Submit training job
        TRAIN_JOBID=$(qsub -v MODEL="$model",RATIO="$ratio",METHOD="$method",TAG="$tag" pbs_train_generic_ratio.sh)
        TRAIN_JOBID_SHORT="${TRAIN_JOBID%%.*}"
        
        sleep 1
        
        # Submit evaluation job with dependency
        EVAL_JOBID=$(qsub -W depend=afterok:$TRAIN_JOBID \
            -v MODEL="$model",TAG="$tag",TRAIN_JOBID="$TRAIN_JOBID_SHORT",SEED=42 \
            pbs_evaluate.sh)
        
        echo "$method,$ratio,$model,$TRAIN_JOBID,$EVAL_JOBID" >> "$JOB_LOG"
        
        sleep 1
    done
done

echo ""
echo "============================================================"
echo "All jobs submitted! Job log: $JOB_LOG"
echo "============================================================"
cat "$JOB_LOG" | tail -20

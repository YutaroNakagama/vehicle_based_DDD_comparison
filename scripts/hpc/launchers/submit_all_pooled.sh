#!/bin/bash
# Submit all imbalance methods for pooled mode experiments
# Date: 2025-12-10

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/imbalance

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="job_ids_all_pooled_${TIMESTAMP}.txt"

echo "# All pooled mode experiments: $(date)" > "$LOG_FILE"
echo "# Format: METHOD TRAIN_JOB EVAL_JOB" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Array of all methods and their scripts
declare -A METHODS=(
    ["baseline"]="pbs_train_baseline.sh"
    ["smote"]="pbs_train_smote.sh"
    ["smote_tomek"]="pbs_train_smote_tomek.sh"
    ["smote_enn"]="pbs_train_smote_enn.sh"
    ["smote_rus"]="pbs_train_smote_rus.sh"
    ["smote_balanced_rf"]="pbs_train_smote_balanced_rf.sh"
    ["balanced_rf"]="pbs_train_balanced_rf.sh"
    ["easy_ensemble"]="pbs_train_easy_ensemble.sh"
    ["undersample_rus"]="pbs_train_undersample_rus.sh"
    ["undersample_enn"]="pbs_train_undersample_enn.sh"
    ["undersample_tomek"]="pbs_train_undersample_tomek.sh"
)

# Submit each method
for METHOD in "${!METHODS[@]}"; do
    SCRIPT="${METHODS[$METHOD]}"
    
    if [[ ! -f "$SCRIPT" ]]; then
        echo "[SKIP] $METHOD: $SCRIPT not found"
        continue
    fi
    
    # Submit training job
    TRAIN_JOB=$(qsub "$SCRIPT" 2>&1)
    if [[ $? -ne 0 ]]; then
        echo "[ERROR] $METHOD: Failed to submit training - $TRAIN_JOB"
        continue
    fi
    TRAIN_ID=$(echo "$TRAIN_JOB" | grep -oP '^\d+')
    
    # Submit evaluation job with dependency
    EVAL_JOB=$(qsub -W depend=afterok:${TRAIN_ID} \
        -v MODEL=${METHOD%%_*},TAG=imbal_v2_${METHOD},TRAIN_JOBID=${TRAIN_ID} \
        pbs_evaluate.sh 2>&1)
    if [[ $? -ne 0 ]]; then
        echo "[WARN] $METHOD: Training submitted ($TRAIN_ID), but eval failed - $EVAL_JOB"
        echo "$METHOD $TRAIN_ID EVAL_FAILED" >> "$LOG_FILE"
        continue
    fi
    EVAL_ID=$(echo "$EVAL_JOB" | grep -oP '^\d+')
    
    echo "[OK] $METHOD: train=$TRAIN_ID, eval=$EVAL_ID"
    echo "$METHOD $TRAIN_ID $EVAL_ID" >> "$LOG_FILE"
done

echo ""
echo "Job IDs saved to: $LOG_FILE"
cat "$LOG_FILE"

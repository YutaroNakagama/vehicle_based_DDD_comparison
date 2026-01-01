#!/bin/bash
# ============================================================
# submit_optuna_convergence_experiments.sh
# 11手法 × 3比率 = 33ジョブを投入
# 収束ログ付きの再実験
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# 11手法
METHODS=(
    "baseline"
    "smote"
    "smote_tomek"
    "smote_enn"
    "smote_rus"
    "smote_balanced_rf"
    "undersample_rus"
    "undersample_tomek"
    "undersample_enn"
    "balanced_rf"
    "easy_ensemble"
)

# 3比率
RATIOS=("0.1" "0.5" "1.0")

# ジョブID記録ファイル
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_LOG="scripts/hpc/logs/imbalance/job_ids_convergence_${TIMESTAMP}.txt"

echo "# Optuna Convergence Experiments" > "$JOB_LOG"
echo "# Submitted: $(date)" >> "$JOB_LOG"
echo "" >> "$JOB_LOG"

TOTAL_JOBS=0

for METHOD in "${METHODS[@]}"; do
    echo "=== Submitting $METHOD ===" | tee -a "$JOB_LOG"
    
    for RATIO in "${RATIOS[@]}"; do
        # baseline, balanced_rf, easy_ensemble は比率不要（1.0のみ）
        if [[ "$METHOD" == "baseline" || "$METHOD" == "balanced_rf" || "$METHOD" == "easy_ensemble" ]]; then
            if [[ "$RATIO" != "1.0" ]]; then
                continue
            fi
        fi
        
        RATIO_TAG=$(echo "$RATIO" | tr '.' '_')
        TAG="optuna_conv_${METHOD}_ratio${RATIO_TAG}"
        
        JOB_ID=$(qsub -v "METHOD=$METHOD,RATIO=$RATIO,TAG=$TAG,MODEL=RF,SEED=42" \
            scripts/hpc/jobs/imbalance/pbs_train_generic_ratio.sh)
        
        echo "${METHOD}_ratio${RATIO_TAG}=$JOB_ID" | tee -a "$JOB_LOG"
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
        
        # 投入間隔
        sleep 1
    done
done

echo "" >> "$JOB_LOG"
echo "# Total jobs: $TOTAL_JOBS" >> "$JOB_LOG"
echo ""
echo "============================================================"
echo "Total jobs submitted: $TOTAL_JOBS"
echo "Job IDs saved to: $JOB_LOG"
echo "============================================================"

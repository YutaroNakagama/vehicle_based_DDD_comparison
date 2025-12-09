#!/bin/bash
#PBS -N RF_imbal_v2_eval
#PBS -J 1-16
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -q SMALL
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ==============================================================================
# pbs_eval_imbalance_v2_batch.sh
# ==============================================================================
# Purpose: Evaluate RF/BalancedRF models trained with imbalance handling methods
#          Matches structure of pbs_train_imbalance_v2.sh
#
# Imbalance Methods:
#   - baseline:     No imbalance handling (RF only)
#   - smote:        SMOTE only (oversampling)
#   - smote_enn:    SMOTE + ENN (oversampling + cleaning)
#   - balanced_rf:  BalancedRandomForest (internal balancing)
#
# Array Job Structure (1-16):
#   Jobs 1-4:   knn             × (baseline, SMOTE, SMOTE+ENN, BalancedRF)
#   Jobs 5-8:   lof             × (baseline, SMOTE, SMOTE+ENN, BalancedRF)
#   Jobs 9-12:  median_distance × (baseline, SMOTE, SMOTE+ENN, BalancedRF)
#   Jobs 13-16: pooled          × (baseline, SMOTE, SMOTE+ENN, BalancedRF)
#
# Total: 12 × 18 + 4 × 1 = 220 evaluations
# ==============================================================================

set -uo pipefail
source ~/conda/etc/profile.d/conda.sh
conda activate python310

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Configuration
RANKING_METHODS=("knn" "lof" "median_distance")
IMBALANCE_METHODS=("baseline" "smote" "smote_enn" "balanced_rf")
METRICS=("mmd" "wasserstein" "dtw")
LEVELS=("out_domain" "mid_domain" "in_domain")

# Get training job ID (required for finding model artifacts)
TRAIN_JOB_ID="${TRAIN_JOB_ID:-LATEST}"

IDX="${PBS_ARRAY_INDEX:-1}"

# Determine job type based on index
if [[ $IDX -le 12 ]]; then
    RANKING_IDX=$(( (IDX - 1) / 4 ))
    IMBALANCE_IDX=$(( (IDX - 1) % 4 ))
    
    RANKING_METHOD="${RANKING_METHODS[$RANKING_IDX]}"
    IMBALANCE_METHOD="${IMBALANCE_METHODS[$IMBALANCE_IDX]}"
    IS_POOLED=false
else
    IMBALANCE_IDX=$(( IDX - 13 ))
    IMBALANCE_METHOD="${IMBALANCE_METHODS[$IMBALANCE_IDX]}"
    RANKING_METHOD="none"
    IS_POOLED=true
fi

# Determine model type based on imbalance method
if [[ "$IMBALANCE_METHOD" == "balanced_rf" ]]; then
    MODEL="BalancedRF"
else
    MODEL="RF"
fi

RANKS_BASE="${PROJECT_ROOT}/results/domain_analysis/distance/subject-wise/ranks/ranks29"

echo "============================================================"
echo "[INFO] Imbalance Evaluation V2"
echo "[INFO] Job ID: ${PBS_JOBID:-local}"
echo "[INFO] Array Index: ${IDX}"
echo "[INFO] Ranking Method: ${RANKING_METHOD}"
echo "[INFO] Imbalance Method: ${IMBALANCE_METHOD}"
echo "[INFO] Model: ${MODEL}"
echo "[INFO] Is Pooled: ${IS_POOLED}"
echo "[INFO] Train Job ID: ${TRAIN_JOB_ID}"
echo "[INFO] Started at: $(date)"
echo "============================================================"

COMPLETED=0
FAILED=0

if [[ "$IS_POOLED" == "true" ]]; then
    # Pooled mode: single experiment
    echo ""
    echo "============================================================"
    echo "[EVAL] POOLED | ${IMBALANCE_METHOD}"
    echo "============================================================"
    
    TAG="imbalv2_pooled_${IMBALANCE_METHOD}"
    
    if python "$PROJECT_ROOT/scripts/python/evaluate.py" \
        --model "$MODEL" \
        --mode "pooled" \
        --tag "$TAG" \
        --jobid "$TRAIN_JOB_ID"; then
        COMPLETED=$((COMPLETED + 1))
        echo "[SUCCESS] Completed: $TAG"
    else
        FAILED=$((FAILED + 1))
        echo "[FAILED] Failed: $TAG"
    fi
    
    TOTAL_EXPECTED=1
else
    # Non-pooled: iterate over metrics, levels, modes
    TOTAL_EXPECTED=$((3 * 3 * 2))  # 18 experiments
    
    for METRIC in "${METRICS[@]}"; do
        for LEVEL in "${LEVELS[@]}"; do
            GROUP_FILE="${RANKS_BASE}/${RANKING_METHOD}/${METRIC}_${LEVEL}.txt"
            TAG="imbalv2_${RANKING_METHOD}_${METRIC}_${LEVEL}_${IMBALANCE_METHOD}"
            
            if [[ ! -f "$GROUP_FILE" ]]; then
                echo "[ERROR] Group file not found: $GROUP_FILE"
                FAILED=$((FAILED + 2))
                continue
            fi
            
            for MODE in "source_only" "target_only"; do
                echo ""
                echo "============================================================"
                echo "[EVAL] ${RANKING_METHOD} | ${METRIC} | ${LEVEL} | ${MODE} | ${IMBALANCE_METHOD}"
                echo "[INFO] Experiment $((COMPLETED + FAILED + 1))/${TOTAL_EXPECTED}"
                echo "============================================================"
                
                if python "$PROJECT_ROOT/scripts/python/evaluate.py" \
                    --model "$MODEL" \
                    --mode "$MODE" \
                    --target_file "$GROUP_FILE" \
                    --tag "$TAG" \
                    --jobid "$TRAIN_JOB_ID"; then
                    COMPLETED=$((COMPLETED + 1))
                    echo "[SUCCESS] Completed: $TAG ($MODE)"
                else
                    FAILED=$((FAILED + 1))
                    echo "[FAILED] Failed: $TAG ($MODE)"
                fi
            done
        done
    done
fi

echo ""
echo "============================================================"
echo "[SUMMARY] Evaluation Complete"
echo "============================================================"
echo "  Model:            ${MODEL}"
echo "  Ranking Method:   ${RANKING_METHOD}"
echo "  Imbalance Method: ${IMBALANCE_METHOD}"
echo "  Completed: $COMPLETED / $TOTAL_EXPECTED"
echo "  Failed:    $FAILED / $TOTAL_EXPECTED"
echo "  Finished:  $(date)"
echo "============================================================"

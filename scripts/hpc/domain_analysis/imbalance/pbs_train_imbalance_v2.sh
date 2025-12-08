#!/bin/bash
#PBS -N RF_imbal_v2
#PBS -J 1-16
#PBS -l select=1:ncpus=8:mem=256gb
#PBS -l walltime=168:00:00
#PBS -j oe
#PBS -q SMALL
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ==============================================================================
# pbs_train_imbalance_v2.sh
# ==============================================================================
# Purpose: Train RF/BalancedRF model with different imbalance handling methods
#          3 ranking methods × 4 imbalance methods = 12 array jobs
#          + 4 pooled jobs (one per imbalance method)
#
# Imbalance Methods (updated 2025-12-08):
#   - baseline:     No imbalance handling (RF only)
#   - smote:        SMOTE only (oversampling)
#   - smote_enn:    SMOTE + ENN (oversampling + cleaning)
#   - balanced_rf:  BalancedRandomForest (internal balancing, no oversampling)
#
# Array Job Structure (1-16):
#   Jobs 1-4:   knn             × (baseline, SMOTE, SMOTE+ENN, BalancedRF)
#   Jobs 5-8:   lof             × (baseline, SMOTE, SMOTE+ENN, BalancedRF)
#   Jobs 9-12:  median_distance × (baseline, SMOTE, SMOTE+ENN, BalancedRF)
#   Jobs 13-16: pooled          × (baseline, SMOTE, SMOTE+ENN, BalancedRF)
#
# Per non-pooled job: 3 metrics × 3 levels × 2 modes = 18 experiments
# Per pooled job: 1 experiment
#
# Total: 12 × 18 + 4 × 1 = 216 + 4 = 220 experiments
#
# Note: This version uses F2 for threshold optimization (fixed from F0.5 bug)
# ==============================================================================

set -uo pipefail
source ~/conda/etc/profile.d/conda.sh
conda activate python310

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SEED="${SEED:-42}"

# Reduce N_TRIALS to avoid memory issues
export N_TRIALS_OVERRIDE=20

# Thread control for stability
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Configuration
RANKING_METHODS=("knn" "lof" "median_distance")
# Updated imbalance methods: baseline, SMOTE, SMOTE+ENN, BalancedRF
IMBALANCE_METHODS=("baseline" "smote" "smote_enn" "balanced_rf")
METRICS=("mmd" "wasserstein" "dtw")
LEVELS=("out_domain" "mid_domain" "in_domain")
MODES=("source_only" "target_only")

IDX="${PBS_ARRAY_INDEX:-1}"

# Determine job type based on index
# Jobs 1-12: ranking method + imbalance method combinations
# Jobs 13-16: pooled only

if [[ $IDX -le 12 ]]; then
    # Non-pooled jobs
    RANKING_IDX=$(( (IDX - 1) / 4 ))
    IMBALANCE_IDX=$(( (IDX - 1) % 4 ))
    
    RANKING_METHOD="${RANKING_METHODS[$RANKING_IDX]}"
    IMBALANCE_METHOD="${IMBALANCE_METHODS[$IMBALANCE_IDX]}"
    IS_POOLED=false
else
    # Pooled jobs (13, 14, 15, 16)
    IMBALANCE_IDX=$(( IDX - 13 ))
    IMBALANCE_METHOD="${IMBALANCE_METHODS[$IMBALANCE_IDX]}"
    RANKING_METHOD="none"
    IS_POOLED=true
fi

RANKS_BASE="${PROJECT_ROOT}/results/domain_analysis/distance/subject-wise/ranks/ranks29"

echo "============================================================"
echo "[INFO] Imbalance Training V2 (F2-optimized)"
echo "[INFO] Job ID: ${PBS_JOBID:-local}"
echo "[INFO] Array Index: ${IDX}"
echo "[INFO] Ranking Method: ${RANKING_METHOD}"
echo "[INFO] Imbalance Method: ${IMBALANCE_METHOD}"
echo "[INFO] Is Pooled: ${IS_POOLED}"
echo "[INFO] Started at: $(date)"
echo "============================================================"

# Helper function to run training
run_training() {
    local MODEL="$1"
    local MODE="$2"
    local TAG="$3"
    local TARGET_FILE="$4"
    local USE_OVERSAMPLING="$5"
    local OVERSAMPLE_METHOD="$6"
    
    local CMD=(python "$PROJECT_ROOT/scripts/python/train.py"
        --model "$MODEL"
        --mode "$MODE"
        --tag "$TAG"
        --seed "$SEED"
        --time_stratify_labels
    )
    
    # Add target file if specified
    if [[ -n "$TARGET_FILE" ]]; then
        CMD+=(--target_file "$TARGET_FILE")
    fi
    
    # Add oversampling if needed
    if [[ "$USE_OVERSAMPLING" == "true" ]]; then
        CMD+=(--use_oversampling --oversample_method "$OVERSAMPLE_METHOD")
    fi
    
    echo "[CMD] ${CMD[*]}"
    "${CMD[@]}"
}

COMPLETED=0
FAILED=0

if [[ "$IS_POOLED" == "true" ]]; then
    # Pooled mode: single experiment
    echo ""
    echo "============================================================"
    echo "[TRAIN] POOLED | ${IMBALANCE_METHOD}"
    echo "============================================================"
    
    TAG="imbalv2_pooled_${IMBALANCE_METHOD}"
    
    if [[ "$IMBALANCE_METHOD" == "balanced_rf" ]]; then
        # BalancedRF: use BalancedRF model, no oversampling
        if run_training "BalancedRF" "pooled" "$TAG" "" "false" ""; then
            COMPLETED=$((COMPLETED + 1))
            echo "[SUCCESS] Completed: $TAG"
        else
            FAILED=$((FAILED + 1))
            echo "[FAILED] Failed: $TAG"
        fi
    elif [[ "$IMBALANCE_METHOD" == "baseline" ]]; then
        # Baseline: use RF model, no oversampling
        if run_training "RF" "pooled" "$TAG" "" "false" ""; then
            COMPLETED=$((COMPLETED + 1))
            echo "[SUCCESS] Completed: $TAG"
        else
            FAILED=$((FAILED + 1))
            echo "[FAILED] Failed: $TAG"
        fi
    else
        # SMOTE or SMOTE+ENN: use RF model with oversampling
        if run_training "RF" "pooled" "$TAG" "" "true" "$IMBALANCE_METHOD"; then
            COMPLETED=$((COMPLETED + 1))
            echo "[SUCCESS] Completed: $TAG"
        else
            FAILED=$((FAILED + 1))
            echo "[FAILED] Failed: $TAG"
        fi
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
                FAILED=$((FAILED + 2))  # Count both modes as failed
                continue
            fi
            
            for MODE in "${MODES[@]}"; do
                echo ""
                echo "============================================================"
                echo "[TRAIN] ${RANKING_METHOD} | ${METRIC} | ${LEVEL} | ${MODE} | ${IMBALANCE_METHOD}"
                echo "[INFO] Experiment $((COMPLETED + FAILED + 1))/${TOTAL_EXPECTED}"
                echo "============================================================"
                
                if [[ "$IMBALANCE_METHOD" == "balanced_rf" ]]; then
                    # BalancedRF: use BalancedRF model, no oversampling
                    if run_training "BalancedRF" "$MODE" "$TAG" "$GROUP_FILE" "false" ""; then
                        COMPLETED=$((COMPLETED + 1))
                        echo "[SUCCESS] Completed: $TAG ($MODE)"
                    else
                        FAILED=$((FAILED + 1))
                        echo "[FAILED] Failed: $TAG ($MODE)"
                    fi
                elif [[ "$IMBALANCE_METHOD" == "baseline" ]]; then
                    # Baseline: use RF model, no oversampling
                    if run_training "RF" "$MODE" "$TAG" "$GROUP_FILE" "false" ""; then
                        COMPLETED=$((COMPLETED + 1))
                        echo "[SUCCESS] Completed: $TAG ($MODE)"
                    else
                        FAILED=$((FAILED + 1))
                        echo "[FAILED] Failed: $TAG ($MODE)"
                    fi
                else
                    # SMOTE or SMOTE+ENN: use RF model with oversampling
                    if run_training "RF" "$MODE" "$TAG" "$GROUP_FILE" "true" "$IMBALANCE_METHOD"; then
                        COMPLETED=$((COMPLETED + 1))
                        echo "[SUCCESS] Completed: $TAG ($MODE)"
                    else
                        FAILED=$((FAILED + 1))
                        echo "[FAILED] Failed: $TAG ($MODE)"
                    fi
                fi
            done
        done
    done
fi

echo ""
echo "============================================================"
echo "[SUMMARY] Training Complete"
echo "============================================================"
echo "  Ranking Method:   ${RANKING_METHOD}"
echo "  Imbalance Method: ${IMBALANCE_METHOD}"
echo "  Completed: $COMPLETED / $TOTAL_EXPECTED"
echo "  Failed:    $FAILED / $TOTAL_EXPECTED"
echo "  Finished:  $(date)"
echo "============================================================"

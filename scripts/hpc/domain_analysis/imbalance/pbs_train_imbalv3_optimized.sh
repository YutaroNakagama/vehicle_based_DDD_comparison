#!/bin/bash
#PBS -N imbalv3_train
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q SINGLE
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/logs/hpc/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/logs/hpc/

# =============================================================================
# pbs_train_imbalv3_optimized.sh - Optimized imbalv3 Training
# =============================================================================
# Array structure: 36 jobs (12 imbalance configs × 3 ranking methods)
# Each job: 18 experiments (3 metrics × 3 levels × 2 modes)
#
# Total: 36 × 18 = 648 experiments
#
# Submit: qsub -J 1-36 scripts/hpc/domain_analysis/imbalance/pbs_train_imbalv3_optimized.sh
# =============================================================================

set -uo pipefail
source ~/conda/etc/profile.d/conda.sh
conda activate python310

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SEED="${SEED:-42}"
export N_TRIALS_OVERRIDE=20

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Ranking methods (no pooled - it's independent of domain analysis)
RANKING_METHODS=("knn" "lof" "median_distance")
METRICS=("mmd" "wasserstein" "dtw")
LEVELS=("out_domain" "mid_domain" "in_domain")
MODES=("source_only" "target_only")

# Imbalance configurations: (method, ratio, model)
# 4 methods × 3 ratios = 12 configs
declare -a IMBALANCE_CONFIGS=(
    "smote:0.1:RF"
    "smote:0.5:RF"
    "smote:1.0:RF"
    "smote_tomek:0.1:RF"
    "smote_tomek:0.5:RF"
    "smote_tomek:1.0:RF"
    "smote_balanced_rf:0.1:BalancedRF"
    "smote_balanced_rf:0.5:BalancedRF"
    "smote_balanced_rf:1.0:BalancedRF"
    "undersample_rus:0.1:RF"
    "undersample_rus:0.5:RF"
    "undersample_rus:1.0:RF"
)

IDX="${PBS_ARRAY_INDEX:-1}"

# Calculate config and ranking indices
# 12 configs × 3 rankings = 36 jobs
CONFIG_IDX=$(( (IDX - 1) / 3 ))
RANK_IDX=$(( (IDX - 1) % 3 ))

CONFIG="${IMBALANCE_CONFIGS[$CONFIG_IDX]}"
RANKING_METHOD="${RANKING_METHODS[$RANK_IDX]}"

IFS=':' read -r IMBALANCE_METHOD TARGET_RATIO MODEL_TYPE <<< "$CONFIG"

RANKS_BASE="${PROJECT_ROOT}/results/domain_analysis/distance/subject-wise/ranks/ranks29"

# Create ratio tag
RATIO_TAG="ratio${TARGET_RATIO//./_}"

echo "============================================================"
echo "[INFO] imbalv3 Training (Optimized)"
echo "[INFO] Job ID: ${PBS_JOBID:-local}"
echo "[INFO] Array Index: ${IDX}"
echo "[INFO] Imbalance: ${IMBALANCE_METHOD} | Ratio: ${TARGET_RATIO}"
echo "[INFO] Model: ${MODEL_TYPE} | Ranking: ${RANKING_METHOD}"
echo "[INFO] Started: $(date)"
echo "============================================================"

# Helper function
run_training() {
    local MODEL="$1"
    local MODE="$2"
    local TAG="$3"
    local TARGET_FILE="$4"
    local USE_OVERSAMPLING="$5"
    local OVERSAMPLE_METHOD="$6"
    local RATIO="$7"
    
    local CMD=(python "$PROJECT_ROOT/scripts/python/train.py"
        --model "$MODEL"
        --mode "$MODE"
        --tag "$TAG"
        --seed "$SEED"
        --time_stratify_labels
    )
    
    if [[ -n "$TARGET_FILE" ]]; then
        CMD+=(--target_file "$TARGET_FILE")
    fi
    
    if [[ "$USE_OVERSAMPLING" == "true" ]]; then
        CMD+=(--use_oversampling --oversample_method "$OVERSAMPLE_METHOD")
        if [[ -n "$RATIO" ]]; then
            CMD+=(--target_ratio "$RATIO")
        fi
    fi
    
    echo "[CMD] ${CMD[*]}"
    "${CMD[@]}"
}

get_oversampling_settings() {
    local method="$1"
    case "$method" in
        "smote_balanced_rf")
            echo "true:smote"
            ;;
        "undersample_rus")
            echo "true:undersample_rus"
            ;;
        *)
            echo "true:$method"
            ;;
    esac
}

COMPLETED=0
FAILED=0
TOTAL_EXPECTED=18

OVERSAMPLE_SETTINGS=$(get_oversampling_settings "$IMBALANCE_METHOD")
IFS=':' read -r USE_OVERSAMPLING OVERSAMPLE_METHOD_ACTUAL <<< "$OVERSAMPLE_SETTINGS"

for METRIC in "${METRICS[@]}"; do
    for LEVEL in "${LEVELS[@]}"; do
        GROUP_FILE="${RANKS_BASE}/${RANKING_METHOD}/${METRIC}_${LEVEL}.txt"
        TAG="imbalv3_${RANKING_METHOD}_${METRIC}_${LEVEL}_${IMBALANCE_METHOD}_${RATIO_TAG}"
        
        if [[ ! -f "$GROUP_FILE" ]]; then
            echo "[ERROR] Group file not found: $GROUP_FILE"
            FAILED=$((FAILED + 2))
            continue
        fi
        
        for MODE in "${MODES[@]}"; do
            echo ""
            echo "[TRAIN] ${RANKING_METHOD} | ${METRIC} | ${LEVEL} | ${MODE}"
            echo "[INFO] Experiment $((COMPLETED + FAILED + 1))/${TOTAL_EXPECTED}"
            
            if run_training "$MODEL_TYPE" "$MODE" "$TAG" "$GROUP_FILE" "$USE_OVERSAMPLING" "$OVERSAMPLE_METHOD_ACTUAL" "$TARGET_RATIO"; then
                COMPLETED=$((COMPLETED + 1))
                echo "[SUCCESS] Completed: $TAG ($MODE)"
            else
                FAILED=$((FAILED + 1))
                echo "[FAILED] Failed: $TAG ($MODE)"
            fi
        done
    done
done

echo ""
echo "============================================================"
echo "[SUMMARY] Training Complete"
echo "============================================================"
echo "  Imbalance: ${IMBALANCE_METHOD} ${RATIO_TAG}"
echo "  Ranking:   ${RANKING_METHOD}"
echo "  Completed: $COMPLETED / $TOTAL_EXPECTED"
echo "  Failed:    $FAILED / $TOTAL_EXPECTED"
echo "  Finished:  $(date)"
echo "============================================================"

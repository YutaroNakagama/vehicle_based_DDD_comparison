#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/domain/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/domain/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Re-evaluation Script for split2 domain experiments (v2)
# ============================================================
# Runs evaluation only (no training) with:
#   - --target_file (Bug #1 fix)
#   - --seed        (Bug #1 fix)
#   - --jobid       (Bug #4 fix — load correct model)
#
# Environment Variables:
#   CONDITION : baseline | smote | smote_plain | undersample | balanced_rf (required)
#   MODE      : source_only | target_only | mixed (required)
#   DISTANCE  : mmd | wasserstein | dtw (required)
#   DOMAIN    : in_domain | out_domain (required)
#   RATIO     : Target ratio for SMOTE (default: 0.5)
#   SEED      : Random seed (default: 42)
#   RANKING   : Ranking method (default: knn)
#   TRAIN_JOBID : PBS job ID of the training job (required for Bug #4 fix)
# ============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Environment setup
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

# Parameters
CONDITION="${CONDITION:-baseline}"
MODE="${MODE:-source_only}"
DISTANCE="${DISTANCE:-mmd}"
DOMAIN="${DOMAIN:-in_domain}"
RATIO="${RATIO:-0.5}"
SEED="${SEED:-42}"
RANKING="${RANKING:-knn}"
TRAIN_JOBID="${TRAIN_JOBID:-}"

# Auto-select model based on condition
case "$CONDITION" in
    balanced_rf) MODEL="BalancedRF" ;;
    *)           MODEL="RF" ;;
esac

# Target file path (split2 directory)
TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${DOMAIN}.txt"

# Generate tag (must match the training tag exactly)
case "$CONDITION" in
    baseline)
        TAG="baseline_domain_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_s${SEED}"
        ;;
    smote)
        TAG="imbalv3_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_subjectwise_ratio${RATIO}_s${SEED}"
        ;;
    smote_plain)
        TAG="smote_plain_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_ratio${RATIO}_s${SEED}"
        ;;
    undersample)
        TAG="undersample_rus_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_ratio${RATIO}_s${SEED}"
        ;;
    balanced_rf)
        TAG="balanced_rf_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_s${SEED}"
        ;;
    *)
        echo "[ERROR] Unknown condition: $CONDITION"
        exit 1
        ;;
esac

echo "============================================================"
echo "[RE-EVAL v2 - SPLIT2] ${CONDITION^^} / ${MODE}"
echo "============================================================"
echo "CONDITION: $CONDITION"
echo "MODE: $MODE"
echo "DISTANCE: $DISTANCE"
echo "DOMAIN: $DOMAIN"
echo "SEED: $SEED"
echo "TAG: $TAG"
echo "TARGET_FILE: $TARGET_FILE"
echo "TRAIN_JOBID: ${TRAIN_JOBID:-auto}"
echo "============================================================"

# Verify target file exists
if [[ ! -f "$TARGET_FILE" ]]; then
    echo "[ERROR] Target file not found: $TARGET_FILE"
    exit 1
fi

EVAL_CMD="python scripts/python/evaluation/evaluate.py \
    --model $MODEL \
    --tag $TAG \
    --mode $MODE \
    --target_file $TARGET_FILE \
    --seed $SEED"

# Add --jobid if specified (Bug #4 fix)
if [[ -n "$TRAIN_JOBID" ]]; then
    EVAL_CMD="$EVAL_CMD --jobid $TRAIN_JOBID"
fi

echo ""
echo "[EVAL] $EVAL_CMD"
echo ""

eval $EVAL_CMD

echo ""
echo "[DONE] Re-evaluation completed successfully"
echo "============================================================"

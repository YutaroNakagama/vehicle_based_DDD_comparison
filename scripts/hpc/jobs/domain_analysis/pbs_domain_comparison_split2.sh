#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/domain/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/domain/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
# Note: -N, -l select, -l walltime, -q are passed dynamically via qsub options

# ============================================================
# Domain Analysis Comparison Script (2グループ分割版)
# ============================================================
# split2ディレクトリを使用した2グループ分割実験
# in_domain (44名) vs out_domain (43名)
#
# Environment Variables:
#   CONDITION : baseline | smote | smote_plain | undersample | balanced_rf (required)
#   MODE      : source_only | target_only | mixed (required)
#   DISTANCE  : mmd | wasserstein | dtw (required)
#   DOMAIN    : in_domain | out_domain (required) ※mid_domainなし
#   RATIO     : Target ratio for SMOTE (default: 0.5)
#   SEED      : Random seed (default: 42)
#   N_TRIALS  : Optuna trials (default: 100)
#   RANKING   : Ranking method (default: knn)
#   RUN_EVAL  : Run evaluation after training (default: true)
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

# Thread optimization for HPC
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# Parameters
CONDITION="${CONDITION:-smote}"
MODE="${MODE:-source_only}"
DISTANCE="${DISTANCE:-mmd}"
DOMAIN="${DOMAIN:-in_domain}"
RATIO="${RATIO:-0.5}"
SEED="${SEED:-42}"
RANKING="${RANKING:-knn}"
RUN_EVAL="${RUN_EVAL:-true}"
export N_TRIALS_OVERRIDE="${N_TRIALS:-100}"

# Validate DOMAIN (mid_domain not supported in split2)
if [[ "$DOMAIN" == "mid_domain" ]]; then
    echo "[ERROR] mid_domain is not supported in split2. Use in_domain or out_domain."
    exit 1
fi

# Validate MODE
if [[ "$MODE" != "source_only" && "$MODE" != "target_only" && "$MODE" != "mixed" ]]; then
    echo "[ERROR] Unknown MODE: $MODE. Use source_only, target_only, or mixed."
    exit 1
fi

# Auto-select model based on condition
case "$CONDITION" in
    balanced_rf)
        MODEL="BalancedRF"
        ;;
    *)
        MODEL="RF"
        ;;
esac

# Target file path (split2 directory)
# For mixed mode, target file defines the EVALUATION domain
TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${DOMAIN}.txt"

# Generate tag based on condition
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
echo "[DOMAIN COMPARISON - SPLIT2] ${CONDITION^^}"
echo "============================================================"
echo "CONDITION: $CONDITION"
echo "MODE: $MODE"
echo "DISTANCE: $DISTANCE"
echo "DOMAIN: $DOMAIN"
echo "RATIO: $RATIO"
echo "SEED: $SEED"
echo "RANKING: $RANKING"
echo "TAG: $TAG"
echo "TARGET_FILE: $TARGET_FILE"
echo "N_TRIALS: $N_TRIALS_OVERRIDE"
echo "RUN_EVAL: $RUN_EVAL"
echo "JOBID: $PBS_JOBID"
echo "SPLIT: split2 (2-group)"
echo "============================================================"

# Verify target file exists
if [[ ! -f "$TARGET_FILE" ]]; then
    echo "[ERROR] Target file not found: $TARGET_FILE"
    exit 1
fi
SUBJECT_COUNT=$(wc -l < "$TARGET_FILE")
echo "[INFO] Subject count: $SUBJECT_COUNT"

# Validate subject count (should be 43 or 44 for split2)
if [[ "$SUBJECT_COUNT" -ne 43 && "$SUBJECT_COUNT" -ne 44 ]]; then
    echo "[WARNING] Unexpected subject count for split2: $SUBJECT_COUNT (expected 43 or 44)"
fi

# Build training command
CMD="python scripts/python/train/train.py \
    --model $MODEL \
    --mode $MODE \
    --seed $SEED \
    --target_file $TARGET_FILE \
    --tag $TAG \
    --time_stratify_labels"

case "$CONDITION" in
    baseline)
        # No oversampling
        ;;
    smote)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO --subject_wise_oversampling"
        ;;
    smote_plain)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO"
        ;;
    undersample)
        CMD="$CMD --use_oversampling --oversample_method undersample_rus --target_ratio $RATIO"
        ;;
    balanced_rf)
        # BalancedRF handles class imbalance internally, no additional sampling needed
        ;;
esac

echo ""
echo "[TRAIN] $CMD"
echo ""
eval $CMD

# Run evaluation if requested
if [[ "$RUN_EVAL" == "true" ]]; then
    echo ""
    echo "[EVAL] Running evaluation..."
    echo ""
    
    # Strip PBS domain suffix (e.g. "14738867.spcc-adm1" -> "14738867")
    JOBID_CLEAN="${PBS_JOBID%%.*}"

    EVAL_CMD="python scripts/python/evaluation/evaluate.py \
        --model $MODEL \
        --tag $TAG \
        --mode $MODE \
        --target_file $TARGET_FILE \
        --seed $SEED \
        --jobid $JOBID_CLEAN"
    
    echo "[EVAL] $EVAL_CMD"
    eval $EVAL_CMD || echo "[WARNING] Evaluation failed but continuing..."
fi

echo ""
echo "[DONE] Job completed successfully"
echo "============================================================"

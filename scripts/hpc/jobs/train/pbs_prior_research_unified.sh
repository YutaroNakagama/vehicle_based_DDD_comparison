#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
# Note: -N, -l select, -l walltime, -q are passed dynamically via qsub options

# ============================================================
# 先行研究実験 (統一版) - 1回の学習 + 2回の評価
# ============================================================
# 変更点（split2との違い）:
#   - 学習は各ドメインに対し1回のみ（domain_trainモード）
#   - 分割比率: train(70%) / val(15%) / test(15%)
#   - 評価①: within-domain → 同ドメインのtest(15%)で評価
#   - 評価②: cross-domain  → 逆ドメインのtest(15%)で評価
#   - ジョブ数が半分に削減（source_only/target_only の重複トレーニング解消）
#
# Environment Variables:
#   MODEL      : SvmA | SvmW | Lstm (required)
#   CONDITION  : baseline | smote | smote_plain | undersample (required)
#   DISTANCE   : mmd | wasserstein | dtw (required)
#   DOMAIN     : in_domain | out_domain (required)
#   RATIO      : Target ratio for SMOTE (default: 0.5)
#   SEED       : Random seed (default: 42)
#   N_TRIALS   : Optuna trials for SvmW (default: 100)
#   RANKING    : Ranking method (default: knn)
#   RUN_EVAL   : Run evaluation after training (default: true)
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
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""  # Force CPU for reproducibility

# Parameters
MODEL="${MODEL:-SvmW}"
CONDITION="${CONDITION:-baseline}"
DISTANCE="${DISTANCE:-mmd}"
DOMAIN="${DOMAIN:-in_domain}"
RATIO="${RATIO:-0.5}"
SEED="${SEED:-42}"
RANKING="${RANKING:-knn}"
RUN_EVAL="${RUN_EVAL:-true}"
export N_TRIALS_OVERRIDE="${N_TRIALS:-100}"

# Validate MODEL
if [[ "$MODEL" != "SvmA" && "$MODEL" != "SvmW" && "$MODEL" != "Lstm" ]]; then
    echo "[ERROR] Invalid MODEL: $MODEL. Must be SvmA, SvmW, or Lstm."
    exit 1
fi

# Validate CONDITION for each model
case "$MODEL" in
    SvmA|Lstm)
        if [[ "$CONDITION" == "balanced_rf" ]]; then
            echo "[ERROR] $MODEL does not support balanced_rf condition."
            exit 1
        fi
        ;;
esac

# Target file path (split2 directory - same domain grouping)
TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${DOMAIN}.txt"

# Cross-domain target file (opposite domain)
if [[ "$DOMAIN" == "in_domain" ]]; then
    CROSS_DOMAIN="out_domain"
else
    CROSS_DOMAIN="in_domain"
fi
CROSS_TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${CROSS_DOMAIN}.txt"

# Generate tag (unified: domain_train replaces source_only/target_only)
case "$CONDITION" in
    baseline)
        TAG="prior_${MODEL}_baseline_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_s${SEED}"
        ;;
    smote)
        TAG="prior_${MODEL}_imbalv3_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_subjectwise_ratio${RATIO}_s${SEED}"
        ;;
    smote_plain)
        TAG="prior_${MODEL}_smote_plain_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_ratio${RATIO}_s${SEED}"
        ;;
    undersample)
        TAG="prior_${MODEL}_undersample_rus_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_ratio${RATIO}_s${SEED}"
        ;;
    *)
        echo "[ERROR] Unknown condition: $CONDITION"
        exit 1
        ;;
esac

echo "============================================================"
echo "[UNIFIED] ${MODEL} - ${CONDITION^^} - domain_train"
echo "============================================================"
echo "MODEL: $MODEL"
echo "CONDITION: $CONDITION"
echo "MODE: domain_train (train once, eval twice)"
echo "DISTANCE: $DISTANCE"
echo "DOMAIN: $DOMAIN (train/val/test from same domain)"
echo "CROSS_DOMAIN: $CROSS_DOMAIN (cross-domain test)"
echo "RATIO: $RATIO"
echo "SEED: $SEED"
echo "RANKING: $RANKING"
echo "TAG: $TAG"
echo "TARGET_FILE: $TARGET_FILE"
echo "CROSS_TARGET_FILE: $CROSS_TARGET_FILE"
echo "N_TRIALS: $N_TRIALS_OVERRIDE (for SvmW)"
echo "RUN_EVAL: $RUN_EVAL"
echo "JOBID: $PBS_JOBID"
echo "SPLIT: 70/15/15 (train/val/test)"
echo "============================================================"

# Verify target files exist
if [[ ! -f "$TARGET_FILE" ]]; then
    echo "[ERROR] Target file not found: $TARGET_FILE"
    exit 1
fi
if [[ ! -f "$CROSS_TARGET_FILE" ]]; then
    echo "[ERROR] Cross-domain target file not found: $CROSS_TARGET_FILE"
    exit 1
fi

SUBJECT_COUNT=$(wc -l < "$TARGET_FILE")
CROSS_SUBJECT_COUNT=$(wc -l < "$CROSS_TARGET_FILE")
echo "[INFO] Domain subjects: $SUBJECT_COUNT | Cross-domain subjects: $CROSS_SUBJECT_COUNT"

# Build training command (train once on DOMAIN with 70/15/15 split)
CMD="python scripts/python/train/train.py \
    --model $MODEL \
    --mode domain_train \
    --seed $SEED \
    --target_file $TARGET_FILE \
    --tag $TAG \
    --time_stratify_labels"

# Add condition-specific flags
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
esac

echo ""
echo "[TRAIN] $CMD"
echo ""
eval $CMD

EXIT_CODE=$?

# Run evaluations if requested and training succeeded
if [[ "$RUN_EVAL" == "true" && $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "============================================================"
    echo "[EVAL 1/2] Within-domain evaluation (${DOMAIN} test 15%)"
    echo "============================================================"
    
    EVAL_CMD_WD="python scripts/python/evaluation/evaluate.py \
        --model $MODEL \
        --tag $TAG \
        --mode domain_train \
        --target_file $TARGET_FILE \
        --eval_type within \
        --jobid $PBS_JOBID"
    
    echo "[EVAL-WD] $EVAL_CMD_WD"
    eval $EVAL_CMD_WD || echo "[WARNING] Within-domain evaluation failed but continuing..."
    
    echo ""
    echo "============================================================"
    echo "[EVAL 2/2] Cross-domain evaluation (${CROSS_DOMAIN} test 15%)"
    echo "============================================================"
    
    EVAL_CMD_CD="python scripts/python/evaluation/evaluate.py \
        --model $MODEL \
        --tag ${TAG} \
        --mode domain_train \
        --target_file $CROSS_TARGET_FILE \
        --eval_type cross \
        --jobid $PBS_JOBID"
    
    echo "[EVAL-CD] $EVAL_CMD_CD"
    eval $EVAL_CMD_CD || echo "[WARNING] Cross-domain evaluation failed but continuing..."
fi

echo ""
echo "[DONE] Unified job completed (exit code: $EXIT_CODE)"
echo "  Training domain: $DOMAIN"
echo "  Within-domain eval: $DOMAIN test (15%)"
echo "  Cross-domain eval: $CROSS_DOMAIN test (15%)"
echo "============================================================"

exit $EXIT_CODE

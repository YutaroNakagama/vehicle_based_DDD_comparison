#!/bin/bash
#PBS -N Ls_array
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m a
# Note: -J range, -q, and -l are passed via qsub options

# ============================================================
# PBS Array Job: Lstm GPU (unified: domain_train + within/cross eval)
# ============================================================
# Each array element picks one task line from TASK_FILE
# using $PBS_ARRAY_INDEX as the 0-based line index.
#
# Submit example:
#   qsub -J 0-3 -q GPU-1 \
#     -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 \
#     -v TASK_FILE=scripts/hpc/logs/train/task_files/array_lstm_domain_train_remaining.txt \
#     scripts/hpc/jobs/train/pbs_array_lstm_gpu.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Environment setup
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PBS_JOBID="${PBS_JOBID:-manual}"

# Load CUDA via hpc_sdk module
module load hpc_sdk/22.2 2>/dev/null || true

# Thread control
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

TASK_FILE="${TASK_FILE:-}"
if [[ -z "$TASK_FILE" || ! -f "$TASK_FILE" ]]; then
    echo "[ERROR] TASK_FILE not set or not found: '$TASK_FILE'"
    exit 1
fi

# Read the task line at PBS_ARRAY_INDEX (0-based, sed is 1-based)
LINE_NUM=$((PBS_ARRAY_INDEX + 1))
TASK_LINE=$(sed -n "${LINE_NUM}p" "$TASK_FILE")

if [[ -z "$TASK_LINE" ]]; then
    echo "[ERROR] No task at index $PBS_ARRAY_INDEX (line $LINE_NUM)"
    exit 1
fi

# Parse: MODEL|CONDITION|MODE|DISTANCE|DOMAIN|RATIO|SEED|N_TRIALS|RANKING|RUN_EVAL|SCRIPT_TYPE
IFS='|' read -r MODEL CONDITION MODE DISTANCE DOMAIN RATIO SEED N_TRIALS RANKING RUN_EVAL SCRIPT_TYPE <<< "$TASK_LINE"

echo "============================================================"
echo "[ARRAY-GPU] Index: $PBS_ARRAY_INDEX | Job: $PBS_JOBID"
echo "[ARRAY-GPU] $MODEL | $CONDITION | $MODE | $DISTANCE | $DOMAIN | ratio=$RATIO | s$SEED"
echo "============================================================"

# Validate MODEL
if [[ "$MODEL" != "Lstm" ]]; then
    echo "[ERROR] GPU array script is for Lstm only, got: $MODEL"
    exit 1
fi

# Target file
TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${DOMAIN}.txt"
if [[ ! -f "$TARGET_FILE" ]]; then
    echo "[ERROR] Target file not found: $TARGET_FILE"
    exit 1
fi

# Generate tag
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

echo "[TRAIN] Tag: $TAG"
echo "[TRAIN] Start: $(date)"

# Build training command
CMD="python scripts/python/train/train.py \
    --model $MODEL \
    --mode domain_train \
    --seed $SEED \
    --target_file $TARGET_FILE \
    --tag $TAG \
    --time_stratify_labels"

case "$CONDITION" in
    baseline) ;;
    smote)       CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO --subject_wise_oversampling" ;;
    smote_plain) CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO" ;;
    undersample) CMD="$CMD --use_oversampling --oversample_method undersample_rus --target_ratio $RATIO" ;;
esac

eval $CMD
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[FAIL] Training failed (exit=$EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi

echo "[TRAIN] Done at $(date)"

# Evaluation (within + cross)
if [[ "$RUN_EVAL" == "true" ]]; then
    echo "[EVAL] Within-domain..."
    python scripts/python/evaluation/evaluate.py \
        --model "$MODEL" --tag "$TAG" --mode domain_train \
        --target_file "$TARGET_FILE" --eval_type within \
        --jobid "$PBS_JOBID" || echo "[WARN] Within-domain eval failed"

    # Cross-domain
    if [[ "$DOMAIN" == "in_domain" ]]; then
        CROSS_DOMAIN="out_domain"
    else
        CROSS_DOMAIN="in_domain"
    fi
    CROSS_TARGET="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${CROSS_DOMAIN}.txt"

    echo "[EVAL] Cross-domain..."
    python scripts/python/evaluation/evaluate.py \
        --model "$MODEL" --tag "$TAG" --mode domain_train \
        --target_file "$CROSS_TARGET" --eval_type cross \
        --jobid "$PBS_JOBID" || echo "[WARN] Cross-domain eval failed"
fi

echo "[DONE] Array index $PBS_ARRAY_INDEX completed at $(date)"
exit 0

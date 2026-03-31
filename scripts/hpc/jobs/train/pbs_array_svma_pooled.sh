#!/bin/bash
#PBS -N Sa_pooled
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m a
# Note: -J range and -q are passed via qsub options

# ============================================================
# PBS Array Job: SvmA pooled (train + eval, all subjects)
# ============================================================
# Task file format: MODEL|CONDITION|MODE||RATIO|SEED
#   SvmA|baseline|pooled|||42
#   SvmA|smote|pooled||0.5|42
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
export CUDA_VISIBLE_DEVICES=""

# Thread control
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

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

# Parse: MODEL|CONDITION|MODE||RATIO|SEED
IFS='|' read -r MODEL CONDITION MODE _UNUSED RATIO SEED <<< "$TASK_LINE"

echo "============================================================"
echo "[POOLED] Index: $PBS_ARRAY_INDEX | Job: $PBS_JOBID"
echo "[POOLED] $MODEL | $CONDITION | ratio=$RATIO | s$SEED"
echo "============================================================"

# Generate tag
case "$CONDITION" in
    baseline)
        TAG="prior_${MODEL}_baseline_s${SEED}"
        ;;
    smote)
        TAG="prior_${MODEL}_imbalv3_subjectwise_ratio${RATIO}_s${SEED}"
        ;;
    smote_plain)
        TAG="prior_${MODEL}_smote_plain_ratio${RATIO}_s${SEED}"
        ;;
    undersample)
        TAG="prior_${MODEL}_undersample_rus_ratio${RATIO}_s${SEED}"
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
    --mode pooled \
    --subject_wise_split \
    --seed $SEED \
    --time_stratify_labels \
    --tag $TAG"

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

# Evaluation (pooled mode)
echo "[EVAL] Pooled evaluation..."
python scripts/python/evaluation/evaluate.py \
    --model "$MODEL" \
    --tag "$TAG" \
    --mode pooled \
    --jobid "$PBS_JOBID" || echo "[WARN] Pooled eval failed"

echo "[DONE] Array index $PBS_ARRAY_INDEX completed at $(date)"
exit 0

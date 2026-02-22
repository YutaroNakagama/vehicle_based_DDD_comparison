#!/bin/bash
#PBS -N prior_train
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -q SINGLE
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# =============================================================================
# PBS Job Script: Prior Research Replication (SvmA, SvmW, Lstm)
# =============================================================================
# Usage:
#   qsub -v MODEL=SvmA,SEED=42,CONDITION=baseline scripts/hpc/jobs/train/pbs_prior_research.sh
#   qsub -v MODEL=SvmW,SEED=42,CONDITION=smote_plain,RATIO=0.5 scripts/hpc/jobs/train/pbs_prior_research.sh
#   qsub -v MODEL=Lstm,SEED=42,CONDITION=smote,RATIO=0.5 scripts/hpc/jobs/train/pbs_prior_research.sh
#   qsub -v MODEL=SvmW,SEED=42,CONDITION=undersample,RATIO=0.5 scripts/hpc/jobs/train/pbs_prior_research.sh
#
# Environment Variables:
#   MODEL      : SvmA | SvmW | Lstm (required)
#   CONDITION  : baseline | smote_plain | smote (sw) | undersample (required)
#   RATIO      : Target ratio for SMOTE/undersampling (default: 0.5)
#   SEED       : Random seed (default: 42)
#   RUN_EVAL   : Run evaluation after training (default: true)
# =============================================================================

set -euo pipefail

# ===== Environment Setup =====
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

# --- Thread limits for reproducibility ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export JOBLIB_MULTIPROCESSING=0
export JOBLIB_START_METHOD=spawn
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# --- TensorFlow GPU settings (if Lstm) ---
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""  # Force CPU for reproducibility

echo "[ENV] Thread limits applied:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  TF_NUM_INTRAOP_THREADS=$TF_NUM_INTRAOP_THREADS"

PROJECT_ROOT="${PBS_O_WORKDIR:-/home/s2240011/git/ddd/vehicle_based_DDD_comparison}"
# Handle case where PBS_O_WORKDIR is scripts/hpc/jobs/train/
if [[ "$PROJECT_ROOT" == *"scripts/hpc/jobs"* ]]; then
    PROJECT_ROOT="${PROJECT_ROOT%%/scripts/hpc*}"
fi
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

# ===== Parameters =====
MODEL="${MODEL:-SvmA}"
SEED="${SEED:-42}"
CONDITION="${CONDITION:-baseline}"
RATIO="${RATIO:-0.5}"
RUN_EVAL="${RUN_EVAL:-true}"

echo "=============================================="
echo "  Prior Research Replication Experiment"
echo "=============================================="
echo "  MODEL:     $MODEL"
echo "  CONDITION: $CONDITION"
echo "  RATIO:     $RATIO"
echo "  SEED:      $SEED"
echo "  RUN_EVAL:  $RUN_EVAL"
echo "  JOBID:     $PBS_JOBID"
echo "  START:     $(date)"
echo "=============================================="

# ===== Validate Model =====
if [[ "$MODEL" != "SvmA" && "$MODEL" != "SvmW" && "$MODEL" != "Lstm" ]]; then
    echo "[ERROR] Invalid MODEL: $MODEL. Must be SvmA, SvmW, or Lstm."
    exit 1
fi

# ===== Build tag based on condition =====
case "$CONDITION" in
    baseline)
        TAG="prior_${MODEL}_baseline_s${SEED}"
        ;;
    smote_plain)
        TAG="prior_${MODEL}_smote_plain_ratio${RATIO}_s${SEED}"
        ;;
    smote)
        TAG="prior_${MODEL}_imbalv3_subjectwise_ratio${RATIO}_s${SEED}"
        ;;
    undersample)
        TAG="prior_${MODEL}_undersample_rus_ratio${RATIO}_s${SEED}"
        ;;
    *)
        echo "[ERROR] Unknown CONDITION: $CONDITION. Must be baseline, smote_plain, smote, or undersample."
        exit 1
        ;;
esac

# ===== Build training command =====
CMD="python scripts/python/train/train.py \
    --model $MODEL \
    --mode pooled \
    --subject_wise_split \
    --seed $SEED \
    --time_stratify_labels \
    --tag $TAG"

# Add condition-specific flags
case "$CONDITION" in
    baseline)
        # No oversampling
        ;;
    smote_plain)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO"
        ;;
    smote)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO --subject_wise_oversampling"
        ;;
    undersample)
        CMD="$CMD --use_oversampling --oversample_method undersample_rus --target_ratio $RATIO"
        ;;
esac

# ===== Run Training =====
echo "[INFO] Starting training for $MODEL (condition=$CONDITION)..."
echo "[TRAIN] $CMD"

eval $CMD

EXIT_CODE=$?

# ===== Run evaluation if requested and training succeeded =====
if [[ "$RUN_EVAL" == "true" && $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "[EVAL] Running evaluation..."
    EVAL_CMD="python scripts/python/evaluation/evaluate.py \
        --model $MODEL \
        --tag $TAG \
        --mode pooled \
        --jobid $PBS_JOBID"
    echo "[EVAL] $EVAL_CMD"
    eval $EVAL_CMD || echo "[WARNING] Evaluation failed but continuing..."
fi

echo "=============================================="
echo "  JOB COMPLETED"
echo "  MODEL:     $MODEL"
echo "  CONDITION: $CONDITION"
echo "  SEED:      $SEED"
echo "  EXIT:      $EXIT_CODE"
echo "  END:       $(date)"
echo "=============================================="

exit $EXIT_CODE

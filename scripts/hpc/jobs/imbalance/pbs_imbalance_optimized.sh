#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/imbalance/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/imbalance/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
# Note: -N, -l select, -l walltime, -q are passed dynamically via qsub options

# ============================================================
# Imbalance Comparison Training Script (OPTIMIZED HPC Version)
# ============================================================
# Parallelization + Memory optimized version
# - Uses multiple cores for RF training
# - Increased memory for large dataset operations
# - Extended walltime for reliability
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

# ============================================================
# OPTIMIZATION: Parallelization settings
# ============================================================
# Number of cores to use (from PBS or default 4)
NCPUS="${NCPUS:-4}"
export N_JOBS_OVERRIDE="${NCPUS}"

# Thread settings for parallel libraries
export OMP_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}"
export OPENBLAS_NUM_THREADS="${NCPUS}"
export NUMEXPR_NUM_THREADS="${NCPUS}"

# Joblib settings for better parallel performance
export JOBLIB_START_METHOD="fork"

echo "[OPTIMIZATION] Using ${NCPUS} cores for parallel processing"
echo "[OPTIMIZATION] N_JOBS_OVERRIDE=${N_JOBS_OVERRIDE}"

# Parameters
METHOD="${METHOD:-baseline}"
RATIO="${RATIO:-0.5}"
SEED="${SEED:-42}"
RUN_EVAL="${RUN_EVAL:-true}"
export N_TRIALS_OVERRIDE="${N_TRIALS:-100}"  # 論文用: 100 trials

# Auto-select model based on method
case "$METHOD" in
    balanced_rf)
        MODEL="BalancedRF"
        ;;
    *)
        MODEL="RF"
        ;;
esac

# Generate tag
case "$METHOD" in
    baseline)
        TAG="baseline_s${SEED}"
        ;;
    smote)
        TAG="smote_ratio${RATIO}_s${SEED}"
        ;;
    smote_subjectwise)
        TAG="subjectwise_smote_ratio${RATIO}_s${SEED}"
        ;;
    undersample_rus)
        TAG="undersample_rus_ratio${RATIO}_s${SEED}"
        ;;
    balanced_rf)
        TAG="balanced_rf_s${SEED}"
        ;;
    *)
        echo "[ERROR] Unknown method: $METHOD"
        exit 1
        ;;
esac

echo "============================================================"
echo "[IMBALANCE COMPARISON - OPTIMIZED] ${METHOD^^}"
echo "============================================================"
echo "METHOD: $METHOD"
echo "RATIO: $RATIO"
echo "SEED: $SEED"
echo "TAG: $TAG"
echo "N_TRIALS: $N_TRIALS_OVERRIDE"
echo "N_JOBS: $N_JOBS_OVERRIDE"
echo "RUN_EVAL: $RUN_EVAL"
echo "JOBID: $PBS_JOBID"
echo "============================================================"

# Build training command
CMD="python scripts/python/train/train.py \
    --model $MODEL \
    --mode pooled \
    --subject_wise_split \
    --seed $SEED \
    --tag $TAG"

case "$METHOD" in
    baseline)
        # No oversampling
        ;;
    smote)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO"
        ;;
    smote_subjectwise)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO --subject_wise_oversampling"
        ;;
    undersample_rus)
        CMD="$CMD --use_oversampling --oversample_method undersample_rus --target_ratio $RATIO"
        ;;
    balanced_rf)
        # BalancedRF handles class imbalance internally, no additional sampling needed
        ;;
esac

echo ""
echo "[TRAIN] $CMD"
echo ""

# Monitor memory and time
START_TIME=$(date +%s)
eval $CMD
TRAIN_END=$(date +%s)
echo "[TIMING] Training completed in $((TRAIN_END - START_TIME)) seconds"

# Run evaluation if requested
if [[ "$RUN_EVAL" == "true" ]]; then
    echo ""
    echo "============================================================"
    echo "[EVALUATION] Starting..."
    echo "============================================================"
    
    EVAL_CMD="python scripts/python/evaluation/evaluate.py \
        --model $MODEL \
        --mode pooled \
        --seed $SEED \
        --tag $TAG \
        --subject_wise_split"
    
    echo "[EVAL] $EVAL_CMD"
    echo ""
    eval $EVAL_CMD
    EVAL_END=$(date +%s)
    echo "[TIMING] Evaluation completed in $((EVAL_END - TRAIN_END)) seconds"
fi

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
echo ""
echo "============================================================"
echo "[INFO] Completed at $(date)"
echo "[INFO] Total execution time: ${TOTAL_TIME} seconds ($(echo "scale=2; ${TOTAL_TIME}/3600" | bc) hours)"
echo "============================================================"

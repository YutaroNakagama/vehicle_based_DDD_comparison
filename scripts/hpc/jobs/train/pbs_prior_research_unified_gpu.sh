#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
# Note: -N, -l select, -l walltime, -q are passed dynamically via qsub options

# ============================================================
# Prior research experiment (unified version - GPU) - LSTM only
# ============================================================
# Differences from CPU version:
#   - Uses GPU (A40/A100) to accelerate LSTM training and evaluation
#   - Does not set CUDA_VISIBLE_DEVICES (TF auto-detects)
#   - TensorFlow GPU memory is managed via configure_gpu()
#   - Loads CUDA libraries via hpc_sdk module
#
# Environment Variables:
#   MODEL      : Lstm (required — GPU version is LSTM only)
#   CONDITION  : baseline | smote | smote_plain | undersample (required)
#   DISTANCE   : mmd | wasserstein | dtw (required)
#   DOMAIN     : in_domain | out_domain (required)
#   RATIO      : Target ratio for SMOTE (default: 0.5)
#   SEED       : Random seed (default: 42)
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

# CUDA 12.8 + cuDNN 9 (hpc_sdk/22.2 was replaced by nvhpc/26.3 which lacks cuDNN;
# kagayaki CUDA 12.8u1 provides libcudart.so.12 + libcudnn.so.9 needed by TF 2.19)
CUDA12_TARGET="/app/kagayaki/CUDA/12.8u1/targets/x86_64-linux/lib"
CUDA12_LIB="/app/kagayaki/CUDA/12.8u1/lib64"
export LD_LIBRARY_PATH="${CUDA12_TARGET}:${CUDA12_LIB}:${LD_LIBRARY_PATH:-}"

# Thread optimization for HPC (keep CPU threads low, let GPU do the work)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=1

# GPU settings: DO NOT set CUDA_VISIBLE_DEVICES="" — let TF detect GPU
# TF_FORCE_GPU_ALLOW_GROWTH is a fallback for configure_gpu()
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Parameters
MODEL="${MODEL:-Lstm}"
CONDITION="${CONDITION:-baseline}"
DISTANCE="${DISTANCE:-mmd}"
DOMAIN="${DOMAIN:-in_domain}"
RATIO="${RATIO:-0.5}"
SEED="${SEED:-42}"
RANKING="${RANKING:-knn}"
RUN_EVAL="${RUN_EVAL:-true}"

# Validate MODEL (GPU script is Lstm only)
if [[ "$MODEL" != "Lstm" ]]; then
    echo "[ERROR] GPU script is for Lstm only, got: $MODEL"
    exit 1
fi

# Target file path
TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${DOMAIN}.txt"

# Cross-domain target file (opposite domain)
if [[ "$DOMAIN" == "in_domain" ]]; then
    CROSS_DOMAIN="out_domain"
else
    CROSS_DOMAIN="in_domain"
fi
CROSS_TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${CROSS_DOMAIN}.txt"

# Generate tag (same as CPU version for consistency)
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
echo "[UNIFIED-GPU] ${MODEL} - ${CONDITION^^} - domain_train"
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
echo "RUN_EVAL: $RUN_EVAL"
echo "JOBID: $PBS_JOBID"
echo "SPLIT: 70/15/15 (train/val/test)"

# Report GPU info
echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available on this node)"
GPU_COUNT=$(python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(len(gpus))" 2>/dev/null || echo "0")
echo "TF GPU count: $GPU_COUNT"
if [[ "$GPU_COUNT" -lt 1 ]]; then
    echo "[ERROR] No GPU detected by TensorFlow — aborting to avoid silent CPU fallback"
    echo "[DEBUG] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    exit 1
fi
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

# Build training command
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
echo "[DONE] Unified GPU job completed (exit code: $EXIT_CODE)"
echo "  Training domain: $DOMAIN"
echo "  Within-domain eval: $DOMAIN test (15%)"
echo "  Cross-domain eval: $CROSS_DOMAIN test (15%)"
echo "============================================================"

exit $EXIT_CODE

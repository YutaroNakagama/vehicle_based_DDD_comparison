#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
#PBS -N Ls_retrain
#PBS -q GPU-1
#PBS -l select=1:ncpus=8:mem=8gb:ngpus=1
#PBS -l walltime=20:00:00
#PBS -J 0-4

# ============================================================
# Lstm GPU 再学習: CUDAなしで失敗した5ジョブ
# ============================================================
# 元のジョブ:
#   [0] 14873005 - smote_plain / dtw / out_domain / s42
#   [1] 14873006 - undersample / dtw / out_domain / s42
#   [2] 14873021 - smote_plain / dtw / in_domain  / s123
#   [3] 14873023 - undersample / dtw / in_domain  / s123
#   [4] 14873035 - smote_plain / wasserstein / out_domain / s123
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

# Load CUDA via hpc_sdk module
module load hpc_sdk/22.2 2>/dev/null || true

# Thread optimization for HPC
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

MODEL="Lstm"
RANKING="knn"
RUN_EVAL="true"

# Job array configuration
IDX=${PBS_ARRAY_INDEX}

CONDITIONS=(smote_plain undersample smote_plain undersample smote_plain)
DISTANCES=(dtw dtw dtw dtw wasserstein)
DOMAINS=(out_domain out_domain in_domain in_domain out_domain)
SEEDS=(42 42 123 123 123)
OLD_JOBIDS=(14873005 14873006 14873021 14873023 14873035)

CONDITION="${CONDITIONS[$IDX]}"
DISTANCE="${DISTANCES[$IDX]}"
DOMAIN="${DOMAINS[$IDX]}"
SEED="${SEEDS[$IDX]}"
OLD_JOBID="${OLD_JOBIDS[$IDX]}"

# Target file path
TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${DOMAIN}.txt"

# Cross-domain target file (opposite domain)
if [[ "$DOMAIN" == "in_domain" ]]; then
    CROSS_DOMAIN="out_domain"
else
    CROSS_DOMAIN="in_domain"
fi
CROSS_TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${CROSS_DOMAIN}.txt"

# Generate tag
case "$CONDITION" in
    smote_plain)
        TAG="prior_${MODEL}_smote_plain_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_ratio0.1_s${SEED}"
        ;;
    undersample)
        TAG="prior_${MODEL}_undersample_rus_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_ratio0.1_s${SEED}"
        ;;
esac

echo "============================================================"
echo "[RETRAIN-GPU] ${MODEL} - ${CONDITION^^} - domain_train"
echo "============================================================"
echo "ARRAY_INDEX: $IDX"
echo "OLD_JOBID: $OLD_JOBID (failed — no CUDA)"
echo "MODEL: $MODEL"
echo "CONDITION: $CONDITION"
echo "DISTANCE: $DISTANCE"
echo "DOMAIN: $DOMAIN"
echo "CROSS_DOMAIN: $CROSS_DOMAIN"
echo "SEED: $SEED"
echo "TAG: $TAG"
echo "TARGET_FILE: $TARGET_FILE"
echo "CROSS_TARGET_FILE: $CROSS_TARGET_FILE"
echo "JOBID: $PBS_JOBID"
echo "SPLIT: 70/15/15 (train/val/test)"

# Invalidate old model directory
if [[ -d "models/Lstm/${OLD_JOBID}" ]]; then
    mv "models/Lstm/${OLD_JOBID}" "models/Lstm/_invalidated_nocuda_${OLD_JOBID}"
    echo "[CLEANUP] Moved models/Lstm/${OLD_JOBID} -> _invalidated_nocuda_${OLD_JOBID}"
fi

# Report GPU info
echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available on this node)"
python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(f'TF GPUs: {gpus}')" 2>/dev/null || echo "(TF GPU check failed)"
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

case "$CONDITION" in
    smote_plain)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio 0.1"
        ;;
    undersample)
        CMD="$CMD --use_oversampling --oversample_method undersample_rus --target_ratio 0.1"
        ;;
esac

echo ""
echo "[TRAIN] $CMD"
echo ""
eval $CMD

EXIT_CODE=$?

# Run evaluations if training succeeded
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
else
    echo "[SKIP] Evaluation skipped (RUN_EVAL=$RUN_EVAL, EXIT_CODE=$EXIT_CODE)"
fi

echo ""
echo "============================================================"
echo "[DONE] ${MODEL} retrain completed (array[$IDX], old=$OLD_JOBID)"
echo "============================================================"

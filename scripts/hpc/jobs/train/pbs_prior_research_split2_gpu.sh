#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
# Note: -N, -l select, -l walltime, -q are passed dynamically via qsub options

# ============================================================
# Prior research experiment - 2-group split version (GPU, Lstm only)
# ============================================================
# GPU variant of pbs_prior_research_split2.sh; supports the legacy
# modes (source_only / target_only / mixed) for Lstm so that GPU
# acceleration is available.
#
# Environment Variables:
#   MODEL      : Lstm (required — GPU script is Lstm only)
#   CONDITION  : baseline | smote | smote_plain | undersample (required)
#   MODE       : source_only | target_only | mixed (required)
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
# PBS_JOBID must be a pure numeric ID — savers.py's regex `\d{5,}\[\d+\]`
# treats the LAST numeric run before [N] as the jobid, so a "manual_yyyymmdd_hhmmss"
# prefix gets stripped at save time but kept at eval time, breaking lookup.
# Prefer SLURM's job id when available; fall back to epoch+PID for manual runs.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export PBS_JOBID="${SLURM_JOB_ID}"
elif [[ -n "${SLURM_JOBID:-}" ]]; then
    export PBS_JOBID="${SLURM_JOBID}"
elif [[ -z "${PBS_JOBID:-}" ]]; then
    export PBS_JOBID="$(date +%s)$$"
fi

# CUDA 12.8 + cuDNN 9 — use cluster's cuda/12.8u1 module (bundles cuDNN).
# /app/kagayaki/CUDA/12.8u1 is only visible from login, not compute nodes.
# `module` references unset vars under `set -u`; relax briefly.
set +u
source /etc/profile.d/modules.sh
module load cuda/12.8u1
MODULE_RC=$?
set -u
if [[ "$MODULE_RC" -ne 0 ]]; then
    echo "[WARNING] cuda/12.8u1 module load failed (rc=$MODULE_RC); falling back"
    export LD_LIBRARY_PATH="/app/kagayaki/CUDA/12.8u1/targets/x86_64-linux/lib:/app/kagayaki/CUDA/12.8u1/lib64:${LD_LIBRARY_PATH:-}"
fi

# Thread optimization for HPC (keep CPU threads low, let GPU do the work)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=1

# GPU settings: DO NOT set CUDA_VISIBLE_DEVICES="" — let TF detect GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Parameters
MODEL="${MODEL:-Lstm}"
CONDITION="${CONDITION:-baseline}"
MODE="${MODE:-source_only}"
DISTANCE="${DISTANCE:-mmd}"
DOMAIN="${DOMAIN:-in_domain}"
RATIO="${RATIO:-0.5}"
SEED="${SEED:-42}"
RANKING="${RANKING:-knn}"
RUN_EVAL="${RUN_EVAL:-true}"
export N_TRIALS_OVERRIDE="${N_TRIALS:-100}"

# Validate MODEL (GPU script is Lstm only)
if [[ "$MODEL" != "Lstm" ]]; then
    echo "[ERROR] GPU split2 script is Lstm only, got: $MODEL"
    exit 1
fi

# Hard GPU check — fail loudly instead of silently falling back to CPU
GPU_COUNT=$(python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(len(gpus))" 2>/dev/null || echo "0")
echo "TF GPU count: $GPU_COUNT"
if [[ "$GPU_COUNT" -lt 1 ]]; then
    echo "[ERROR] No GPU detected by TensorFlow — aborting"
    echo "[DEBUG] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    exit 1
fi

# Target file path (split2 directory)
TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${DOMAIN}.txt"

# Generate tag based on condition and model
case "$CONDITION" in
    baseline)
        TAG="prior_${MODEL}_baseline_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_s${SEED}"
        ;;
    smote)
        TAG="prior_${MODEL}_imbalv3_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_subjectwise_ratio${RATIO}_s${SEED}"
        ;;
    smote_plain)
        TAG="prior_${MODEL}_smote_plain_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_ratio${RATIO}_s${SEED}"
        ;;
    undersample)
        TAG="prior_${MODEL}_undersample_rus_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_ratio${RATIO}_s${SEED}"
        ;;
    balanced_rf)
        TAG="prior_${MODEL}_balanced_rf_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_s${SEED}"
        ;;
    *)
        echo "[ERROR] Unknown condition: $CONDITION"
        exit 1
        ;;
esac

echo "============================================================"
echo "[PRIOR RESEARCH - SPLIT2] ${MODEL} - ${CONDITION^^}"
echo "============================================================"
echo "MODEL: $MODEL"
echo "CONDITION: $CONDITION"
echo "MODE: $MODE"
echo "DISTANCE: $DISTANCE"
echo "DOMAIN: $DOMAIN"
echo "RATIO: $RATIO"
echo "SEED: $SEED"
echo "RANKING: $RANKING"
echo "TAG: $TAG"
echo "TARGET_FILE: $TARGET_FILE"
echo "N_TRIALS: $N_TRIALS_OVERRIDE (for SvmW)"
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
    balanced_rf)
        echo "[ERROR] balanced_rf not supported for $MODEL"
        exit 1
        ;;
esac

echo ""
echo "[TRAIN] $CMD"
echo ""
eval $CMD

EXIT_CODE=$?

# Run evaluation if requested and training succeeded
if [[ "$RUN_EVAL" == "true" && $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "[EVAL] Running evaluation..."
    echo ""
    
    EVAL_CMD="python scripts/python/evaluation/evaluate.py \
        --model $MODEL \
        --tag $TAG \
        --mode $MODE \
        --jobid $PBS_JOBID"
    
    echo "[EVAL] $EVAL_CMD"
    eval $EVAL_CMD || echo "[WARNING] Evaluation failed but continuing..."
fi

echo ""
echo "[DONE] Job completed (exit code: $EXIT_CODE)"
echo "============================================================"

exit $EXIT_CODE

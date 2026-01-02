#!/bin/bash
#PBS -N smote_brf_rank
#PBS -l select=1:ncpus=8:mem=8gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# SMOTE + BalancedRF + Ranking Training Script
# For source_only/target_only mode with ranking-based subject selection
# ============================================================
set -euo pipefail

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

# Memory optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export JOBLIB_MULTIPROCESSING=0
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export N_TRIALS_OVERRIDE=50

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2240011/git/ddd/vehicle_based_DDD_comparison}"
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

# Parameters from environment
TAG="${TAG:-smote_brf_rank_test}"
RATIO="${RATIO:-0.33}"
MODE="${MODE:-source_only}"
SUBJECT_FILE="${SUBJECT_FILE:-}"
SEED="${SEED:-42}"

echo "============================================================"
echo "[SMOTE + BalancedRF + Ranking] Training"
echo "============================================================"
echo "TAG: $TAG"
echo "RATIO: $RATIO"
echo "MODE: $MODE"
echo "SUBJECT_FILE: $SUBJECT_FILE"
echo "SEED: $SEED"
echo "N_TRIALS: 50"
echo "JOBID: $PBS_JOBID"
echo "============================================================"

# Build command
CMD="python scripts/python/train/train.py \
    --model BalancedRF \
    --mode $MODE \
    --seed $SEED \
    --time_stratify_labels \
    --use_oversampling \
    --oversample_method smote \
    --target_ratio $RATIO \
    --tag $TAG"

# Add target_file if mode is not pooled
if [[ "$MODE" != "pooled" && -n "$SUBJECT_FILE" ]]; then
    if [[ -f "$SUBJECT_FILE" ]]; then
        CMD="$CMD --target_file $SUBJECT_FILE"
        echo "[INFO] Using subject file: $SUBJECT_FILE"
        echo "[INFO] Subject count: $(wc -l < "$SUBJECT_FILE")"
    else
        echo "[WARN] Subject file not found: $SUBJECT_FILE"
        echo "[INFO] Falling back to default subjects"
    fi
fi

echo ""
echo "[CMD] $CMD"
echo ""

eval $CMD

echo "=== SMOTE + BalancedRF TRAINING DONE ==="

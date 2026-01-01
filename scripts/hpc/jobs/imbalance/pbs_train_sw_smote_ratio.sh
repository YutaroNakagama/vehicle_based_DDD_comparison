#!/bin/bash
#PBS -j oe
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Subject-wise SMOTE with Configurable Ratio
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

# Parameters
SEED="${SEED:-42}"
RATIO="${RATIO:-0.33}"
TAG="${TAG:-imbal_v2_smote_sw_ratio_test}"

echo "============================================================"
echo "[Subject-wise SMOTE] Ratio Comparison"
echo "============================================================"
echo "SEED: $SEED"
echo "RATIO: $RATIO"
echo "TAG: $TAG"
echo "MODE: pooled"
echo "N_TRIALS: 50"
echo "JOBID: $PBS_JOBID"
echo "============================================================"

python scripts/python/train/train.py \
    --model RF \
    --mode pooled \
    --subject_wise_split \
    --seed "$SEED" \
    --time_stratify_labels \
    --use_oversampling \
    --oversample_method smote \
    --target_ratio "$RATIO" \
    --subject_wise_oversampling \
    --tag "$TAG"

echo "=== TRAINING DONE ==="

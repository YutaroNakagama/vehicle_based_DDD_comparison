#!/bin/bash
#PBS -N imbal_v2_smote_brf_r
#PBS -l select=1:ncpus=8:mem=8gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Imbalance Comparison V2: SMOTE + BalancedRandomForest with Ratio
# ============================================================
set -euo pipefail

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

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

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

SEED="${SEED:-42}"
RATIO="${RATIO:-0.5}"
RATIO_TAG="${RATIO_TAG:-0_5}"
TAG="imbal_v2_smote_balanced_rf_ratio${RATIO_TAG}_seed${SEED}"

echo "============================================================"
echo "[IMBALANCE COMPARISON V2] SMOTE + BalancedRandomForest"
echo "============================================================"
echo "MODEL: BalancedRF"
echo "MODE: pooled"
echo "OVERSAMPLING: SMOTE"
echo "SAMPLING_RATIO: $RATIO"
echo "SEED: $SEED"
echo "TAG: $TAG"
echo "JOBID: $PBS_JOBID"
echo "============================================================"

python scripts/python/train/train.py \
    --model BalancedRF \
    --mode pooled \
    --subject_wise_split \
    --seed "$SEED" \
    --time_stratify_labels \
    --use_oversampling \
    --oversample_method smote \
    --target_ratio "$RATIO" \
    --tag "$TAG"

echo "=== SMOTE + BALANCED RF TRAINING DONE ==="

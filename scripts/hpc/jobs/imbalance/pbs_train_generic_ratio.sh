#!/bin/bash
#PBS -N imbal_v2_ratio
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
# Note: -l select, -l walltime, -q are passed dynamically via qsub options

# ============================================================
# Imbalance Comparison V2: Generic Training with Target Ratio
# Supports all methods with configurable target_ratio
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

# Parameters passed via -v option
MODEL="${MODEL:-RF}"
RATIO="${RATIO:-0.33}"
METHOD="${METHOD:-smote}"
TAG="${TAG:-imbal_v2_ratio_test}"
SEED="${SEED:-42}"

echo "============================================================"
echo "[IMBALANCE COMPARISON V2] Generic Ratio Training"
echo "============================================================"
echo "MODEL: $MODEL"
echo "METHOD: $METHOD"
echo "TARGET_RATIO: $RATIO"
echo "TAG: $TAG"
echo "MODE: pooled"
echo "N_TRIALS: 50"
echo "OBJECTIVE: F2 Score"
echo "SPLIT: Time-stratified"
echo "JOBID: $PBS_JOBID"
echo "============================================================"

python scripts/python/train/train.py \
    --model "$MODEL" \
    --mode pooled \
    --subject_wise_split \
    --seed "$SEED" \
    --time_stratify_labels \
    $(if [[ "$METHOD" != "baseline" && "$METHOD" != "balanced_rf" && "$METHOD" != "easy_ensemble" ]]; then
        # Use oversampling for actual oversampling methods
        if [[ "$METHOD" == "smote_balanced_rf" ]]; then
            echo "--use_oversampling --oversample_method smote --target_ratio $RATIO"
        else
            echo "--use_oversampling --oversample_method $METHOD --target_ratio $RATIO"
        fi
    fi) \
    --tag "$TAG"

echo "=== RATIO TRAINING DONE ==="

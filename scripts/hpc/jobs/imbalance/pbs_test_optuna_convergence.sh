#!/bin/bash
#PBS -N test_optuna_conv
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -q SINGLE
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Test Script: Optuna Convergence Logging Verification
# ============================================================
# N_TRIALS=10 でOptunaの収束ログが正しく出力されるか確認
# 所要時間: 約15-30分
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

# === TEST: N_TRIALS=10 for quick verification ===
export N_TRIALS_OVERRIDE=10

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "[TEST] Optuna Convergence Logging Verification"
echo "============================================================"
echo "N_TRIALS: 10 (reduced for quick test)"
echo "METHOD: SMOTE"
echo "JOBID: $PBS_JOBID"
echo "============================================================"

python scripts/python/train/train.py \
    --model RF \
    --mode pooled \
    --subject_wise_split \
    --seed 42 \
    --time_stratify_labels \
    --use_oversampling \
    --oversample_method smote \
    --tag test_optuna_convergence

echo "============================================================"
echo "[DONE] Test completed at $(date)"
echo "============================================================"

#!/bin/bash
#PBS -N imbal_v2_smote
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=10:00:00
#PBS -j oe
#PBS -q SINGLE
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Imbalance Comparison V2: SMOTE (単体)
# ============================================================
# SMOTE単体の効果を測定するためのベースライン実験
# 後処理（Tomek/ENN/RUS）の追加効果を分離して評価するため
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

MODEL="${MODEL:-RF}"
SEED="${SEED:-42}"
TAG="imbal_v2_smote_seed${SEED}"

echo "============================================================"
echo "[IMBALANCE COMPARISON V2] SMOTE (単体)"
echo "============================================================"
echo "MODEL: $MODEL"
echo "SEED: $SEED"
echo "TAG: $TAG"
echo "MODE: pooled"
echo "OVERSAMPLING: SMOTE only (no post-processing)"
echo "N_TRIALS: 50"
echo "CV_FOLDS: 5"
echo "OBJECTIVE: F2 Score"
echo "SPLIT: Time-stratified"
echo "JOBID: $PBS_JOBID"
echo "============================================================"

python scripts/python/train.py \
    --model "$MODEL" \
    --mode pooled \
    --subject_wise_split \
    --seed "$SEED" \
    --time_stratify_labels \
    --use_oversampling \
    --oversample_method smote \
    --tag "$TAG"

echo "=== SMOTE TRAINING DONE ==="

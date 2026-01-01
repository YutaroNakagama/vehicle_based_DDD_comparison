#!/bin/bash
#PBS -N imbal_smote_sw
#PBS -l select=1:ncpus=4:mem=6gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -q SINGLE
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Imbalance Comparison: Subject-wise SMOTE
# ============================================================
# SMOTEを被験者ごとに個別適用し、被験者間のノイズ混入を防止
# - 各被験者内でのみ合成サンプルを生成
# - 異なる被験者の特性が混ざらない
# ============================================================
set -euo pipefail

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

# メモリ最適化: スレッド数を制限
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
TAG="imbal_v2_smote_subjectwise_seed${SEED}"

echo "============================================================"
echo "[IMBALANCE COMPARISON] Subject-wise SMOTE"
echo "============================================================"
echo "MODEL: $MODEL"
echo "SEED: $SEED"
echo "TAG: $TAG"
echo "MODE: pooled"
echo "OVERSAMPLING: Subject-wise SMOTE"
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
    --subject_wise_oversampling \
    --tag "$TAG"

echo "=== SUBJECT-WISE SMOTE TRAINING DONE ==="

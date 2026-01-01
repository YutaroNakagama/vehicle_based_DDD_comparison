#!/bin/bash
#PBS -N imbal_v2_jitter_scale
#PBS -l select=1:ncpus=8:mem=8gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Imbalance Comparison V2: Jittering + Scaling Augmentation
# Time-series inspired data augmentation for imbalanced data
# Reference: Um et al. (2017) ICMI - Data Augmentation of Wearable Sensor Data
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

echo "============================================================"
echo "[IMBALANCE COMPARISON V2] Jittering + Scaling Augmentation"
echo "============================================================"
echo "MODEL: $MODEL"
echo "MODE: pooled"
echo "AUGMENTATION: Jittering (σ=0.03) + Scaling (σ=0.1)"
echo "TARGET_RATIO: 0.33 (minority:majority = 1:3)"
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
    --use_oversampling \
    --oversample_method jitter_scale \
    --tag "imbal_v2_jitter_scale"

echo "=== JITTERING + SCALING TRAINING DONE ==="

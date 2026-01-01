#!/bin/bash
#PBS -N imbal_v2_thr
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Imbalance Comparison V2: Threshold Optimization Evaluation
# Evaluate each model with optimized threshold for Recall
# ============================================================
set -euo pipefail

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JOBLIB_MULTIPROCESSING=0

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

# These should be passed via environment or -v option
MODEL="${MODEL:-RF}"
TAG="${TAG:-imbal_v2_baseline}"
TRAIN_JOBID="${TRAIN_JOBID:-}"
THRESHOLD="${THRESHOLD:-0.15}"
SEED="${SEED:-42}"

echo "============================================================"
echo "[IMBALANCE V2] Threshold Optimization Evaluation"
echo "============================================================"
echo "MODEL: $MODEL"
echo "TAG: $TAG"
echo "TRAIN_JOBID: $TRAIN_JOBID"
echo "THRESHOLD: $THRESHOLD"
echo "EVAL_JOBID: $PBS_JOBID"
echo "============================================================"

python scripts/python/evaluate.py \
    --model "$MODEL" \
    --mode pooled \
    --seed "$SEED" \
    --tag "$TAG" \
    --jobid "$TRAIN_JOBID" \
    --threshold "$THRESHOLD"

echo "=== THRESHOLD EVALUATION DONE ==="

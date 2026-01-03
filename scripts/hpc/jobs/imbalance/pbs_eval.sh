#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/evaluate/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/evaluate/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
# Note: -N, -l select, -l walltime, -q are passed dynamically via qsub options

# ============================================================
# Unified Evaluation Script
# Supports standard evaluation and threshold optimization
# ============================================================
# Usage:
#   qsub -N eval_smote -l select=1:ncpus=2:mem=4gb -l walltime=02:00:00 -q SINGLE \
#        -v MODEL=RF,TAG=imbal_v2_smote,TRAIN_JOBID=12345678 pbs_eval.sh
#
# Environment Variables:
#   MODEL       : Model type (RF, BalancedRF, EasyEnsemble)
#   TAG         : Experiment tag (required)
#   TRAIN_JOBID : Training job ID (required)
#   SEED        : Random seed (default: 42)
#   MODE        : Evaluation mode (default: pooled)
#   THRESHOLD   : Custom threshold for prediction (default: 0.5)
# ============================================================
set -euo pipefail

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

# Thread optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JOBLIB_MULTIPROCESSING=0

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

# === Parameters ===
MODEL="${MODEL:-RF}"
TAG="${TAG:-}"
TRAIN_JOBID="${TRAIN_JOBID:-}"
SEED="${SEED:-42}"
MODE="${MODE:-pooled}"
THRESHOLD="${THRESHOLD:-}"

# Validate required parameters
if [[ -z "$TAG" ]]; then
    echo "[ERROR] TAG is required"
    exit 1
fi

if [[ -z "$TRAIN_JOBID" ]]; then
    echo "[ERROR] TRAIN_JOBID is required"
    exit 1
fi

echo "============================================================"
echo "[EVALUATION] ${TAG}"
echo "============================================================"
echo "MODEL: $MODEL"
echo "TAG: $TAG"
echo "TRAIN_JOBID: $TRAIN_JOBID"
echo "MODE: $MODE"
echo "SEED: $SEED"
echo "THRESHOLD: ${THRESHOLD:-default}"
echo "EVAL_JOBID: $PBS_JOBID"
echo "============================================================"

# === Build Command ===
CMD="python scripts/python/evaluation/evaluate.py \
    --model $MODEL \
    --mode $MODE \
    --seed $SEED \
    --tag $TAG \
    --jobid $TRAIN_JOBID"

# Add threshold if specified
if [[ -n "$THRESHOLD" ]]; then
    CMD="$CMD --threshold $THRESHOLD"
    echo "[INFO] Using custom threshold: $THRESHOLD"
fi

echo ""
echo "[CMD] $CMD"
echo ""

eval $CMD

echo "=== EVALUATION DONE ==="

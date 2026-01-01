#!/bin/bash
#PBS -N RF_train_test
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

set -euo pipefail

# ===== env =====
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

# --- Limit thread usage (OpenMP, MKL, BLAS, TensorFlow) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

# --- Additional safeguard for scikit-learn / joblib ---
export JOBLIB_MULTIPROCESSING=0
export JOBLIB_START_METHOD=spawn
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

echo "[ENV] Thread limits applied:"
echo " OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo " MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo " OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo " JOBLIB_MULTIPROCESSING=$JOBLIB_MULTIPROCESSING"

PROJECT_ROOT="${PBS_O_WORKDIR}"
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}

# Always ensure PBS_JOBID is available (even in local test)
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

MODEL="${MODEL:-RF}"
SEED="${SEED:-42}"

# ===== TEST MODE: Reduced trials and simplified KSS labels =====
export TEST_MODE=1
export N_TRIALS_OVERRIDE=5
export KSS_SIMPLIFIED=1
export TRAIN_RATIO=0.4
export VAL_RATIO=0.3
export TEST_RATIO=0.3

echo "=== TEST JOB START (MODEL=$MODEL, SEED=$SEED, JOBID=$PBS_JOBID) ==="
echo "[TEST] N_TRIALS reduced to 5"
echo "[TEST] KSS labels: 1-3=0, 8-9=1"
echo "[TEST] Data split: train=0.4, val=0.3, test=0.3"

python scripts/python/train/train.py \
    --model "$MODEL" \
    --mode pooled \
    --subject_wise_split \
    --seed "$SEED" \
    --time_stratify_labels

echo "=== TEST JOB DONE (MODEL=$MODEL) ==="

LOG_BASE="$PBS_O_WORKDIR/scripts/hpc/logs/train_test_${MODEL}_${PBS_JOBID}.log"
cat "$PBS_O_WORKDIR/scripts/hpc/logs/train_test.o${PBS_JOBID}" \
    > "$LOG_BASE" 2>/dev/null || true

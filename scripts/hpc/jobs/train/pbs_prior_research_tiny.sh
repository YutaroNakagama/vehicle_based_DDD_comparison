#!/bin/bash
#PBS -N prior_tiny
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q TINY
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# =============================================================================
# PBS Job Script: Prior Research Replication (TINY queue - 30min limit)
# =============================================================================
# Usage:
#   qsub -v MODEL=SvmA,SEED=42 scripts/hpc/jobs/train/pbs_prior_research_tiny.sh
#   qsub -v MODEL=Lstm,SEED=42 scripts/hpc/jobs/train/pbs_prior_research_tiny.sh
# =============================================================================

set -euo pipefail

# ===== Environment Setup =====
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

# --- Thread limits for reproducibility ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export JOBLIB_MULTIPROCESSING=0
export JOBLIB_START_METHOD=spawn
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# --- TensorFlow GPU settings (if Lstm) ---
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""  # Force CPU for reproducibility

PROJECT_ROOT="${PBS_O_WORKDIR:-/home/s2240011/git/ddd/vehicle_based_DDD_comparison}"
if [[ "$PROJECT_ROOT" == *"scripts/hpc/jobs"* ]]; then
    PROJECT_ROOT="${PROJECT_ROOT%%/scripts/hpc*}"
fi
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

# ===== Parameters =====
MODEL="${MODEL:-SvmA}"
SEED="${SEED:-42}"

echo "=============================================="
echo "  Prior Research (TINY Queue - 30min)"
echo "=============================================="
echo "  MODEL:   $MODEL"
echo "  SEED:    $SEED"
echo "  JOBID:   $PBS_JOBID"
echo "  START:   $(date)"
echo "=============================================="

# ===== Validate Model =====
if [[ "$MODEL" != "SvmA" && "$MODEL" != "SvmW" && "$MODEL" != "Lstm" ]]; then
    echo "[ERROR] Invalid MODEL: $MODEL. Must be SvmA, SvmW, or Lstm."
    exit 1
fi

# ===== Run Training =====
echo "[INFO] Starting training for $MODEL..."

python scripts/python/train/train.py \
    --model "$MODEL" \
    --mode pooled \
    --subject_wise_split \
    --seed "$SEED" \
    --time_stratify_labels \
    --tag "prior_research_s${SEED}"

echo "=============================================="
echo "  COMPLETED: $MODEL"
echo "  END:   $(date)"
echo "=============================================="

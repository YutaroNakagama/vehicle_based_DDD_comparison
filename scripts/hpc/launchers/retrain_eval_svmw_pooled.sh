#!/bin/bash
#PBS -N retrain_svmw_pooled
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -q SINGLE
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# =============================================================================
# PBS Job Script: SvmW Pooled Re-train + Re-evaluate
# =============================================================================
# Purpose: Re-run SvmW pooled experiments (s42, s123) because original results
#          had f1_pos=None due to metrics.py bug (key "1" vs "1.0" mismatch),
#          fixed in commit 6c5f14c (2026-02-09).
#
# Original runs: job 14662837 (s42, Jan 10), job 14662838 (s123, Feb 8)
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

export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""

PROJECT_ROOT="${PBS_O_WORKDIR:-/home/s2240011/git/ddd/vehicle_based_DDD_comparison}"
if [[ "$PROJECT_ROOT" == *"scripts/hpc/jobs"* ]]; then
    PROJECT_ROOT="${PROJECT_ROOT%%/scripts/hpc*}"
fi
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

JOBID_BASE=$(echo "$PBS_JOBID" | sed 's/\..*//')

echo "=============================================="
echo "  SvmW Pooled Re-train + Re-evaluate"
echo "=============================================="
echo "  JOBID:   $PBS_JOBID"
echo "  START:   $(date)"
echo "=============================================="

# ===== Seed 42: Train =====
echo ""
echo "===== [1/4] Training SvmW pooled s42 ====="
python scripts/python/train/train.py \
    --model SvmW \
    --mode pooled \
    --subject_wise_split \
    --seed 42 \
    --time_stratify_labels \
    --tag "prior_research_s42"
echo "[1/4] Training SvmW pooled s42 DONE ($(date))"

# ===== Seed 42: Evaluate =====
echo ""
echo "===== [2/4] Evaluating SvmW pooled s42 ====="
python scripts/python/evaluation/evaluate.py \
    --model SvmW \
    --tag prior_research_s42 \
    --mode pooled \
    --seed 42 \
    --jobid "$JOBID_BASE"
echo "[2/4] Evaluating SvmW pooled s42 DONE ($(date))"

# ===== Seed 123: Train =====
echo ""
echo "===== [3/4] Training SvmW pooled s123 ====="
python scripts/python/train/train.py \
    --model SvmW \
    --mode pooled \
    --subject_wise_split \
    --seed 123 \
    --time_stratify_labels \
    --tag "prior_research_s123"
echo "[3/4] Training SvmW pooled s123 DONE ($(date))"

# ===== Seed 123: Evaluate =====
echo ""
echo "===== [4/4] Evaluating SvmW pooled s123 ====="
python scripts/python/evaluation/evaluate.py \
    --model SvmW \
    --tag prior_research_s123 \
    --mode pooled \
    --seed 123 \
    --jobid "$JOBID_BASE"
echo "[4/4] Evaluating SvmW pooled s123 DONE ($(date))"

echo ""
echo "=============================================="
echo "  ALL DONE"
echo "  JOBID:   $PBS_JOBID"
echo "  END:     $(date)"
echo "=============================================="

#!/bin/bash
#PBS -N subj_scores
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Compute Per-Subject Scores for Statistical Testing
# ============================================================
# This script:
# 1. Computes per-subject evaluation metrics from saved predictions
# 2. Runs statistical tests (Wilcoxon, Cohen's d)
# 3. Generates summary report for paper
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

echo "============================================================"
echo "[STATISTICAL ANALYSIS] Aggregate Metrics Comparison"
echo "============================================================"
echo "PBS_JOBID: $PBS_JOBID"
echo "Date: $(date)"
echo "============================================================"

# Run statistical tests on aggregate metrics
# Uses paired comparisons across experimental configurations
echo ""
echo "[INFO] Running statistical tests on aggregate metrics..."
python scripts/python/analysis/imbalance/statistical_tests_aggregate.py

echo ""
echo "============================================================"
echo "[DONE] Statistical analysis completed"
echo "============================================================"
echo "Output files:"
echo "  - results/imbalance_analysis/domain/statistical_tests.csv"
echo "  - results/imbalance_analysis/domain/statistical_tests_summary.txt"
echo "============================================================"

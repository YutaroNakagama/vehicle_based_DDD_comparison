#!/bin/bash
#PBS -N viz_split2
#PBS -q TINY
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/evaluate/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/evaluate/
# ==============================================================================
# PBS Script: Collect split2 metrics & generate summary_metrics_bar plots
# ==============================================================================
# Scans BalancedRF evaluation JSONs with split2 tag, produces:
#   - CSV: results/analysis/exp2_domain_shift/figures/csv/split2/{condition}/*.csv
#   - PNG: results/analysis/exp2_domain_shift/figures/png/split2/{condition}/*.png
#
# Usage:
#   qsub scripts/hpc/jobs/evaluate/pbs_visualize_split2.sh
#   qsub -v CONDITION=baseline scripts/hpc/jobs/evaluate/pbs_visualize_split2.sh
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# === Environment setup ===
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh || true
conda activate python310 || { echo "[ERROR] conda env failed"; exit 1; }

export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export MPLBACKEND=Agg

# === Optional condition filter ===
CONDITION_ARG=""
if [[ -n "${CONDITION:-}" ]]; then
    CONDITION_ARG="--condition $CONDITION"
    echo "[INFO] Filtering to condition=$CONDITION"
fi

echo "======================================================================"
echo "Job Started : $(date)"
echo "Job ID      : ${PBS_JOBID:-local}"
echo "Node        : $(hostname)"
echo "Python      : $(which python3)"
echo "======================================================================"

# === Run collection + visualization ===
python3 scripts/python/analysis/domain/collect_split2_metrics.py $CONDITION_ARG

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Job Finished: $(date)"
echo "Exit Code   : $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE

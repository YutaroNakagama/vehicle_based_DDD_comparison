#!/bin/bash
#PBS -N viz_prior_split2
#PBS -q TINY
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/evaluate/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/evaluate/
# ==============================================================================
# PBS Script: Collect split2 metrics & generate summary_metrics_bar plots
#             for Prior Research models (SvmW, SvmA, Lstm)
# ==============================================================================
# Scans SvmW / SvmA / Lstm evaluation JSONs with split2 tag, produces:
#   - CSV: results/analysis/exp3_prior_research/figures/csv/split2/{Model}/*.csv
#   - PNG: results/analysis/exp3_prior_research/figures/png/split2/{Model}/*.png
#
# Usage:
#   qsub scripts/hpc/jobs/evaluate/pbs_visualize_prior_research_split2.sh
#   qsub -v MODEL=SvmW scripts/hpc/jobs/evaluate/pbs_visualize_prior_research_split2.sh
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

# === Optional model filter ===
MODEL_ARG=""
if [[ -n "${MODEL:-}" ]]; then
    MODEL_ARG="--model $MODEL"
    echo "[INFO] Filtering to model=$MODEL"
fi

echo "======================================================================"
echo "Job Started : $(date)"
echo "Job ID      : ${PBS_JOBID:-local}"
echo "Node        : $(hostname)"
echo "Python      : $(which python3)"
echo "Script      : collect_split2_prior_research_metrics.py"
echo "======================================================================"

# === Run collection + visualization ===
python3 scripts/python/analysis/domain/collect_split2_prior_research_metrics.py $MODEL_ARG

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Job Finished: $(date)"
echo "Exit Code   : $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE

#!/bin/bash
#PBS -N viz_imbalance
#PBS -q TINY
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=00:15:00
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/logs/visualize_imbalance.out
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/logs/visualize_imbalance.err

# ==============================================================================
# PBS Script: Visualize Imbalance Domain Analysis
# ==============================================================================
# Generates 12 summary_metrics_bar plots:
#   3 ranking methods (knn, lof, median_distance) ×
#   4 imbalance methods (baseline, smote, smote_tomek, smote_rus)
#
# Output: results/domain_analysis/summary/png/{ranking_method}/
#         summary_metrics_bar_{imbalance_method}.png
# ==============================================================================

echo "======================================================================"
echo "Job Started: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "======================================================================"

# Change to project directory
cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison

# Activate conda environment
source /home/s2240011/.bashrc
conda activate python310

echo ""
echo "[INFO] Environment: $(conda info --envs | grep '*')"
echo "[INFO] Python: $(which python)"
echo ""

# Step 1: Collect metrics from all job IDs
echo "[STEP 1] Collecting metrics from all job IDs..."
python scripts/python/analysis/imbalance/collect_imbalance_metrics.py

# Step 2: Generate visualization plots
echo ""
echo "[STEP 2] Generating visualization plots..."
python scripts/python/analysis/imbalance/visualize_imbalance_domain.py

EXIT_CODE=$?

echo ""
echo "======================================================================"
echo "Job Finished: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "======================================================================"

exit $EXIT_CODE

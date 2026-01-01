#!/bin/bash
#PBS -N viz_imbal_v2
#PBS -q TINY
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/

# ==============================================================================
# PBS Script: Visualize Imbalance V2 Results
# ==============================================================================
# Generates visualizations for imbalance-only experiments (no domain analysis)
#
# Output: results/imbalance_analysis/v1/
#   - performance_dashboard.png (updated summary dashboard)
#   - auprc_comparison.png
#   - f2_comparison.png
#   - recall_precision_tradeoff.png
#   - metrics_heatmap.png
# ==============================================================================

echo "======================================================================"
echo "Job Started: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "======================================================================"

# Change to project directory
cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# Activate conda environment
source ~/conda/etc/profile.d/conda.sh
conda activate python310

echo ""
echo "[INFO] Environment: $(conda info --envs | grep '*')"
echo "[INFO] Python: $(which python)"
echo ""

# Output directory
OUTPUT_DIR="results/imbalance_analysis/v1"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "[INFO] Generating Imbalance V2 Visualizations"
echo "============================================================"
echo "  Output dir: $OUTPUT_DIR"
echo ""

# ==============================================================================
# Step 1: Performance metrics visualization (summary_dashboard, auprc, f2, etc.)
# ==============================================================================
echo "[Step 1/2] Generating performance metrics visualizations..."

# Use the default V2 model configurations from imbalance_visualize.py
# These are the imbalance-only experiments (no domain analysis)
python scripts/python/visualization/imbalance_visualize.py \
    --output-dir "$OUTPUT_DIR" \
    --plots all \
    --dpi 150 \
    --format png

VIZ_EXIT_CODE=$?

# Rename summary_dashboard.png to performance_dashboard.png for consistency
if [ -f "$OUTPUT_DIR/summary_dashboard.png" ]; then
    mv "$OUTPUT_DIR/summary_dashboard.png" "$OUTPUT_DIR/performance_dashboard.png"
    echo "[INFO] Renamed: summary_dashboard.png -> performance_dashboard.png"
fi

# ==============================================================================
# Step 2: Sample distribution visualization (train_val_test_split, etc.)
# ==============================================================================
echo ""
echo "[Step 2/2] Generating sample distribution visualizations..."
python -c "
from src.analysis.imbalance.sample_distribution import generate_all_visualizations
generate_all_visualizations(output_dir='$OUTPUT_DIR')
"

SAMPLE_EXIT_CODE=$?

# Report results
if [ $VIZ_EXIT_CODE -eq 0 ] && [ $SAMPLE_EXIT_CODE -eq 0 ]; then
    EXIT_CODE=0
    echo "[SUCCESS] All visualizations generated successfully"
else
    EXIT_CODE=1
    echo "[WARN] Some visualizations may have failed (viz: $VIZ_EXIT_CODE, sample: $SAMPLE_EXIT_CODE)"
fi

echo ""
echo "======================================================================"
echo "Job Finished: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "======================================================================"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*.png 2>/dev/null

exit $EXIT_CODE

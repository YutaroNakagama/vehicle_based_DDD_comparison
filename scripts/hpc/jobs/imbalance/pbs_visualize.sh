#!/bin/bash
#PBS -N imbal_v2_viz
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Imbalance Comparison V2: Result Visualization
# Generate analysis figures and summary tables
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

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-results/imbalance_analysis/v2_${PBS_JOBID}}"
ENSEMBLE_DIR="${ENSEMBLE_DIR:-}"
BASELINE_AUPRC="${BASELINE_AUPRC:-0.039}"
FORMAT="${FORMAT:-png}"
DPI="${DPI:-150}"

echo "============================================================"
echo "[IMBALANCE V2] Result Visualization"
echo "============================================================"
echo "PBS_JOBID: $PBS_JOBID"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "ENSEMBLE_DIR: $ENSEMBLE_DIR"
echo "BASELINE_AUPRC: $BASELINE_AUPRC"
echo "============================================================"

# Build command
CMD="python scripts/python/visualization/imbalance_visualize.py"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --baseline-auprc $BASELINE_AUPRC"
CMD="$CMD --format $FORMAT"
CMD="$CMD --dpi $DPI"
CMD="$CMD --verbose"

# Add ensemble directory if specified
if [ -n "$ENSEMBLE_DIR" ]; then
    CMD="$CMD --ensemble-dir $ENSEMBLE_DIR"
fi

echo "Running: $CMD"
$CMD

echo ""
echo "============================================================"
echo "VISUALIZATION COMPLETE"
echo "============================================================"
echo "Output directory: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
echo "============================================================"

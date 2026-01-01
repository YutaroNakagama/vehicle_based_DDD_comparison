#!/bin/bash
#PBS -N RF_baseline_viz
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# === Environment setup ===
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh || true
conda activate python310 || { echo "[ERROR] conda env failed"; exit 1; }

export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export MPLBACKEND=Agg

MODEL="${MODEL:-RF}"
MODE="${MODE:-pooled}"

echo "[INFO] Visualizing baseline metrics for MODEL=$MODEL, MODE=$MODE"

python3 scripts/python/visualization/visualize_baseline_metrics.py \
    --model "$MODEL" \
    --mode "$MODE"

echo "[INFO] === BASELINE VISUALIZATION DONE ==="

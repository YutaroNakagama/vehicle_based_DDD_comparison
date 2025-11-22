#!/bin/bash
#PBS -N domain_distance_test
#PBS -J 1-3
#PBS -l select=1:ncpus=8
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

set -euo pipefail

# === Set working directory to project root ===
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# === Conda environment setup ===
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# === Thread and backend settings ===
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1

echo "[TEST MODE] Job started at $(date)"
echo "[INFO] Job ID: ${PBS_JOBID:-unknown}, Array Index: ${PBS_ARRAY_INDEX:-N/A}"

# === Metric selection based on array index ===
METRICS=("mmd" "wasserstein" "dtw")
METRIC=${METRICS[$((PBS_ARRAY_INDEX-1))]}

echo "[INFO] Running metric: ${METRIC} (TEST MODE with 8 cores)"

# Use reduced parallelism for test
python "${PROJECT_ROOT}/scripts/python/analyze.py" comp-dist --metric "$METRIC" \
  --subject_list ../dataset/mdapbe/subject_list.txt \
  --data_root "${PROJECT_ROOT}/data/processed/common" \
  --groups_file "${PROJECT_ROOT}/config/subjects/target_groups.txt" \
  --n_jobs 4

echo "[DONE] ${METRIC} computation complete."
echo "[INFO] Results saved to results/domain_analysis/distance/${METRIC}"

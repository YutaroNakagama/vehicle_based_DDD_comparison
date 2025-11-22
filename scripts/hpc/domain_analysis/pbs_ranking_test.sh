#!/bin/bash
#PBS -N domain_ranking_test
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1

echo "[TEST MODE] Generating rankings at $(date)"

# Generate rankings (this creates ranks29 directory and files)
python "${PROJECT_ROOT}/scripts/python/analyze.py" rank-export \
  --k 29 \
  --method mean \
  --outdir "${PROJECT_ROOT}/results/domain_analysis/distance/ranks29"

echo "[DONE] Ranking complete. Created ranks29/ directory with high/middle/low splits."
echo "[INFO] Results saved to results/domain_analysis/distance/ranks29/"

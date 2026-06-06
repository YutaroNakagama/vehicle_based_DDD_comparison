#!/usr/bin/env bash
# Launch the WSL2 cuML SvmA Phase-2 unified full run with 1 worker.
# Use this config when 2 workers cause GPU compute contention (observed:
# 2W cache=2048 gave 13.9h per job, worse than 1W cache=4096 at 4.5h).
#
# Run with:
#   nohup bash scripts/bash/svma_cuml_full_1w.sh > /home/ynakagama/svma_cuml_full.log 2>&1 &
set -u
source /home/ynakagama/.venv_svma_cuml/bin/activate
cd /mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison

# Single worker — entire RTX 3060 dedicated to one cuML process.
# cache_size 4096 MiB matches the validation config that achieved 4.5h/job.
export LOCAL_PARALLEL_SVMA=1
export SVMA_USE_CUML=1
export SVMA_CACHE_SIZE_MIB=4096
export SVMA_PSO_PROCESSES=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[FULL1W] Started $(date)"
echo "[FULL1W] LOCAL_PARALLEL_SVMA=${LOCAL_PARALLEL_SVMA}"
echo "[FULL1W] SVMA_CACHE_SIZE_MIB=${SVMA_CACHE_SIZE_MIB}"
echo "[FULL1W] SVMA_USE_CUML=${SVMA_USE_CUML}"

python scripts/python/train/local_exp3_svma_cuml_launcher.py

echo "[FULL1W] Finished $(date)"

#!/usr/bin/env bash
# Launch the WSL2 cuML SvmA Phase-2 unified full run (12 seeds x 144 jobs).
# Designed to be invoked via:
#   nohup bash scripts/bash/svma_cuml_full_2w.sh > /home/ynakagama/svma_cuml_full.log 2>&1 &
set -u
source /home/ynakagama/.venv_svma_cuml/bin/activate
cd /mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison

# 2 concurrent worker threads, each spawning its own cuML train.py subprocess.
# cache_size 2048 MiB x 2 workers + cuML overhead < 6 GiB VRAM total.
export LOCAL_PARALLEL_SVMA=2
export SVMA_USE_CUML=1
export SVMA_CACHE_SIZE_MIB=2048
export SVMA_PSO_PROCESSES=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[FULL2W] Started $(date)"
echo "[FULL2W] LOCAL_PARALLEL_SVMA=${LOCAL_PARALLEL_SVMA}"
echo "[FULL2W] SVMA_CACHE_SIZE_MIB=${SVMA_CACHE_SIZE_MIB}"
echo "[FULL2W] SVMA_USE_CUML=${SVMA_USE_CUML}"

python scripts/python/train/local_exp3_svma_cuml_launcher.py

echo "[FULL2W] Finished $(date)"

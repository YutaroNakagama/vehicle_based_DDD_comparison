#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
# ============================================================
# Eval-only retry — minimal resources.
# Required env: PROJECT, MODEL=Lstm|SvmA|SvmW, TAG, KIND=within|cross, JID, TGT
# ============================================================
set +u
PROJECT_ROOT="${PROJECT:-/home/s2240011/git/ddd/vehicle_based_DDD_comparison}"
cd "$PROJECT_ROOT"
export PATH="${PATH:-/usr/bin:/bin}:$HOME/conda/bin"
source $HOME/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Force CPU; eval is light so few threads are enough.
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

set -u
MODEL="${MODEL:-Lstm}"
echo "[EVAL-RETRY] model=$MODEL tag=$TAG kind=$KIND jid=$JID target=$TGT"

python scripts/python/evaluation/evaluate.py \
    --model "$MODEL" --tag "$TAG" --mode domain_train \
    --target_file "$TGT" --eval_type "$KIND" --jobid "$JID"

echo "[DONE] model=$MODEL tag=$TAG kind=$KIND"

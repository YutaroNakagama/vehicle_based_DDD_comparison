#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
# ============================================================
# Eval-only retry for one Lstm tag×eval_type, using existing model.
# Required env: PROJECT, MODEL=Lstm, TAG, KIND=within|cross, JID, TGT
# ============================================================
set +u
PROJECT_ROOT="${PROJECT:-/home/s2240011/git/ddd/vehicle_based_DDD_comparison}"
cd "$PROJECT_ROOT"
export PATH="${PATH:-/usr/bin:/bin}:$HOME/conda/bin"
source $HOME/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Force CPU — Lstm eval is fast on CPU and avoids the H100 GPU init issues
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2

echo "============================================================"
echo "[LSTM-EVAL-RETRY] tag=$TAG kind=$KIND jid=$JID"
echo "  target=$TGT"
echo "============================================================"

set -u
python scripts/python/evaluation/evaluate.py \
    --model Lstm \
    --tag "$TAG" \
    --mode domain_train \
    --target_file "$TGT" \
    --eval_type "$KIND" \
    --jobid "$JID"

echo ""
echo "[DONE] eval-only retry complete (tag=$TAG kind=$KIND)"

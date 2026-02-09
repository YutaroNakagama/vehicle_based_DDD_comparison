#!/bin/bash
# ============================================================
# TINYキューで失敗したジョブの再投入
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/imbalance/pbs_imbalance_comparison.sh"

N_TRIALS=100

echo "============================================================"
echo "TINYキューで失敗したジョブの再投入"
echo "============================================================"
echo ""

# Baseline (2ジョブ)
echo "[1/8] Submitting: baseline (seed=42) → SINGLE"
qsub -N baseli_s42 -l select=1:ncpus=4:mem=8gb -l walltime=04:00:00 -q SINGLE \
     -v METHOD=baseline,SEED=42,N_TRIALS=$N_TRIALS $JOB_SCRIPT

echo "[2/8] Submitting: baseline (seed=123) → SINGLE"
qsub -N baseli_s123 -l select=1:ncpus=4:mem=8gb -l walltime=04:00:00 -q SINGLE \
     -v METHOD=baseline,SEED=123,N_TRIALS=$N_TRIALS $JOB_SCRIPT

# Random Undersampling (4ジョブ)
echo "[3/8] Submitting: undersample_rus (ratio=0.1, seed=42) → SINGLE"
qsub -N unders_r0.1_s42 -l select=1:ncpus=4:mem=8gb -l walltime=04:00:00 -q SINGLE \
     -v METHOD=undersample_rus,RATIO=0.1,SEED=42,N_TRIALS=$N_TRIALS $JOB_SCRIPT

echo "[4/8] Submitting: undersample_rus (ratio=0.5, seed=42) → SINGLE"
qsub -N unders_r0.5_s42 -l select=1:ncpus=4:mem=8gb -l walltime=04:00:00 -q SINGLE \
     -v METHOD=undersample_rus,RATIO=0.5,SEED=42,N_TRIALS=$N_TRIALS $JOB_SCRIPT

echo "[5/8] Submitting: undersample_rus (ratio=0.1, seed=123) → SINGLE"
qsub -N unders_r0.1_s123 -l select=1:ncpus=4:mem=8gb -l walltime=04:00:00 -q SINGLE \
     -v METHOD=undersample_rus,RATIO=0.1,SEED=123,N_TRIALS=$N_TRIALS $JOB_SCRIPT

echo "[6/8] Submitting: undersample_rus (ratio=0.5, seed=123) → SINGLE"
qsub -N unders_r0.5_s123 -l select=1:ncpus=4:mem=8gb -l walltime=04:00:00 -q SINGLE \
     -v METHOD=undersample_rus,RATIO=0.5,SEED=123,N_TRIALS=$N_TRIALS $JOB_SCRIPT

# Balanced RF (2ジョブ) - LONGキュー使用
echo "[7/8] Submitting: balanced_rf (seed=42) → LONG"
qsub -N balanc_s42 -l select=1:ncpus=8:mem=8gb -l walltime=08:00:00 -q LONG \
     -v METHOD=balanced_rf,SEED=42,N_TRIALS=$N_TRIALS $JOB_SCRIPT

echo "[8/8] Submitting: balanced_rf (seed=123) → LONG"
qsub -N balanc_s123 -l select=1:ncpus=8:mem=8gb -l walltime=08:00:00 -q LONG \
     -v METHOD=balanced_rf,SEED=123,N_TRIALS=$N_TRIALS $JOB_SCRIPT

echo ""
echo "============================================================"
echo "8ジョブを再投入しました"
echo "============================================================"

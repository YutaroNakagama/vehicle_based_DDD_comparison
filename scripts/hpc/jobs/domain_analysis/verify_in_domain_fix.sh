#!/bin/bash
#PBS -N verify_fix
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -J 1-2
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ==============================================================================
# verify_in_domain_fix.sh - 修正確認用ジョブ
# ==============================================================================
# in_domain で source_only と target_only が同じ結果になることを確認
# 
# Task 1: source_only (knn, mmd, in_domain)
# Task 2: target_only (knn, mmd, in_domain)
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# === Ultra-fast: N_TRIALS=1 ===
export N_TRIALS_OVERRIDE=1

# === Fixed parameters for verification ===
RANKING_METHOD="knn"
DISTANCE="mmd"
LEVEL="in_domain"

TASK_ID=${PBS_ARRAY_INDEX:-1}

# Determine mode based on task ID
if [[ $TASK_ID -eq 1 ]]; then
    MODE="source_only"
else
    MODE="target_only"
fi

echo "============================================================"
echo "[INFO] Fix Verification Job Started at $(date)"
echo "============================================================"
echo "Task ID: ${TASK_ID}"
echo "Ranking Method: ${RANKING_METHOD}"
echo "Distance: ${DISTANCE}, Level: ${LEVEL}"
echo "Mode: ${MODE}"
echo "N_TRIALS: ${N_TRIALS_OVERRIDE}"
echo "============================================================"

TAG="verify_fix_rank_${RANKING_METHOD}_${DISTANCE}_${LEVEL}"

# Get subject list from correct ranking method directory
SUBJECT_LIST="results/domain_analysis/distance/subject-wise/ranks/ranks29/${RANKING_METHOD}/${DISTANCE}_${LEVEL}.txt"

if [[ ! -f "$SUBJECT_LIST" ]]; then
    echo "[ERROR] Subject list not found: $SUBJECT_LIST"
    echo "[INFO] Checking available files..."
    ls -la "results/domain_analysis/distance/subject-wise/ranks/ranks29/${RANKING_METHOD}/" || true
    exit 1
fi

echo "[INFO] Using subject list: $SUBJECT_LIST"
echo "[INFO] Subject count: $(wc -l < "$SUBJECT_LIST")"

# Get PBS job ID for this run
CURRENT_JOBID="${PBS_JOBID:-local}"
echo "[INFO] Current Job ID: $CURRENT_JOBID"

# Run training
echo "[INFO] Starting training..."
python scripts/python/train.py \
    --model RF \
    --mode "$MODE" \
    --tag "$TAG" \
    --target_file "$SUBJECT_LIST"

echo "============================================================"
echo "[INFO] Training Completed at $(date)"
echo "============================================================"

# Run evaluation with explicit jobid
echo "[INFO] Starting evaluation..."
python scripts/python/evaluate.py \
    --model RF \
    --mode "$MODE" \
    --tag "$TAG" \
    --target_file "$SUBJECT_LIST" \
    --jobid "$CURRENT_JOBID"

echo "============================================================"
echo "[INFO] Evaluation Completed at $(date)"
echo "============================================================"

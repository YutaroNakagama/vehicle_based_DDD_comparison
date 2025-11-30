#!/bin/bash
#PBS -q SINGLE
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=4:00:00
#PBS -J 1-6
#PBS -N knn_base
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/logs/knn_imbalance/out/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/logs/knn_imbalance/err/

# ============================================================
# KNN Ranking Baseline (No Imbalance Handling)
# ============================================================
# 
# Baseline experiments without any imbalance handling.
# 
# Array Job Layout (6 jobs):
#   Jobs 1-3: target_only  × 3 levels (out_domain, mid_domain, in_domain)
#   Jobs 4-6: source_only × 3 levels
# ============================================================

set -eu

# Load conda
source /etc/profile.d/modules.sh
module load miniconda/py310_24.7.1
source ~/.bashrc
conda activate python310

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison

# ============================================================
# Configuration
# ============================================================
MODEL="RF"
DISTANCE="mmd"
RANKING_METHOD="knn"

LEVELS=("out_domain" "mid_domain" "in_domain")
MODES=("target_only" "source_only")

GROUP_BASE="results/domain_analysis/distance/subject-wise/${DISTANCE}/groups/clustering_ranked"

# ============================================================
# Compute indices
# ============================================================
IDX=$((PBS_ARRAY_INDEX - 1))

N_LEVELS=3
MODE_IDX=$((IDX / N_LEVELS))
LEVEL_IDX=$((IDX % N_LEVELS))

MODE="${MODES[$MODE_IDX]}"
LEVEL="${LEVELS[$LEVEL_IDX]}"

GROUP_FILE="${GROUP_BASE}/${DISTANCE}_${RANKING_METHOD}_${LEVEL}.txt"
TAG="rank_${DISTANCE}_${RANKING_METHOD}_${LEVEL}_baseline"

echo "============================================================"
echo "KNN Ranking Baseline (No Imbalance Handling)"
echo "============================================================"
echo "PBS_ARRAY_INDEX: ${PBS_ARRAY_INDEX}"
echo "Mode:            ${MODE}"
echo "Level:           ${LEVEL}"
echo "Group File:      ${GROUP_FILE}"
echo "Tag:             ${TAG}"
echo "============================================================"

if [ ! -f "${GROUP_FILE}" ]; then
    echo "[ERROR] Group file not found: ${GROUP_FILE}"
    exit 1
fi

echo "Target subjects:"
cat "${GROUP_FILE}"
echo ""

# ============================================================
# Run training (NO oversampling)
# ============================================================
python scripts/python/train.py \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --target_file "${GROUP_FILE}" \
    --tag "${TAG}" \
    --subject_wise_split \
    --seed 42

echo ""
echo "[DONE] Baseline training completed for ${TAG}"

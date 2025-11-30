#!/bin/bash
#PBS -q SINGLE
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=8:00:00
#PBS -J 1-24
#PBS -N knn_imbal
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/logs/knn_imbalance/out/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/logs/knn_imbalance/err/

# ============================================================
# KNN Ranking + Imbalance Handling Comparison
# ============================================================
# 
# This script compares imbalance handling methods on KNN-ranked groups.
# 
# Experiment Design:
# - Ranking Method: KNN (best performing)
# - Distance Metric: MMD (most stable)
# - Levels: out_domain, mid_domain, in_domain
# - Modes: target_only (best), source_only
# - Imbalance Methods: none, undersample_rus, undersample_tomek, smote_rus, smote_tomek
#
# Array Job Layout (24 jobs total):
#   Jobs 1-12:  target_only  × 3 levels × 4 imbalance methods
#   Jobs 13-24: source_only × 3 levels × 4 imbalance methods
#
# Imbalance methods (simplified, ordered by complexity):
#   1. none              - Baseline (no resampling)
#   2. undersample_rus   - Random Under-Sampling only
#   3. undersample_tomek - Tomek Links only
#   4. smote_rus         - SMOTE + Random Under-Sampling
#   5. smote_tomek       - SMOTE + Tomek Links
# ============================================================

set -eu

# Load conda
source ~/conda/etc/profile.d/conda.sh
conda activate python310

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

# ============================================================
# Configuration
# ============================================================
MODEL="RF"
DISTANCE="mmd"
RANKING_METHOD="knn"

# Levels and modes
LEVELS=("out_domain" "mid_domain" "in_domain")
MODES=("target_only" "source_only")

# Imbalance methods to compare (from simple to complex)
IMBAL_METHODS=("none" "undersample_rus" "undersample_tomek" "smote_rus" "smote_tomek")

# Group file base path
GROUP_BASE="results/domain_analysis/distance/subject-wise/${DISTANCE}/groups/clustering_ranked"

# ============================================================
# Compute indices from PBS_ARRAY_INDEX
# ============================================================
IDX=$((PBS_ARRAY_INDEX - 1))

# Total: 2 modes × 3 levels × 4 imbalance methods (excluding "none" for simplicity, or include all 5)
# Let's use 4 methods: undersample_rus, undersample_tomek, smote_rus, smote_tomek
# Plus baseline (none) = 5 methods
# But to keep it manageable, let's do:
#   - 3 levels × 4 methods × 2 modes = 24 jobs
# Methods: undersample_rus(0), undersample_tomek(1), smote_rus(2), smote_tomek(3)

# Calculate indices
N_LEVELS=3
N_METHODS=4
N_MODES=2

IMBAL_METHODS_SUBSET=("undersample_rus" "undersample_tomek" "smote_rus" "smote_tomek")

MODE_IDX=$((IDX / (N_LEVELS * N_METHODS)))
REMAINDER=$((IDX % (N_LEVELS * N_METHODS)))
LEVEL_IDX=$((REMAINDER / N_METHODS))
METHOD_IDX=$((REMAINDER % N_METHODS))

MODE="${MODES[$MODE_IDX]}"
LEVEL="${LEVELS[$LEVEL_IDX]}"
IMBAL_METHOD="${IMBAL_METHODS_SUBSET[$METHOD_IDX]}"

# Group file
GROUP_FILE="${GROUP_BASE}/${DISTANCE}_${RANKING_METHOD}_${LEVEL}.txt"

# Tag for this experiment
TAG="rank_${DISTANCE}_${RANKING_METHOD}_${LEVEL}_${IMBAL_METHOD}"

echo "============================================================"
echo "KNN Ranking + Imbalance Comparison"
echo "============================================================"
echo "PBS_ARRAY_INDEX: ${PBS_ARRAY_INDEX}"
echo "Mode:            ${MODE}"
echo "Level:           ${LEVEL}"
echo "Imbalance:       ${IMBAL_METHOD}"
echo "Group File:      ${GROUP_FILE}"
echo "Tag:             ${TAG}"
echo "============================================================"

# Check group file exists
if [ ! -f "${GROUP_FILE}" ]; then
    echo "[ERROR] Group file not found: ${GROUP_FILE}"
    exit 1
fi

echo "Target subjects:"
cat "${GROUP_FILE}"
echo ""

# ============================================================
# Run training
# ============================================================
python scripts/python/train.py \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --target_file "${GROUP_FILE}" \
    --tag "${TAG}" \
    --use_oversampling \
    --oversample_method "${IMBAL_METHOD}" \
    --subject_wise_split \
    --seed 42

echo ""
echo "[DONE] Training completed for ${TAG}"

#!/bin/bash
#PBS -q SINGLE
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=2:00:00
#PBS -J 1-30
#PBS -N knn_eval
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/logs/knn_imbalance/out/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/logs/knn_imbalance/err/

# ============================================================
# Evaluation for KNN Ranking + Imbalance Experiments
# ============================================================
#
# Array Job Layout (30 jobs):
#   2 modes × 3 levels × 5 methods = 30 jobs
#
# Methods:
#   0: baseline (no resampling)
#   1: undersample_rus
#   2: undersample_tomek
#   3: smote_rus
#   4: smote_tomek
# ============================================================

set -eu

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
IMBAL_METHODS=("baseline" "undersample_rus" "undersample_tomek" "smote_rus" "smote_tomek")

GROUP_BASE="results/domain_analysis/distance/subject-wise/${DISTANCE}/groups/clustering_ranked"

# ============================================================
# Compute indices
# ============================================================
IDX=$((PBS_ARRAY_INDEX - 1))

N_LEVELS=3
N_METHODS=5
N_MODES=2

MODE_IDX=$((IDX / (N_LEVELS * N_METHODS)))
REMAINDER=$((IDX % (N_LEVELS * N_METHODS)))
LEVEL_IDX=$((REMAINDER / N_METHODS))
METHOD_IDX=$((REMAINDER % N_METHODS))

MODE="${MODES[$MODE_IDX]}"
LEVEL="${LEVELS[$LEVEL_IDX]}"
IMBAL_METHOD="${IMBAL_METHODS[$METHOD_IDX]}"

GROUP_FILE="${GROUP_BASE}/${DISTANCE}_${RANKING_METHOD}_${LEVEL}.txt"
TAG="rank_${DISTANCE}_${RANKING_METHOD}_${LEVEL}_${IMBAL_METHOD}"

echo "============================================================"
echo "KNN Ranking + Imbalance Evaluation"
echo "============================================================"
echo "PBS_ARRAY_INDEX: ${PBS_ARRAY_INDEX}"
echo "Mode:            ${MODE}"
echo "Level:           ${LEVEL}"
echo "Imbalance:       ${IMBAL_METHOD}"
echo "Tag:             ${TAG}"
echo "============================================================"

# Find latest job ID for this tag
MODEL_DIR="models/${MODEL}"
LATEST_JOBID=""

for dir in "${MODEL_DIR}"/*; do
    if [ -d "${dir}" ]; then
        jobid=$(basename "${dir}")
        # Check if this job matches our tag
        if [ -f "${dir}/${jobid}[0]/train_config.json" ]; then
            config_tag=$(python3 -c "import json; print(json.load(open('${dir}/${jobid}[0]/train_config.json')).get('tag', ''))" 2>/dev/null || echo "")
            if [ "${config_tag}" == "${TAG}" ]; then
                LATEST_JOBID="${jobid}"
            fi
        fi
    fi
done

if [ -z "${LATEST_JOBID}" ]; then
    echo "[ERROR] No model found for tag: ${TAG}"
    exit 1
fi

echo "Found model: ${LATEST_JOBID}"

# ============================================================
# Run evaluation
# ============================================================
python scripts/python/evaluate.py \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --target_file "${GROUP_FILE}" \
    --tag "${TAG}" \
    --base_jobid "${LATEST_JOBID}"

echo ""
echo "[DONE] Evaluation completed for ${TAG}"

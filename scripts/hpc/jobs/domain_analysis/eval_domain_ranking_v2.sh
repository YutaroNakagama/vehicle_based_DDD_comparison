#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe
#PBS -M yutaro.nakagama@bosch.com
#PBS -m ae

# ==============================================================================
# eval_domain_ranking_v2.sh - Uses TRAIN_JOBID to find the correct model
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Environment setup
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Get parameters
RANKING_METHOD="${RANKING_METHOD:-knn}"
DISTANCE_METRIC="${DISTANCE_METRIC:-dtw}"
DOMAIN_LEVEL="${DOMAIN_LEVEL:-out_domain}"
MODE="${MODE:-source_only}"
SEED="${SEED:-42}"
TRAIN_JOBID="${TRAIN_JOBID:-}"

echo "============================================================"
echo "[INFO] Domain Ranking Evaluation V2 - Started at $(date)"
echo "============================================================"
echo "Ranking Method: ${RANKING_METHOD}"
echo "Distance Metric: ${DISTANCE_METRIC}"
echo "Domain Level: ${DOMAIN_LEVEL}"
echo "Mode: ${MODE}"
echo "Seed: ${SEED}"
echo "Training Job ID: ${TRAIN_JOBID}"
echo "PBS_JOBID: ${PBS_JOBID:-local}"
echo "============================================================"

# Build subject list path
SUBJECT_LIST="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING_METHOD}/${DISTANCE_METRIC}_${DOMAIN_LEVEL}.txt"

if [[ ! -f "$SUBJECT_LIST" ]]; then
    echo "[ERROR] Subject list not found: $SUBJECT_LIST"
    exit 1
fi

echo "[INFO] Using subject list: $SUBJECT_LIST"

# Build tag
TAG="rank_cmp_${RANKING_METHOD}_${DISTANCE_METRIC}_${DOMAIN_LEVEL}_s${SEED}"
echo "[INFO] Tag: $TAG"

# Use training job ID for evaluation
if [[ -n "$TRAIN_JOBID" ]]; then
    JOBID_ARG="--jobid ${TRAIN_JOBID}"
    echo "[INFO] Using training jobid: $TRAIN_JOBID"
else
    JOBID_ARG=""
    echo "[WARN] No training jobid provided, using default discovery"
fi

# Run evaluation
echo "[INFO] Starting evaluation..."
python scripts/python/evaluation/evaluate.py \
    --model RF \
    --mode pooled \
    --tag "$TAG" \
    --target_file "$SUBJECT_LIST" \
    $JOBID_ARG

echo "============================================================"
echo "[INFO] Evaluation Completed Successfully at $(date)"
echo "============================================================"

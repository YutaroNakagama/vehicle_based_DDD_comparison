#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -j oe
#PBS -M yutaro.nakagama@bosch.com
#PBS -m ae

# ==============================================================================
# train_domain_ranking.sh - Domain Analysis Ranking Comparison Training Job
# ==============================================================================
# Environment variables expected:
#   RANKING_METHOD: mean_distance, median_distance, knn, lof, isolation_forest, centroid_umap
#   DISTANCE_METRIC: dtw, mmd, wasserstein
#   DOMAIN_LEVEL: out_domain, in_domain, cross_domain
#   MODE: source_only, target_only, pooled
#   SEED: random seed (42, 123, 456, etc.)
#   N_TRIALS_OVERRIDE: number of Optuna trials (default: 75)
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Environment setup
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Get parameters from environment or use defaults
RANKING_METHOD="${RANKING_METHOD:-knn}"
DISTANCE_METRIC="${DISTANCE_METRIC:-dtw}"
DOMAIN_LEVEL="${DOMAIN_LEVEL:-out_domain}"
MODE="${MODE:-source_only}"
SEED="${SEED:-42}"
N_TRIALS="${N_TRIALS_OVERRIDE:-75}"

# Export N_TRIALS_OVERRIDE for train.py
export N_TRIALS_OVERRIDE="${N_TRIALS}"

echo "============================================================"
echo "[INFO] Domain Ranking Comparison - Training Started at $(date)"
echo "============================================================"
echo "Ranking Method: ${RANKING_METHOD}"
echo "Distance Metric: ${DISTANCE_METRIC}"
echo "Domain Level: ${DOMAIN_LEVEL}"
echo "Mode: ${MODE}"
echo "Seed: ${SEED}"
echo "N_TRIALS: ${N_TRIALS}"
echo "PBS_JOBID: ${PBS_JOBID:-local}"
echo "============================================================"

# Build subject list path
SUBJECT_LIST="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING_METHOD}/${DISTANCE_METRIC}_${DOMAIN_LEVEL}.txt"

if [[ ! -f "$SUBJECT_LIST" ]]; then
    echo "[ERROR] Subject list not found: $SUBJECT_LIST"
    echo "[INFO] Available files in ranks29/${RANKING_METHOD}/:"
    ls -la "results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING_METHOD}/" 2>/dev/null || echo "Directory not found"
    exit 1
fi

echo "[INFO] Using subject list: $SUBJECT_LIST"
echo "[INFO] Subject count: $(wc -l < "$SUBJECT_LIST")"

# Build tag
TAG="rank_cmp_${RANKING_METHOD}_${DISTANCE_METRIC}_${DOMAIN_LEVEL}_s${SEED}"
echo "[INFO] Tag: $TAG"

# Run training
echo "[INFO] Starting training..."
python scripts/python/train/train.py \
    --model RF \
    --mode "$MODE" \
    --tag "$TAG" \
    --target_file "$SUBJECT_LIST" \
    --seed "$SEED"

echo "============================================================"
echo "[INFO] Training Completed Successfully at $(date)"
echo "============================================================"

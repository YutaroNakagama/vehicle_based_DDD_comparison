#!/bin/bash
#PBS -N verify_2x3x2x2
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -J 1-24
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ==============================================================================
# verify_in_domain_full.sh - 修正確認用ジョブ（フルテスト）
# ==============================================================================
# ランキング手法 2種 × 距離指標 3種 × 不均衡対策 2種 × モード 2種 = 24ジョブ
# 
# ランキング手法: knn, median_distance
# 距離指標: dtw, mmd, wasserstein
# 不均衡対策: none, smote_tomek
# モード: source_only, target_only
#
# N_TRIALS=10
# 推定時間: 約10分/ジョブ
# ==============================================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# === N_TRIALS=10 ===
export N_TRIALS_OVERRIDE=10

# === Fixed parameters ===
LEVEL="in_domain"

# === Task mapping (24 tasks) ===
# Task 1-12: source_only
# Task 13-24: target_only
# Within each mode (12 tasks):
#   Ranking: knn (1-6), median_distance (7-12)
#   Within each ranking (6 tasks):
#     Distance: dtw (1-2), mmd (3-4), wasserstein (5-6)
#     Within each distance (2 tasks):
#       Imbalance: none (1), smote_tomek (2)

TASK_ID=${PBS_ARRAY_INDEX:-1}

# Determine mode
if [[ $TASK_ID -le 12 ]]; then
    MODE="source_only"
    LOCAL_ID=$TASK_ID
else
    MODE="target_only"
    LOCAL_ID=$((TASK_ID - 12))
fi

# Mapping arrays
declare -A RANKING_MAP=(
    [1]="knn" [2]="knn" [3]="knn" [4]="knn" [5]="knn" [6]="knn"
    [7]="median_distance" [8]="median_distance" [9]="median_distance" 
    [10]="median_distance" [11]="median_distance" [12]="median_distance"
)

declare -A DISTANCE_MAP=(
    [1]="dtw" [2]="dtw" [3]="mmd" [4]="mmd" [5]="wasserstein" [6]="wasserstein"
    [7]="dtw" [8]="dtw" [9]="mmd" [10]="mmd" [11]="wasserstein" [12]="wasserstein"
)

declare -A IMBALANCE_MAP=(
    [1]="none" [2]="smote_tomek" [3]="none" [4]="smote_tomek" [5]="none" [6]="smote_tomek"
    [7]="none" [8]="smote_tomek" [9]="none" [10]="smote_tomek" [11]="none" [12]="smote_tomek"
)

RANKING_METHOD="${RANKING_MAP[$LOCAL_ID]}"
DISTANCE="${DISTANCE_MAP[$LOCAL_ID]}"
IMBALANCE_METHOD="${IMBALANCE_MAP[$LOCAL_ID]}"

echo "============================================================"
echo "[INFO] Fix Verification Job (Full) Started at $(date)"
echo "============================================================"
echo "Task ID: ${TASK_ID} (Local: ${LOCAL_ID})"
echo "Ranking Method: ${RANKING_METHOD}"
echo "Distance: ${DISTANCE}, Level: ${LEVEL}"
echo "Mode: ${MODE}"
echo "Imbalance Method: ${IMBALANCE_METHOD}"
echo "N_TRIALS: ${N_TRIALS_OVERRIDE}"
echo "============================================================"

# Build tag
if [[ "$IMBALANCE_METHOD" == "none" ]]; then
    TAG="verify_fix_rank_${RANKING_METHOD}_${DISTANCE}_${LEVEL}"
else
    TAG="verify_fix_rank_${RANKING_METHOD}_${DISTANCE}_${LEVEL}_${IMBALANCE_METHOD}"
fi

# Get subject list from correct ranking method directory
SUBJECT_LIST="results/domain_analysis/distance/subject-wise/ranks/ranks29/${RANKING_METHOD}/${DISTANCE}_${LEVEL}.txt"

if [[ ! -f "$SUBJECT_LIST" ]]; then
    echo "[ERROR] Subject list not found: $SUBJECT_LIST"
    exit 1
fi

echo "[INFO] Using subject list: $SUBJECT_LIST"
echo "[INFO] Subject count: $(wc -l < "$SUBJECT_LIST")"
echo "[INFO] Tag: $TAG"

# Get PBS job ID for this run
CURRENT_JOBID="${PBS_JOBID:-local}"
echo "[INFO] Current Job ID: $CURRENT_JOBID"

# Build train command
TRAIN_CMD="python scripts/python/train/train.py --model RF --mode $MODE --tag $TAG --target_file $SUBJECT_LIST"
if [[ "$IMBALANCE_METHOD" != "none" ]]; then
    TRAIN_CMD="$TRAIN_CMD --oversample_method $IMBALANCE_METHOD"
fi

# Run training
echo "[INFO] Starting training..."
echo "[CMD] $TRAIN_CMD"
eval "$TRAIN_CMD"

echo "============================================================"
echo "[INFO] Training Completed at $(date)"
echo "============================================================"

# Run evaluation with explicit jobid
echo "[INFO] Starting evaluation..."
python scripts/python/evaluation/evaluate.py \
    --model RF \
    --mode "$MODE" \
    --tag "$TAG" \
    --target_file "$SUBJECT_LIST" \
    --jobid "$CURRENT_JOBID"

echo "============================================================"
echo "[INFO] Evaluation Completed at $(date)"
echo "============================================================"

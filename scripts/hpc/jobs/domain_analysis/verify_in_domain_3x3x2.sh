#!/bin/bash
#PBS -N verify_3x3x2
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:15:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -J 1-18
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ==============================================================================
# verify_in_domain_3x3x2.sh - 修正確認用ジョブ（フルテスト）
# ==============================================================================
# ランキング手法 3種 × 不均衡対策 3種 × モード 2種 = 18ジョブ
# 
# ランキング手法: knn, lof, mean_distance
# 不均衡対策: none, smote, smote_tomek
# モード: source_only, target_only
#
# 推定時間: 約3分/ジョブ × 18 = 約54分（逐次）、約3分（並列）
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

# === Fixed parameters ===
DISTANCE="mmd"
LEVEL="in_domain"

# === Task mapping (18 tasks) ===
# Task 1-9: source_only
# Task 10-18: target_only
# Within each mode:
#   1-3: knn (none, smote, smote_tomek)
#   4-6: lof (none, smote, smote_tomek)
#   7-9: median_distance (none, smote, smote_tomek)

TASK_ID=${PBS_ARRAY_INDEX:-1}

# Determine mode
if [[ $TASK_ID -le 9 ]]; then
    MODE="source_only"
    LOCAL_ID=$TASK_ID
else
    MODE="target_only"
    LOCAL_ID=$((TASK_ID - 9))
fi

# Determine ranking method and imbalance method
declare -A RANKING_MAP=(
    [1]="knn" [2]="knn" [3]="knn"
    [4]="lof" [5]="lof" [6]="lof"
    [7]="median_distance" [8]="median_distance" [9]="median_distance"
)

declare -A IMBALANCE_MAP=(
    [1]="none" [2]="smote" [3]="smote_tomek"
    [4]="none" [5]="smote" [6]="smote_tomek"
    [7]="none" [8]="smote" [9]="smote_tomek"
)

RANKING_METHOD="${RANKING_MAP[$LOCAL_ID]}"
IMBALANCE_METHOD="${IMBALANCE_MAP[$LOCAL_ID]}"

echo "============================================================"
echo "[INFO] Fix Verification Job (3x3x2) Started at $(date)"
echo "============================================================"
echo "Task ID: ${TASK_ID}"
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
SUBJECT_LIST="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING_METHOD}/${DISTANCE}_${LEVEL}.txt"

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

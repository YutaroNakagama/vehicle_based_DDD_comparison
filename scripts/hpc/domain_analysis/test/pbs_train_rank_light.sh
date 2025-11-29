#!/bin/bash
#PBS -N RF_train_light
#PBS -J 1-2
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ==============================================================================
# pbs_train_rank_light.sh - 軽量テスト版
# ==============================================================================
# Array Job 1-2: mmd_high, mmd_middle のみテスト
# N_TRIALS=3, source_onlyモードのみで高速検証
# ==============================================================================

set -euo pipefail
source ~/conda/etc/profile.d/conda.sh
conda activate python310

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# === 軽量テスト設定 ===
export N_TRIALS_OVERRIDE=3
export KSS_SIMPLIFIED=1

MODEL="${MODEL:-RF}"
SEED="${SEED:-42}"

# Ranking method (環境変数で指定可能)
RANKING_METHOD="${RANKING_METHOD:-mean_distance}"
RANK_NAMES_FILE="${PROJECT_ROOT}/results/domain_analysis/distance/subject-wise/ranks/ranks29/${RANKING_METHOD}/ranks29_names.txt"

if [[ ! -f "$RANK_NAMES_FILE" ]]; then
    echo "[ERROR] Ranks names file not found: $RANK_NAMES_FILE"
    exit 1
fi

IDX="${PBS_ARRAY_INDEX:-1}"
GROUP_FILE="$(sed -n "${IDX}p" "$RANK_NAMES_FILE" | tr -d '\r\n')"

if [[ -z "$GROUP_FILE" ]] || [[ ! -f "$GROUP_FILE" ]]; then
    echo "[ERROR] Group file not found: $GROUP_FILE"
    exit 1
fi

BASENAME="$(basename "$GROUP_FILE" .txt)"
TAG="rank_${RANKING_METHOD}_${BASENAME}_test"

echo "============================================================"
echo "[TEST] Light Training - ${RANKING_METHOD}"
echo "============================================================"
echo "  Array Index:    $IDX"
echo "  Group File:     $GROUP_FILE"
echo "  Tag:            $TAG"
echo "  N_TRIALS:       $N_TRIALS_OVERRIDE"
echo "  Mode:           source_only only (light test)"
echo "============================================================"

# source_onlyモードのみで高速テスト
echo "[TRAIN] source_only"
python "$PROJECT_ROOT/scripts/python/train.py" \
    --model "$MODEL" \
    --mode source_only \
    --target_file "$GROUP_FILE" \
    --tag "$TAG" \
    --seed "$SEED" \
    --time_stratify_labels

echo "[INFO] Light training completed: $TAG"

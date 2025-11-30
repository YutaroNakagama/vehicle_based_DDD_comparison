#!/bin/bash
#PBS -N RF_eval_light
#PBS -J 1-2
#PBS -l select=1:ncpus=2:mem=16gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ==============================================================================
# pbs_eval_rank_light.sh - 軽量テスト版
# ==============================================================================
# Array Job 1-2: mmd_out_domain, mmd_mid_domain のみテスト
# source_onlyモードのみで高速検証
# ==============================================================================

set -euo pipefail
source ~/conda/etc/profile.d/conda.sh
conda activate python310

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL="${MODEL:-RF}"

# Ranking method (環境変数で指定可能)
RANKING_METHOD="${RANKING_METHOD:-mean_distance}"
RANK_NAMES_FILE="${PROJECT_ROOT}/results/domain_analysis/distance/subject-wise/ranks/ranks29/${RANKING_METHOD}/ranks29_names.txt"

if [[ ! -f "$RANK_NAMES_FILE" ]]; then
    echo "[ERROR] Ranks names file not found: $RANK_NAMES_FILE"
    exit 1
fi

IDX="${PBS_ARRAY_INDEX:-1}"
GROUP_FILE="$(sed -n "${IDX}p" "$RANK_NAMES_FILE" | tr -d '\r\n')"

BASENAME="$(basename "$GROUP_FILE" .txt)"
TAG="rank_${RANKING_METHOD}_${BASENAME}_test"

echo "============================================================"
echo "[TEST] Light Evaluation - ${RANKING_METHOD}"
echo "============================================================"
echo "  Array Index:    $IDX"
echo "  Tag:            $TAG"
echo "  Mode:           source_only only (light test)"
echo "============================================================"

# source_onlyモードのみ
echo "[EVAL] source_only"
python "$PROJECT_ROOT/scripts/python/evaluate.py" \
    --model "$MODEL" \
    --mode source_only \
    --tag "$TAG"

echo "[INFO] Light evaluation completed: $TAG"

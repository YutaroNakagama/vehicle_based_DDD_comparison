#!/bin/bash
#PBS -N RF_eval
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

set -euo pipefail

# ===== env =====
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

PROJECT_ROOT="${PBS_O_WORKDIR}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# ===== settings =====
MODEL="${MODEL:-RF}"
RANK_NAMES_FILE="${RANK_NAMES_FILE:-}"
TAG_LIST="${TAG_LIST:-}"

# optional
FOLD="${FOLD:-}"   # cross-validation fold
MODE="${MODE:-pooled}"   # default evaluation mode aligned with training ("pooled")

if [[ -n "$RANK_NAMES_FILE" ]]; then
  IDX="${PBS_ARRAY_INDEX:-1}"
  NAME="$(sed -n "${IDX}p" "$RANK_NAMES_FILE" | tr -d '\r\n' || true)"
  if [[ -z "$NAME" ]]; then echo "NG: name empty (index=$IDX)"; exit 1; fi
  BASENAME="$(basename "$NAME" .txt)"
  TAG="rank_${BASENAME}"
elif [[ -n "$TAG_LIST" ]]; then
  IDX="${PBS_ARRAY_INDEX:-1}"
  TAG=$(echo "$TAG_LIST" | cut -d',' -f"$IDX")
else
  TAG="${TAG:-default}"
fi

echo "=== JOB INFO ==="
echo "MODEL=$MODEL"
echo "TAG=$TAG"
echo "FOLD=$FOLD"
echo "MODE=$MODE"

# ===== run evaluation =====
CMD=(python scripts/python/evaluation/evaluate.py --model "$MODEL" --tag "$TAG")

[[ -n "$FOLD" ]] && CMD+=(--fold "$FOLD")
CMD+=(--mode "$MODE")  # always pass mode explicitly (even if pooled)

"${CMD[@]}"

echo "=== DONE (TAG=$TAG) ==="


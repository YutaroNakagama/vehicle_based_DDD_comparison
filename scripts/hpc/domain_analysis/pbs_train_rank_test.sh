#!/bin/bash
#PBS -N RF_domain_train_test
#PBS -J 1-2
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/log/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

set -euo pipefail
source ~/conda/etc/profile.d/conda.sh
conda activate python310

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Test mode settings
export N_TRIALS_OVERRIDE=5
export KSS_SIMPLIFIED=1
export TRAIN_RATIO=0.4
export VAL_RATIO=0.3
export TEST_RATIO=0.3

MODEL="${MODEL:-RF}"
SEED="${SEED:-42}"

RANK_DIR="$PROJECT_ROOT/results/domain_analysis/distance/ranks29/"
RANK_NAMES_FILE="$PROJECT_ROOT/results/domain_analysis/distance/ranks29/ranks29_names.txt"
IDX="${PBS_ARRAY_INDEX:-1}"
GROUP_FILE="$(sed -n "${IDX}p" "$RANK_NAMES_FILE" | tr -d '\r\n')"
BASENAME="$(basename "$GROUP_FILE" .txt)"
TAG="rank_${BASENAME}_test"

echo "[INFO] [TEST MODE] Training $MODEL for $TAG (index=$IDX)"
echo "[INFO] N_TRIALS=$N_TRIALS_OVERRIDE, KSS_SIMPLIFIED=$KSS_SIMPLIFIED, TRAIN:VAL:TEST=$TRAIN_RATIO:$VAL_RATIO:$TEST_RATIO"

for MODE in pooled source_only target_only; do
  echo "[TRAIN] $MODE"
  python "$PROJECT_ROOT/scripts/python/train.py" \
    --model "$MODEL" \
    --mode "$MODE" \
    --target_file "$GROUP_FILE" \
    --tag "$TAG" \
    --seed "$SEED" \
    --time_stratify_labels \
    --time_stratify_tolerance 0.02 \
    --time_stratify_window 0.10 \
    --time_stratify_min_chunk 100
done

echo "[INFO] Training completed: $TAG"

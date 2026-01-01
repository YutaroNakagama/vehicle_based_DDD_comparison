#!/bin/bash
#PBS -N imbal_v2_array
#PBS -q LONG-L
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=168:00:00
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -J 1-29

set -euo pipefail

PROJECT_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
EXPERIMENT_LIST="${PROJECT_DIR}/scripts/hpc/logs/imbalance/experiment_list.txt"

source /home/s2240011/conda/etc/profile.d/conda.sh
conda activate python310

cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export N_TRIALS_OVERRIDE=75
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Get experiment parameters from list
LINE=$(sed -n "${PBS_ARRAY_INDEX}p" "$EXPERIMENT_LIST")
IFS=':' read -r METHOD RATIO SEED MODEL OVERSAMPLE_FLAG OVERSAMPLE_METHOD <<< "$LINE"

RATIO_SAFE="${RATIO//./_}"
TAG="imbal_v2_${METHOD}_ratio${RATIO_SAFE}_seed${SEED}"

echo "============================================================"
echo "[IMBALANCE V2 ARRAY] Training"
echo "============================================================"
echo "PBS_ARRAY_INDEX: ${PBS_ARRAY_INDEX}"
echo "METHOD: ${METHOD}"
echo "MODEL: ${MODEL}"
echo "RATIO: ${RATIO}"
echo "SEED: ${SEED}"
echo "TAG: ${TAG}"
echo "OVERSAMPLE: ${OVERSAMPLE_FLAG}"
echo "============================================================"

if [[ "$OVERSAMPLE_FLAG" == "yes" ]]; then
    python scripts/python/train/train.py \
        --model "$MODEL" \
        --mode pooled \
        --tag "$TAG" \
        --seed "$SEED" \
        --time_stratify_labels \
        --use_oversampling \
        --oversample_method "$OVERSAMPLE_METHOD" \
        --target_ratio "$RATIO"
else
    python scripts/python/train/train.py \
        --model "$MODEL" \
        --mode pooled \
        --tag "$TAG" \
        --seed "$SEED" \
        --time_stratify_labels
fi

echo "[DONE] Training complete: ${TAG}"

#!/usr/bin/env bash
# cuML SvmA validation runner — train + 2 evals for s42 mmd_in_domain ratio0.3.
# Run from WSL2:
#   nohup bash scripts/bash/svma_cuml_validate.sh > /home/ynakagama/svma_cuml_validate.log 2>&1 &
set -u
source /home/ynakagama/.venv_svma_cuml/bin/activate
cd /mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison

export PYTHONPATH=.
export SVMA_USE_CUML=1
export CUDA_VISIBLE_DEVICES=0
export SVMA_PSO_PROCESSES=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

JOBID="$(date +%s%N)"
export PBS_JOBID="${JOBID}"
TAG="prior_SvmA_imbalv3_knn_mmd_in_domain_domain_train_split2_subjectwise_ratio0.3_s42"
RANKINGS="results/analysis/exp2_domain_shift/distance/rankings/split2/knn"

echo "[VALIDATE] JOBID=${JOBID} TAG=${TAG}"
echo "[VALIDATE] Started $(date)"
echo "[VALIDATE] SVC backend: cuML"

set +e
python scripts/python/train/train.py \
    --model SvmA --mode domain_train --seed 42 \
    --target_file "${RANKINGS}/mmd_in_domain.txt" \
    --tag "${TAG}" \
    --time_stratify_labels --use_oversampling \
    --oversample_method smote --target_ratio 0.3 \
    --subject_wise_oversampling
RC_TRAIN=$?
echo "[VALIDATE] train.py rc=${RC_TRAIN} at $(date)"
if [ ${RC_TRAIN} -ne 0 ]; then exit ${RC_TRAIN}; fi

python scripts/python/evaluation/evaluate.py \
    --model SvmA --tag "${TAG}" --mode domain_train \
    --target_file "${RANKINGS}/mmd_in_domain.txt" \
    --eval_type within --jobid "${JOBID}"
RC_W=$?
echo "[VALIDATE] eval within rc=${RC_W} at $(date)"

python scripts/python/evaluation/evaluate.py \
    --model SvmA --tag "${TAG}" --mode domain_train \
    --target_file "${RANKINGS}/mmd_out_domain.txt" \
    --eval_type cross --jobid "${JOBID}"
RC_C=$?
echo "[VALIDATE] eval cross  rc=${RC_C} at $(date)"

echo "[VALIDATE] ALL DONE | rc_train=${RC_TRAIN} rc_within=${RC_W} rc_cross=${RC_C}"

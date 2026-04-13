#!/bin/bash
#PBS -N Ls_eval_fix
#PBS -l select=1:ncpus=8:ngpus=1:mem=8gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m a

# ============================================================
# PBS Job: Rerun missing Lstm eval (within/cross) for walltime-killed jobs
# Each array element runs one eval task from TASK_FILE
# Format: TAG|DOMAIN|DISTANCE|EVAL_TYPE|ORIG_JOBID
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# GPU / module setup
module load hpc_sdk/22.2 2>/dev/null || true
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
# Disable XLA JIT to avoid libdevice.10.bc not found error on some GPU nodes
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

TASK_FILE="${TASK_FILE:-}"
if [[ -z "$TASK_FILE" || ! -f "$TASK_FILE" ]]; then
    echo "[ERROR] TASK_FILE not set or not found: '$TASK_FILE'"
    exit 1
fi

LINE_NUM=$((PBS_ARRAY_INDEX + 1))
TASK_LINE=$(sed -n "${LINE_NUM}p" "$TASK_FILE")

if [[ -z "$TASK_LINE" ]]; then
    echo "[ERROR] No task at index $PBS_ARRAY_INDEX (line $LINE_NUM)"
    exit 1
fi

IFS='|' read -r TAG DOMAIN DISTANCE EVAL_TYPE ORIG_JOBID <<< "$TASK_LINE"

echo "============================================================"
echo "[EVAL-FIX] Index: $PBS_ARRAY_INDEX | Job: $PBS_JOBID"
echo "[EVAL-FIX] TAG=$TAG | DOMAIN=$DOMAIN | DIST=$DISTANCE | TYPE=$EVAL_TYPE | ORIG=$ORIG_JOBID"
echo "============================================================"

RANKING="knn"

if [[ "$EVAL_TYPE" == "within" ]]; then
    TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${DOMAIN}.txt"
elif [[ "$EVAL_TYPE" == "cross" ]]; then
    if [[ "$DOMAIN" == "in_domain" ]]; then
        CROSS_DOMAIN="out_domain"
    else
        CROSS_DOMAIN="in_domain"
    fi
    TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${CROSS_DOMAIN}.txt"
fi

echo "[EVAL-FIX] target_file=$TARGET_FILE"
echo "[EVAL-FIX] Start: $(date)"

python scripts/python/evaluation/evaluate.py \
    --model Lstm --tag "$TAG" --mode domain_train \
    --target_file "$TARGET_FILE" --eval_type "$EVAL_TYPE" \
    --jobid "${ORIG_JOBID}.spcc-adm1"
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[FAIL] Eval failed (exit=$EXIT_CODE) at $(date)"
    exit $EXIT_CODE
fi

echo "[DONE] Eval fix completed at $(date)"
exit 0

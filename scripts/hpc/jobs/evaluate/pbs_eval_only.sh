#!/bin/bash
# ============================================================
# Evaluation-only PBS job script
# Re-run evaluation for completed training jobs
# ============================================================
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/domain/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/domain/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# --- Variables (passed via qsub -v) ---
# MODEL        : RF
# TAG          : e.g. smote_plain_knn_wasserstein_out_domain_target_only_split2_ratio0.1_s42
# MODE         : target_only | source_only
# JOBID        : training job ID (e.g. 14735284)
# SEED         : random seed (default: 42)
# TARGET_FILE  : path to target subject list (optional)

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT" || exit 1

# Activate conda environment
source /etc/profile.d/modules.sh
module load miniconda/24.7.1
eval "$(conda shell.bash hook)"
conda activate python310

# Ensure src is importable
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

echo "============================================================"
echo "  Evaluation-only Job"
echo "  $(date)"
echo "============================================================"
echo "  MODEL       : ${MODEL:-RF}"
echo "  TAG         : ${TAG}"
echo "  MODE        : ${MODE}"
echo "  JOBID       : ${JOBID:-auto}"
echo "  SEED        : ${SEED:-42}"
echo "  TARGET_FILE : ${TARGET_FILE:-auto}"
echo "============================================================"

MODEL="${MODEL:-RF}"

EVAL_CMD="python scripts/python/evaluation/evaluate.py \
    --model $MODEL \
    --tag $TAG \
    --mode $MODE"

# If JOBID is specified, add it
if [[ -n "${JOBID:-}" ]]; then
    EVAL_CMD="$EVAL_CMD --jobid $JOBID"
fi

# If SEED is specified, add it
if [[ -n "${SEED:-}" ]]; then
    EVAL_CMD="$EVAL_CMD --seed $SEED"
fi

# If TARGET_FILE is specified, add it
if [[ -n "${TARGET_FILE:-}" ]]; then
    EVAL_CMD="$EVAL_CMD --target_file $TARGET_FILE"
fi

echo ""
echo "[EVAL] $EVAL_CMD"
echo ""

eval $EVAL_CMD
EVAL_EXIT=$?

if [[ $EVAL_EXIT -eq 0 ]]; then
    echo ""
    echo "[DONE] Evaluation completed successfully"
else
    echo ""
    echo "[ERROR] Evaluation failed with exit code $EVAL_EXIT"
fi

echo "============================================================"

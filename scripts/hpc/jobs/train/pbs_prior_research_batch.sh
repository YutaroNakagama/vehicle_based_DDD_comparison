#!/bin/bash
#PBS -N pooled_batch
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m ae

# =============================================================================
# PBS Batch Job: Run multiple pooled prior-research tasks in parallel
# =============================================================================
# Reads TASK_FILE (one "MODEL|CONDITION|SEED|RATIO" per line) and runs all
# tasks concurrently as background processes.
#
# Submitted by rerun_pooled_15seeds.sh daemon with:
#   qsub -v TASK_FILE=<path> -l select=1:ncpus=16:mem=64gb ...
# =============================================================================

set -u

# ===== Environment Setup =====
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

# Thread limits — each task uses 1 core
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export JOBLIB_MULTIPROCESSING=0
export JOBLIB_START_METHOD=spawn
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# TensorFlow — force CPU
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=""
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
cd "$PROJECT_ROOT"

TASK_FILE="${TASK_FILE:?TASK_FILE env var not set}"
JOB_ID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

echo "=============================================="
echo "  Pooled Batch Job"
echo "  PBS_JOBID: $JOB_ID"
echo "  TASK_FILE: $TASK_FILE"
echo "  START:     $(date)"
echo "=============================================="
cat "$TASK_FILE"
echo "----------------------------------------------"

# ===== Task runner =====
run_task() {
    local MODEL="$1" CONDITION="$2" SEED="$3" RATIO="$4"
    local LABEL="${MODEL}:${CONDITION}:s${SEED}"

    # Build tag
    local TAG
    case "$CONDITION" in
        baseline)    TAG="prior_${MODEL}_baseline_s${SEED}" ;;
        smote_plain) TAG="prior_${MODEL}_smote_plain_ratio${RATIO}_s${SEED}" ;;
        smote)       TAG="prior_${MODEL}_imbalv3_subjectwise_ratio${RATIO}_s${SEED}" ;;
        undersample) TAG="prior_${MODEL}_undersample_rus_ratio${RATIO}_s${SEED}" ;;
        *)
            echo "[TASK:${LABEL}] ERROR: unknown condition '$CONDITION'"
            return 1
            ;;
    esac

    # Build training command
    local CMD="python scripts/python/train/train.py \
        --model $MODEL --mode pooled --subject_wise_split \
        --seed $SEED --time_stratify_labels --tag $TAG"

    case "$CONDITION" in
        smote_plain) CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO" ;;
        smote)       CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO --subject_wise_oversampling" ;;
        undersample) CMD="$CMD --use_oversampling --oversample_method undersample_rus --target_ratio $RATIO" ;;
    esac

    echo "[TASK:${LABEL}] Train start $(date +%H:%M:%S)"
    eval $CMD
    local EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[TASK:${LABEL}] Eval start $(date +%H:%M:%S)"
        python scripts/python/evaluation/evaluate.py \
            --model $MODEL --tag $TAG --mode pooled \
            --jobid "$JOB_ID" || echo "[TASK:${LABEL}] WARNING: eval failed"
    else
        echo "[TASK:${LABEL}] TRAIN FAILED (exit=$EXIT_CODE)"
    fi

    echo "[TASK:${LABEL}] Done $(date +%H:%M:%S) exit=$EXIT_CODE"
    return $EXIT_CODE
}

# ===== Launch all tasks in parallel =====
PIDS=()
LABELS=()
while IFS='|' read -r MODEL CONDITION SEED RATIO; do
    [[ -z "$MODEL" || "$MODEL" == \#* ]] && continue
    RATIO="${RATIO:-0.5}"
    run_task "$MODEL" "$CONDITION" "$SEED" "$RATIO" &
    PIDS+=($!)
    LABELS+=("${MODEL}:${CONDITION}:s${SEED}")
done < "$TASK_FILE"

echo "[BATCH] Launched ${#PIDS[@]} parallel tasks"

# ===== Wait & collect results =====
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || {
        echo "[BATCH] FAILED: ${LABELS[$i]}"
        ((FAILED++)) || true
    }
done

echo "=============================================="
echo "  Batch Complete"
echo "  Total:  ${#PIDS[@]}"
echo "  Failed: $FAILED"
echo "  END:    $(date)"
echo "=============================================="

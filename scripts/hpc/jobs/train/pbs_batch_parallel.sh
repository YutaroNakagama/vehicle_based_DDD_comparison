#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
# Note: -N, -l select, -l walltime, -q are passed dynamically via qsub options

# ============================================================
# Batch parallel PBS script
# ============================================================
# Runs multiple training tasks in parallel within a single job.
# Each task gets its own CPU allocation (8 CPUs each).
#
# Environment Variables:
#   TASK_FILE : Path to a file listing task specs, one per line.
#               Format: MODEL|CONDITION|MODE|DISTANCE|DOMAIN|RATIO|SEED|N_TRIALS|RANKING|RUN_EVAL|SCRIPT_TYPE
#               SCRIPT_TYPE: "split2" or "unified"
#   PARALLEL  : Number of tasks to run in parallel (default: 4)
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Environment setup
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

# Force CPU for SvmA/SvmW
export CUDA_VISIBLE_DEVICES=""

TASK_FILE="${TASK_FILE:-}"
PARALLEL="${PARALLEL:-4}"

if [[ -z "$TASK_FILE" || ! -f "$TASK_FILE" ]]; then
    echo "[ERROR] TASK_FILE not set or not found: '$TASK_FILE'"
    exit 1
fi

TOTAL_TASKS=$(wc -l < "$TASK_FILE")
echo "============================================================"
echo "[BATCH] Parallel execution: $PARALLEL tasks at a time"
echo "[BATCH] Total tasks: $TOTAL_TASKS"
echo "[BATCH] Task file: $TASK_FILE"
echo "[BATCH] JOBID: $PBS_JOBID"
echo "============================================================"

# ============================================================
# run_task: execute a single training + optional eval
# ============================================================
run_task() {
    local TASK_LINE="$1"
    local TASK_IDX="$2"

    IFS='|' read -r MODEL CONDITION MODE DISTANCE DOMAIN RATIO SEED N_TRIALS RANKING RUN_EVAL SCRIPT_TYPE <<< "$TASK_LINE"

    # Thread isolation per task
    local CPUS_PER_TASK=8
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export BLIS_NUM_THREADS=1
    export TF_NUM_INTRAOP_THREADS=1
    export TF_NUM_INTEROP_THREADS=1
    export TF_CPP_MIN_LOG_LEVEL=2
    export N_TRIALS_OVERRIDE="${N_TRIALS}"

    # Target file
    local TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${DOMAIN}.txt"

    if [[ ! -f "$TARGET_FILE" ]]; then
        echo "[TASK-$TASK_IDX][ERROR] Target file not found: $TARGET_FILE"
        return 1
    fi

    # Generate tag
    local TAG=""
    case "$CONDITION" in
        baseline)
            if [[ "$SCRIPT_TYPE" == "unified" ]]; then
                TAG="prior_${MODEL}_baseline_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_s${SEED}"
            else
                TAG="prior_${MODEL}_baseline_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_s${SEED}"
            fi
            ;;
        smote)
            if [[ "$SCRIPT_TYPE" == "unified" ]]; then
                TAG="prior_${MODEL}_imbalv3_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_subjectwise_ratio${RATIO}_s${SEED}"
            else
                TAG="prior_${MODEL}_imbalv3_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_subjectwise_ratio${RATIO}_s${SEED}"
            fi
            ;;
        smote_plain)
            if [[ "$SCRIPT_TYPE" == "unified" ]]; then
                TAG="prior_${MODEL}_smote_plain_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_ratio${RATIO}_s${SEED}"
            else
                TAG="prior_${MODEL}_smote_plain_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_ratio${RATIO}_s${SEED}"
            fi
            ;;
        undersample)
            if [[ "$SCRIPT_TYPE" == "unified" ]]; then
                TAG="prior_${MODEL}_undersample_rus_${RANKING}_${DISTANCE}_${DOMAIN}_domain_train_split2_ratio${RATIO}_s${SEED}"
            else
                TAG="prior_${MODEL}_undersample_rus_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_split2_ratio${RATIO}_s${SEED}"
            fi
            ;;
        *)
            echo "[TASK-$TASK_IDX][ERROR] Unknown condition: $CONDITION"
            return 1
            ;;
    esac

    echo "[TASK-$TASK_IDX][START] $MODEL | $CONDITION | $MODE | $DISTANCE | $DOMAIN | s$SEED ($(date +%H:%M:%S))"

    # Build training command
    local TRAIN_MODE="$MODE"
    [[ "$SCRIPT_TYPE" == "unified" ]] && TRAIN_MODE="domain_train"

    local CMD="python scripts/python/train/train.py \
        --model $MODEL \
        --mode $TRAIN_MODE \
        --seed $SEED \
        --target_file $TARGET_FILE \
        --tag $TAG \
        --time_stratify_labels"

    case "$CONDITION" in
        baseline) ;;
        smote)       CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO --subject_wise_oversampling" ;;
        smote_plain) CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO" ;;
        undersample) CMD="$CMD --use_oversampling --oversample_method undersample_rus --target_ratio $RATIO" ;;
    esac

    eval $CMD
    local EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "[TASK-$TASK_IDX][FAIL] Training failed (exit=$EXIT_CODE) ($(date +%H:%M:%S))"
        return $EXIT_CODE
    fi

    # Evaluation
    if [[ "$RUN_EVAL" == "true" ]]; then
        if [[ "$SCRIPT_TYPE" == "unified" ]]; then
            # Within-domain eval
            python scripts/python/evaluation/evaluate.py \
                --model "$MODEL" --tag "$TAG" --mode domain_train \
                --target_file "$TARGET_FILE" --eval_type within \
                --jobid "$PBS_JOBID" || echo "[TASK-$TASK_IDX][WARN] Within-domain eval failed"

            # Cross-domain eval
            local CROSS_DOMAIN
            [[ "$DOMAIN" == "in_domain" ]] && CROSS_DOMAIN="out_domain" || CROSS_DOMAIN="in_domain"
            local CROSS_TARGET="results/analysis/exp2_domain_shift/distance/rankings/split2/${RANKING}/${DISTANCE}_${CROSS_DOMAIN}.txt"

            python scripts/python/evaluation/evaluate.py \
                --model "$MODEL" --tag "$TAG" --mode domain_train \
                --target_file "$CROSS_TARGET" --eval_type cross \
                --jobid "$PBS_JOBID" || echo "[TASK-$TASK_IDX][WARN] Cross-domain eval failed"
        else
            # Standard eval
            python scripts/python/evaluation/evaluate.py \
                --model "$MODEL" --tag "$TAG" --mode "$MODE" \
                --jobid "$PBS_JOBID" || echo "[TASK-$TASK_IDX][WARN] Evaluation failed"
        fi
    fi

    echo "[TASK-$TASK_IDX][DONE] $MODEL | $CONDITION | s$SEED ($(date +%H:%M:%S))"
    return 0
}

export -f run_task
export PROJECT_ROOT PBS_JOBID

# ============================================================
# Execute tasks in parallel using background processes + wait
# ============================================================
RUNNING=0
TASK_IDX=0
PIDS=()
FAIL_COUNT=0

while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    ((TASK_IDX++))

    run_task "$line" "$TASK_IDX" &
    PIDS+=($!)
    ((RUNNING++))

    if [[ $RUNNING -ge $PARALLEL ]]; then
        # Wait for any one to finish
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                wait "${PIDS[$i]}" || ((FAIL_COUNT++))
                unset 'PIDS[$i]'
                ((RUNNING--))
            fi
        done
        # If still full, wait for the first to finish
        if [[ $RUNNING -ge $PARALLEL ]]; then
            wait -n || ((FAIL_COUNT++))
            ((RUNNING--))
            # Clean up finished PIDs
            local_pids=()
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    local_pids+=("$pid")
                fi
            done
            PIDS=("${local_pids[@]}")
        fi
    fi
done < "$TASK_FILE"

# Wait for remaining tasks
for pid in "${PIDS[@]}"; do
    wait "$pid" || ((FAIL_COUNT++))
done

echo "============================================================"
echo "[BATCH] All $TASK_IDX tasks completed"
echo "[BATCH] Failures: $FAIL_COUNT"
echo "============================================================"

[[ $FAIL_COUNT -eq 0 ]] && exit 0 || exit 1

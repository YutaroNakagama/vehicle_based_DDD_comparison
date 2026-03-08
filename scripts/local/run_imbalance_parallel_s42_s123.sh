#!/bin/bash
# Imbalance method stand-alone experiment - parallel execution for seed 42, 123

cd "$(dirname "$0")/../.." || exit 1

# Activate venv
source .venv-linux/bin/activate

# Set PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"

LOG_DIR="scripts/local/logs/imbalance"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL=RF
export N_TRIALS_OVERRIDE=50

echo "=========================================="
echo "Imbalance method experiment started: $(date)"
echo "Seeds: 42, 123"
echo "=========================================="

# Common options
# --subject_wise_split: Use subject_time_split strategy
# --time_stratify_labels: Enable time-stratified label splitting
COMMON_OPTS="--model $MODEL --mode pooled --subject_wise_split --time_stratify_labels"

# Define experiment types
declare -A EXP_CONFIGS=(
    ["baseline"]=""
    ["smote_ratio0.1"]="--use_oversampling --oversample_method smote --target_ratio 0.1"
    ["smote_ratio0.5"]="--use_oversampling --oversample_method smote --target_ratio 0.5"
    ["subjectwise_smote_ratio0.1"]="--use_oversampling --oversample_method smote --target_ratio 0.1 --subject_wise_oversampling"
    ["subjectwise_smote_ratio0.5"]="--use_oversampling --oversample_method smote --target_ratio 0.5 --subject_wise_oversampling"
)

SEEDS=(42 123)
PIDS=()
EXPERIMENTS=()

# Launch all experiments in parallel
for seed in "${SEEDS[@]}"; do
    for exp_name in baseline smote_ratio0.1 smote_ratio0.5 subjectwise_smote_ratio0.1 subjectwise_smote_ratio0.5; do
        exp_opts="${EXP_CONFIGS[$exp_name]}"
        tag="${exp_name}_s${seed}"
        log_file="${LOG_DIR}/${tag}_${TIMESTAMP}.log"
        
        echo "[$(date +%H:%M:%S)] Starting: ${tag}"
        
        python scripts/python/train/train.py \
            $COMMON_OPTS \
            --seed "$seed" \
            --tag "$tag" \
            $exp_opts \
            > "$log_file" 2>&1 &
        
        PIDS+=($!)
        EXPERIMENTS+=("$tag")
    done
done

echo ""
echo "=========================================="
echo "Launched all ${#PIDS[@]} experiments in parallel"
echo "PIDs: ${PIDS[*]}"
echo "=========================================="

# Wait for completion
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    exp=${EXPERIMENTS[$i]}
    wait $pid
    status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✓ Completed: $exp"
    else
        echo "[$(date +%H:%M:%S)] ✗ Failed: $exp (exit code: $status)"
        ((FAILED++))
    fi
done

echo ""
echo "=========================================="
echo "All experiments completed: $(date)"
echo "Succeeded: $((${#PIDS[@]} - FAILED)) / ${#PIDS[@]}"
if [ $FAILED -gt 0 ]; then
    echo "Failed: $FAILED"
fi
echo "=========================================="

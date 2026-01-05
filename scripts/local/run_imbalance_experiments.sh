#!/bin/bash
# =============================================================================
# Imbalance Experiments - SMOTE variants & Baseline
# =============================================================================
# Usage:
#   ./scripts/local/run_imbalance_experiments.sh [OPTIONS]
#
# Options:
#   --trials <int>      Optuna trials (default: 10)
#   --seed <int>        Random seed (default: 42)
#   --fg                Run in foreground (default: background)
#
# Experiments (5 total):
#   1. Baseline (no oversampling)
#   2. SMOTE (ratio=0.1)
#   3. SMOTE (ratio=0.5)
#   4. Subject-wise SMOTE (ratio=0.1)
#   5. Subject-wise SMOTE (ratio=0.5)
# =============================================================================

# Check for foreground mode
if [[ "$*" != *"--fg"* ]] && [[ -z "$_IMBALANCE_RUNNING" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    LOG_DIR="$SCRIPT_DIR/logs/imbalance"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/imbalance_experiments_$(date +%Y%m%d_%H%M%S).log"
    
    export _IMBALANCE_RUNNING=1
    nohup "$0" "$@" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "✅ Started in background (PID: $PID)"
    echo "   Log: $LOG_FILE"
    echo ""
    echo "Experiments:"
    echo "   1. Baseline (no oversampling)"
    echo "   2. SMOTE (ratio=0.1)"
    echo "   3. SMOTE (ratio=0.5)"
    echo "   4. Subject-wise SMOTE (ratio=0.1)"
    echo "   5. Subject-wise SMOTE (ratio=0.5)"
    echo ""
    echo "Monitor:"
    echo "   watch \"ps aux | grep train.py | grep -v grep\""
    echo "   tail -f $LOG_FILE"
    echo ""
    echo "Stop:"
    echo "   pkill -f 'train.py.*(baseline|smote)'"
    exit 0
fi

set -e

# Defaults
TRIALS=10
SEED=42
MODEL=RF
RATIO1=0.1
RATIO2=0.5
RUN_EVAL=true

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --trials) TRIALS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --eval) RUN_EVAL=true; shift ;;
        --fg) shift ;;
        *) shift ;;
    esac
done

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$SCRIPT_DIR/logs/imbalance"
mkdir -p "$LOG_DIR"

# Timestamp for log files (consistent across all experiments in this run)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Activate venv
if [[ -f ".venv-linux/bin/activate" ]]; then
    source .venv-linux/bin/activate
    echo "[OK] Activated .venv-linux environment"
elif [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    echo "[OK] Activated .venv environment"
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export N_TRIALS_OVERRIDE="$TRIALS"

echo "========================================"
echo "IMBALANCE EXPERIMENTS"
echo "========================================"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Trials: $TRIALS | Seed: $SEED | Model: $MODEL | Eval: $RUN_EVAL"
echo ""

# Function to run experiment
run_experiment() {
    local name="$1"
    local tag="$2"
    local extra_args="$3"
    local log_file="${LOG_DIR}/${tag}_${TIMESTAMP}.log"
    
    echo "[START] $name - $(date '+%H:%M:%S')"
    
    python scripts/python/train/train.py \
        --model "$MODEL" \
        --mode pooled \
        --subject_wise_split \
        --seed "$SEED" \
        --tag "$tag" \
        $extra_args \
        > "$log_file" 2>&1
    
    local status=$?
    if [[ $status -eq 0 ]]; then
        local best=$(grep -oP "best=\K[0-9.]+" "$log_file" 2>/dev/null | tail -1)
        echo "[DONE]  $name - F2=$best - $(date '+%H:%M:%S')"
    else
        echo "[FAIL]  $name - exit code: $status"
    fi
    return $status
}

# Run experiments in parallel
echo "Launching 5 experiments in parallel..."
echo ""

# 1. Baseline (no oversampling)
run_experiment "Baseline" "baseline_s${SEED}" "" &
PID1=$!

# 2. Standard SMOTE (ratio=0.1)
run_experiment "SMOTE (0.1)" "smote_ratio${RATIO1}_s${SEED}" \
    "--use_oversampling --oversample_method smote --target_ratio $RATIO1" &
PID2=$!

# 3. Standard SMOTE (ratio=0.5)
run_experiment "SMOTE (0.5)" "smote_ratio${RATIO2}_s${SEED}" \
    "--use_oversampling --oversample_method smote --target_ratio $RATIO2" &
PID3=$!

# 4. Subject-wise SMOTE (ratio=0.1)
run_experiment "Subject-wise SMOTE (0.1)" "subjectwise_smote_ratio${RATIO1}_s${SEED}" \
    "--use_oversampling --oversample_method smote --target_ratio $RATIO1 --subject_wise_oversampling" &
PID4=$!

# 5. Subject-wise SMOTE (ratio=0.5)
run_experiment "Subject-wise SMOTE (0.5)" "subjectwise_smote_ratio${RATIO2}_s${SEED}" \
    "--use_oversampling --oversample_method smote --target_ratio $RATIO2 --subject_wise_oversampling" &
PID5=$!

# Wait for all
wait $PID1 $PID2 $PID3 $PID4 $PID5

echo ""
echo "========================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="
for log in ${LOG_DIR}/*_${TIMESTAMP}.log; do
    if [[ -f "$log" ]] && [[ "$log" != *"experiments"* ]] && [[ "$log" != *"_eval_"* ]]; then
        tag=$(basename "$log" .log)
        f2=$(grep -oP "best=\K[0-9.]+" "$log" 2>/dev/null | tail -1 || echo "N/A")
        echo "$tag: F2=$f2"
    fi
done

# ==========================================
# EVALUATION PHASE
# ==========================================
if [[ "$RUN_EVAL" == true ]]; then
    echo ""
    echo "========================================"
    echo "EVALUATION PHASE"
    echo "========================================"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Function to run evaluation
    run_eval() {
        local name="$1"
        local tag="$2"
        local eval_log="${LOG_DIR}/${tag}_eval_${TIMESTAMP}.log"
        
        echo "[EVAL] $name - $(date '+%H:%M:%S')"
        
        python scripts/python/evaluation/evaluate.py \
            --model "$MODEL" \
            --mode pooled \
            --seed "$SEED" \
            --tag "$tag" \
            --subject_wise_split \
            > "$eval_log" 2>&1
        
        local status=$?
        if [[ $status -eq 0 ]]; then
            echo "[DONE] $name evaluation - $(date '+%H:%M:%S')"
        else
            echo "[FAIL] $name evaluation - exit: $status"
        fi
    }
    
    # Run evaluations in parallel
    run_eval "Baseline" "baseline_s${SEED}" &
    EVAL_PID1=$!
    
    run_eval "SMOTE (0.1)" "smote_ratio${RATIO1}_s${SEED}" &
    EVAL_PID2=$!
    
    run_eval "SMOTE (0.5)" "smote_ratio${RATIO2}_s${SEED}" &
    EVAL_PID3=$!
    
    run_eval "Subject-wise SMOTE (0.1)" "subjectwise_smote_ratio${RATIO1}_s${SEED}" &
    EVAL_PID4=$!
    
    run_eval "Subject-wise SMOTE (0.5)" "subjectwise_smote_ratio${RATIO2}_s${SEED}" &
    EVAL_PID5=$!
    
    wait $EVAL_PID1 $EVAL_PID2 $EVAL_PID3 $EVAL_PID4 $EVAL_PID5
    
    echo ""
    echo "========================================"
    echo "EVALUATION COMPLETED!"
    echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
fi

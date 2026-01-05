#!/bin/bash
# =============================================================================
# Domain Analysis: Parallel execution with baseline/SMOTE options
# =============================================================================
# Usage: ./scripts/local/run_domain_parallel.sh [OPTIONS]
#
# Options:
#   --fg              Run in foreground (default: background)
#   --trials N        Number of Optuna trials (default: 10)
#   --ratio R         SMOTE target ratio (default: 0.5, can also use 0.1)
#   --ratio-all       Run with both ratio 0.1 and 0.5
#   --baseline        Run without oversampling (baseline only)
#   --smote           Run with Subject-wise SMOTE only
#   --smote-plain     Run with plain SMOTE only (not subject-wise)
#   --both            Run both baseline and Subject-wise SMOTE experiments (default, 36 total)
#   --all             Run all three: baseline, Subject-wise SMOTE, and plain SMOTE (54 total)
# =============================================================================

# Check for foreground mode - default is background
if [[ "$*" != *"--fg"* ]] && [[ -z "$_DOMAIN_RUNNING" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    LOG_DIR="$SCRIPT_DIR/logs/domain"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/domain_parallel_$(date +%Y%m%d_%H%M%S).log"
    
    export _DOMAIN_RUNNING=1
    nohup "$0" "$@" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "✅ Started in background (PID: $PID)"
    echo "   Log: $LOG_FILE"
    echo ""
    echo "Monitor:"
    echo "   watch \"ps aux | grep train.py | grep -v grep | wc -l\""
    echo "   tail -f $LOG_FILE"
    echo ""
    echo "Stop:"
    echo "   pkill -f 'train.py.*(imbalv3|baseline_domain)'"
    exit 0
fi

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Timestamp for log files (consistent across all experiments in this run)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
TRIALS_ARG=""
RATIO_ARG=""
SEED_ARG=""
RUN_BASELINE=false
RUN_SMOTE=false
RUN_SMOTE_PLAIN=false
RUN_EVAL=true
RUN_RATIO_ALL=false
RUN_SEED_ALL=false
EXPLICIT_SELECTION=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --trials) TRIALS_ARG="$2"; shift 2 ;;
        --ratio) RATIO_ARG="$2"; shift 2 ;;
        --ratio-all) RUN_RATIO_ALL=true; shift ;;
        --seed) SEED_ARG="$2"; shift 2 ;;
        --seed-all) RUN_SEED_ALL=true; shift ;;
        --baseline) RUN_BASELINE=true; EXPLICIT_SELECTION=true; shift ;;
        --smote) RUN_SMOTE=true; EXPLICIT_SELECTION=true; shift ;;
        --smote-plain) RUN_SMOTE_PLAIN=true; EXPLICIT_SELECTION=true; shift ;;
        --both) RUN_BASELINE=true; RUN_SMOTE=true; RUN_SMOTE_PLAIN=false; EXPLICIT_SELECTION=true; shift ;;
        --all) RUN_BASELINE=true; RUN_SMOTE=true; RUN_SMOTE_PLAIN=true; EXPLICIT_SELECTION=true; shift ;;
        --eval) RUN_EVAL=true; shift ;;
        --fg) shift ;;
        *) shift ;;
    esac
done

# Default: run both baseline and SW-SMOTE if no explicit selection
if [[ "$EXPLICIT_SELECTION" == false ]]; then
    RUN_BASELINE=true
    RUN_SMOTE=true
fi

# Activate virtual environment
if [[ -d ".venv-linux" ]]; then
    source .venv-linux/bin/activate
    echo "[OK] Activated .venv-linux environment"
elif [[ -d ".venv" ]]; then
    source .venv/bin/activate
    echo "[OK] Activated .venv environment"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Configuration
export N_TRIALS_OVERRIDE=${TRIALS_ARG:-${N_TRIALS_OVERRIDE:-50}}
RANKING="knn"
# Support multiple seeds
if [[ "$RUN_SEED_ALL" == true ]]; then
    SEEDS=(42 123)
elif [[ -n "$SEED_ARG" ]]; then
    SEEDS=($SEED_ARG)
else
    SEEDS=(42)
fi
# Support multiple ratios
if [[ "$RUN_RATIO_ALL" == true ]]; then
    RATIOS=(0.1 0.5)
elif [[ -n "$RATIO_ARG" ]]; then
    RATIOS=($RATIO_ARG)
else
    RATIOS=(0.5)
fi
TRIALS=${N_TRIALS_OVERRIDE}
MODEL="RF"
LOG_DIR="$SCRIPT_DIR/logs/domain"
MAX_JOBS=18

mkdir -p "$LOG_DIR"

# Calculate total experiments
TOTAL_EXP=0
NUM_RATIOS=${#RATIOS[@]}
NUM_SEEDS=${#SEEDS[@]}
[[ "$RUN_SMOTE" == true ]] && TOTAL_EXP=$((TOTAL_EXP + 18 * NUM_RATIOS * NUM_SEEDS))
[[ "$RUN_SMOTE_PLAIN" == true ]] && TOTAL_EXP=$((TOTAL_EXP + 18 * NUM_RATIOS * NUM_SEEDS))
[[ "$RUN_BASELINE" == true ]] && TOTAL_EXP=$((TOTAL_EXP + 18 * NUM_SEEDS))

echo "========================================"
echo "DOMAIN ANALYSIS - ${TOTAL_EXP} Experiments (PARALLEL)"
echo "========================================"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Trials: ${TRIALS}"
echo "Ratios: ${RATIOS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "SW-SMOTE: ${RUN_SMOTE} | Plain SMOTE: ${RUN_SMOTE_PLAIN} | Baseline: ${RUN_BASELINE} | Eval: ${RUN_EVAL}"
echo "Max parallel jobs: ${MAX_JOBS}"
echo "CPU cores available: $(nproc)"
echo ""

# Function to run SMOTE experiment (subject-wise)
run_smote_experiment() {
    local exp="$1"
    local mode=$(echo "$exp" | cut -d'|' -f1)
    local distance=$(echo "$exp" | cut -d'|' -f2)
    local domain=$(echo "$exp" | cut -d'|' -f3)
    local ratio=$(echo "$exp" | cut -d'|' -f4)
    
    local TAG="imbalv3_${RANKING}_${distance}_${domain}_${mode}_subjectwise_ratio${ratio}_s${SEED}"
    local TARGET_FILE="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING}/${distance}_${domain}.txt"
    local LOG_FILE="${LOG_DIR}/${TAG}_${TIMESTAMP}.log"
    
    echo "[START] SW-SMOTE r${ratio} | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
    
    python scripts/python/train/train.py \
        --model ${MODEL} \
        --mode ${mode} \
        --seed ${SEED} \
        --target_file "${TARGET_FILE}" \
        --tag "${TAG}" \
        --time_stratify_labels \
        --use_oversampling \
        --oversample_method smote \
        --target_ratio ${ratio} \
        --subject_wise_oversampling \
        > "${LOG_FILE}" 2>&1
    
    local status=$?
    if [[ $status -eq 0 ]]; then
        echo "[DONE]  SW-SMOTE r${ratio} | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
    else
        echo "[FAIL]  SW-SMOTE r${ratio} | ${mode} | ${distance} | ${domain} - exit: $status"
    fi
    return $status
}

# Function to run Baseline experiment (no oversampling)
run_baseline_experiment() {
    local exp="$1"
    local mode=$(echo "$exp" | cut -d'|' -f1)
    local distance=$(echo "$exp" | cut -d'|' -f2)
    local domain=$(echo "$exp" | cut -d'|' -f3)
    
    local TAG="baseline_domain_${RANKING}_${distance}_${domain}_${mode}_s${SEED}"
    local TARGET_FILE="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING}/${distance}_${domain}.txt"
    local LOG_FILE="${LOG_DIR}/${TAG}_${TIMESTAMP}.log"
    
    echo "[START] BASE | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
    
    python scripts/python/train/train.py \
        --model ${MODEL} \
        --mode ${mode} \
        --seed ${SEED} \
        --target_file "${TARGET_FILE}" \
        --tag "${TAG}" \
        --time_stratify_labels \
        > "${LOG_FILE}" 2>&1
    
    local status=$?
    if [[ $status -eq 0 ]]; then
        echo "[DONE]  BASE | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
    else
        echo "[FAIL]  BASE | ${mode} | ${distance} | ${domain} - exit: $status"
    fi
    return $status
}

# Function to run plain SMOTE experiment (not subject-wise)
run_smote_plain_experiment() {
    local exp="$1"
    local mode=$(echo "$exp" | cut -d'|' -f1)
    local distance=$(echo "$exp" | cut -d'|' -f2)
    local domain=$(echo "$exp" | cut -d'|' -f3)
    local ratio=$(echo "$exp" | cut -d'|' -f4)
    
    local TAG="smote_plain_${RANKING}_${distance}_${domain}_${mode}_ratio${ratio}_s${SEED}"
    local TARGET_FILE="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING}/${distance}_${domain}.txt"
    local LOG_FILE="${LOG_DIR}/${TAG}_${TIMESTAMP}.log"
    
    echo "[START] SMOTE-P r${ratio} | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
    
    python scripts/python/train/train.py \
        --model ${MODEL} \
        --mode ${mode} \
        --seed ${SEED} \
        --target_file "${TARGET_FILE}" \
        --tag "${TAG}" \
        --time_stratify_labels \
        --use_oversampling \
        --oversample_method smote \
        --target_ratio ${ratio} \
        > "${LOG_FILE}" 2>&1
    
    local status=$?
    if [[ $status -eq 0 ]]; then
        echo "[DONE]  SMOTE-P r${ratio} | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
    else
        echo "[FAIL]  SMOTE-P r${ratio} | ${mode} | ${distance} | ${domain} - exit: $status"
    fi
    return $status
}

export -f run_smote_experiment run_baseline_experiment run_smote_plain_experiment
export RANKING TRIALS MODEL LOG_DIR PROJECT_ROOT PYTHONPATH TIMESTAMP

# Create base experiment list (for baseline - no ratio)
EXPERIMENTS_BASE=""
for mode in source_only target_only; do
  for distance in mmd wasserstein dtw; do
    for domain in in_domain mid_domain out_domain; do
      EXPERIMENTS_BASE="${EXPERIMENTS_BASE}${mode}|${distance}|${domain}\n"
    done
  done
done

# Create SMOTE experiment list with ratios
create_smote_experiments() {
    local ratios_str="$1"
    local experiments=""
    for mode in source_only target_only; do
      for distance in mmd wasserstein dtw; do
        for domain in in_domain mid_domain out_domain; do
          for ratio in $ratios_str; do
            experiments="${experiments}${mode}|${distance}|${domain}|${ratio}\n"
          done
        done
      done
    done
    echo -e "$experiments"
}

RATIOS_STR="${RATIOS[*]}"
EXPERIMENTS_SMOTE=$(create_smote_experiments "$RATIOS_STR")

# Loop through all seeds
for SEED in "${SEEDS[@]}"; do
    export SEED
    echo "========================================"
    echo "Running experiments with SEED=${SEED}"
    echo "========================================"
    echo ""

    # Run Subject-wise SMOTE experiments
    if [[ "$RUN_SMOTE" == true ]]; then
        NUM_SMOTE=$((18 * ${#RATIOS[@]}))
        echo "=== Running Subject-wise SMOTE experiments (${NUM_SMOTE}) with seed=${SEED} ==="
        echo "Ratios: ${RATIOS[*]}"
        echo ""
        if command -v parallel &> /dev/null; then
            echo -e "$EXPERIMENTS_SMOTE" | grep -v '^$' | parallel -j ${MAX_JOBS} --line-buffer run_smote_experiment {}
        else
            echo -e "$EXPERIMENTS_SMOTE" | grep -v '^$' | xargs -P ${MAX_JOBS} -I {} bash -c 'run_smote_experiment "$@"' _ {}
        fi
        echo ""
    fi

    # Run plain SMOTE experiments (not subject-wise)
    if [[ "$RUN_SMOTE_PLAIN" == true ]]; then
        NUM_PLAIN=$((18 * ${#RATIOS[@]}))
        echo "=== Running Plain SMOTE experiments (${NUM_PLAIN}) with seed=${SEED} ==="
        echo "Ratios: ${RATIOS[*]}"
        echo ""
        if command -v parallel &> /dev/null; then
            echo -e "$EXPERIMENTS_SMOTE" | grep -v '^$' | parallel -j ${MAX_JOBS} --line-buffer run_smote_plain_experiment {}
        else
            echo -e "$EXPERIMENTS_SMOTE" | grep -v '^$' | xargs -P ${MAX_JOBS} -I {} bash -c 'run_smote_plain_experiment "$@"' _ {}
        fi
        echo ""
    fi

    # Run Baseline experiments
    if [[ "$RUN_BASELINE" == true ]]; then
        echo "=== Running Baseline experiments (18) with seed=${SEED} ==="
        echo ""
        if command -v parallel &> /dev/null; then
            echo -e "$EXPERIMENTS_BASE" | grep -v '^$' | parallel -j ${MAX_JOBS} --line-buffer run_baseline_experiment {}
        else
            echo -e "$EXPERIMENTS_BASE" | grep -v '^$' | xargs -P ${MAX_JOBS} -I {} bash -c 'run_baseline_experiment "$@"' _ {}
        fi
        echo ""
    fi
done

echo ""
echo "========================================"
echo "ALL ${TOTAL_EXP} EXPERIMENTS COMPLETED!"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Summary
echo ""
echo "=== RESULTS SUMMARY ==="

if [[ "$RUN_SMOTE" == true ]]; then
    echo ""
    echo "--- SMOTE Results ---"
    for log in ${LOG_DIR}/imbalv3_*.log; do
        if [[ -f "$log" ]]; then
            tag=$(basename "$log" .log)
            f2=$(grep -oP "Best.*value: \K[0-9.]+" "$log" 2>/dev/null | tail -1 || echo "N/A")
            echo "${tag}: F2=${f2}"
        fi
    done
fi

if [[ "$RUN_SMOTE_PLAIN" == true ]]; then
    echo ""
    echo "--- Plain SMOTE Results ---"
    for log in ${LOG_DIR}/smote_plain_*.log; do
        if [[ -f "$log" ]]; then
            tag=$(basename "$log" .log)
            f2=$(grep -oP "Best.*value: \K[0-9.]+" "$log" 2>/dev/null | tail -1 || echo "N/A")
            echo "${tag}: F2=${f2}"
        fi
    done
fi

if [[ "$RUN_BASELINE" == true ]]; then
    echo ""
    echo "--- Baseline Results ---"
    for log in ${LOG_DIR}/baseline_domain_*.log; do
        if [[ -f "$log" ]]; then
            tag=$(basename "$log" .log)
            f2=$(grep -oP "Best.*value: \K[0-9.]+" "$log" 2>/dev/null | tail -1 || echo "N/A")
            echo "${tag}: F2=${f2}"
        fi
    done
fi

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
    
    # Function to run evaluation for SMOTE experiment
    run_smote_eval() {
        local exp="$1"
        local mode=$(echo "$exp" | cut -d'|' -f1)
        local distance=$(echo "$exp" | cut -d'|' -f2)
        local domain=$(echo "$exp" | cut -d'|' -f3)
        local ratio=$(echo "$exp" | cut -d'|' -f4)
        
        local TAG="imbalv3_${RANKING}_${distance}_${domain}_${mode}_subjectwise_ratio${ratio}_s${SEED}"
        local TARGET_FILE="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING}/${distance}_${domain}.txt"
        local EVAL_LOG="${LOG_DIR}/${TAG}_eval_${TIMESTAMP}.log"
        
        echo "[EVAL] SW-SMOTE r${ratio} | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
        
        python scripts/python/evaluation/evaluate.py \
            --model ${MODEL} \
            --mode ${mode} \
            --seed ${SEED} \
            --target_file "${TARGET_FILE}" \
            --tag "${TAG}" \
            --subject_wise_split \
            > "${EVAL_LOG}" 2>&1
        
        local status=$?
        if [[ $status -eq 0 ]]; then
            echo "[DONE] SW-SMOTE r${ratio} | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
        else
            echo "[FAIL] SW-SMOTE r${ratio} | ${mode} | ${distance} | ${domain} - exit: $status"
        fi
    }
    
    # Function to run evaluation for Baseline experiment
    run_baseline_eval() {
        local exp="$1"
        local mode=$(echo "$exp" | cut -d'|' -f1)
        local distance=$(echo "$exp" | cut -d'|' -f2)
        local domain=$(echo "$exp" | cut -d'|' -f3)
        
        local TAG="baseline_domain_${RANKING}_${distance}_${domain}_${mode}_s${SEED}"
        local TARGET_FILE="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING}/${distance}_${domain}.txt"
        local EVAL_LOG="${LOG_DIR}/${TAG}_eval_${TIMESTAMP}.log"
        
        echo "[EVAL] BASE | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
        
        python scripts/python/evaluation/evaluate.py \
            --model ${MODEL} \
            --mode ${mode} \
            --seed ${SEED} \
            --target_file "${TARGET_FILE}" \
            --tag "${TAG}" \
            --subject_wise_split \
            > "${EVAL_LOG}" 2>&1
        
        local status=$?
        if [[ $status -eq 0 ]]; then
            echo "[DONE] BASE | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
        else
            echo "[FAIL] BASE | ${mode} | ${distance} | ${domain} - exit: $status"
        fi
    }
    
    # Function to run evaluation for plain SMOTE experiment
    run_smote_plain_eval() {
        local exp="$1"
        local mode=$(echo "$exp" | cut -d'|' -f1)
        local distance=$(echo "$exp" | cut -d'|' -f2)
        local domain=$(echo "$exp" | cut -d'|' -f3)
        local ratio=$(echo "$exp" | cut -d'|' -f4)
        
        local TAG="smote_plain_${RANKING}_${distance}_${domain}_${mode}_ratio${ratio}_s${SEED}"
        local TARGET_FILE="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING}/${distance}_${domain}.txt"
        local EVAL_LOG="${LOG_DIR}/${TAG}_eval_${TIMESTAMP}.log"
        
        echo "[EVAL] PLN r${ratio} | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
        
        python scripts/python/evaluation/evaluate.py \
            --model ${MODEL} \
            --mode ${mode} \
            --seed ${SEED} \
            --target_file "${TARGET_FILE}" \
            --tag "${TAG}" \
            --subject_wise_split \
            > "${EVAL_LOG}" 2>&1
        
        local status=$?
        if [[ $status -eq 0 ]]; then
            echo "[DONE] PLN r${ratio} | ${mode} | ${distance} | ${domain} - $(date '+%H:%M:%S')"
        else
            echo "[FAIL] PLN r${ratio} | ${mode} | ${distance} | ${domain} - exit: $status"
        fi
    }
    
    export -f run_smote_eval run_baseline_eval run_smote_plain_eval
    
    # Loop through all seeds for evaluation
    for SEED in "${SEEDS[@]}"; do
        export SEED
        echo "========================================"
        echo "Evaluating experiments with SEED=${SEED}"
        echo "========================================"
        echo ""

        # Run SMOTE evaluations
        if [[ "$RUN_SMOTE" == true ]]; then
            NUM_SMOTE=$((18 * ${#RATIOS[@]}))
            echo "=== Evaluating Subject-wise SMOTE models (${NUM_SMOTE}) with seed=${SEED} ==="
            if command -v parallel &> /dev/null; then
                echo -e "$EXPERIMENTS_SMOTE" | grep -v '^$' | parallel -j ${MAX_JOBS} --line-buffer run_smote_eval {}
            else
                echo -e "$EXPERIMENTS_SMOTE" | grep -v '^$' | xargs -P ${MAX_JOBS} -I {} bash -c 'run_smote_eval "$@"' _ {}
            fi
            echo ""
        fi
    
        # Run Baseline evaluations
        if [[ "$RUN_BASELINE" == true ]]; then
            echo "=== Evaluating Baseline models (18) with seed=${SEED} ==="
            if command -v parallel &> /dev/null; then
                echo -e "$EXPERIMENTS_BASE" | grep -v '^$' | parallel -j ${MAX_JOBS} --line-buffer run_baseline_eval {}
            else
                echo -e "$EXPERIMENTS_BASE" | grep -v '^$' | xargs -P ${MAX_JOBS} -I {} bash -c 'run_baseline_eval "$@"' _ {}
            fi
            echo ""
        fi
    
        # Run plain SMOTE evaluations
        if [[ "$RUN_SMOTE_PLAIN" == true ]]; then
            NUM_PLAIN=$((18 * ${#RATIOS[@]}))
            echo "=== Evaluating plain SMOTE models (${NUM_PLAIN}) with seed=${SEED} ==="
            if command -v parallel &> /dev/null; then
                echo -e "$EXPERIMENTS_SMOTE" | grep -v '^$' | parallel -j ${MAX_JOBS} --line-buffer run_smote_plain_eval {}
            else
                echo -e "$EXPERIMENTS_SMOTE" | grep -v '^$' | xargs -P ${MAX_JOBS} -I {} bash -c 'run_smote_plain_eval "$@"' _ {}
            fi
            echo ""
        fi
    done
    
    echo "========================================"
    echo "EVALUATION COMPLETED!"
    echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
fi

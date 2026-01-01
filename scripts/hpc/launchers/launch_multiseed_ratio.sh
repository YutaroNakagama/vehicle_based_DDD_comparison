#!/bin/bash
# ============================================================
# Imbalance Comparison V2: Multi-Seed + Multi-Ratio Launcher
# ============================================================
# Run all imbalance experiments with multiple seeds and ratios.
# - Fixed-ratio methods: baseline, balanced_rf, easy_ensemble
# - Variable-ratio methods: smote, smote_tomek, smote_enn, smote_rus, 
#                           smote_balanced_rf, undersample_*
#
# Usage:
#   ./launch_multiseed_ratio.sh                    # Default 3 seeds
#   ./launch_multiseed_ratio.sh 42 123 456 789     # Custom seeds
#   SEEDS="42 123" RATIOS="0.5 1.0" ./launch_multiseed_ratio.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$SCRIPT_DIR"

# Default seeds (3 seeds for statistical robustness)
DEFAULT_SEEDS="42 123 456"
SEEDS="${SEEDS:-${*:-$DEFAULT_SEEDS}}"

# Target ratios for variable-ratio methods
DEFAULT_RATIOS="0.1 0.5 1.0"
RATIOS="${RATIOS:-$DEFAULT_RATIOS}"

# Fixed-ratio methods (ratio不要、内部でバランシング)
FIXED_METHODS="baseline balanced_rf easy_ensemble"

# Variable-ratio methods (target_ratio指定可能)
# SMOTE系 + undersample単体
VARIABLE_METHODS="smote smote_tomek smote_enn smote_rus smote_balanced_rf undersample_rus undersample_tomek undersample_enn"

echo "============================================================"
echo "Imbalance Comparison V2 - Multi-Seed + Multi-Ratio Launcher"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Seeds: $SEEDS"
echo "Ratios (for variable methods): $RATIOS"
echo "Fixed-ratio methods: $FIXED_METHODS"
echo "Variable-ratio methods: $VARIABLE_METHODS"
echo "Time: $(date)"
echo "============================================================"
echo ""

# Track all job IDs
ALL_JOBS=""
JOB_COUNT=0

# Create log file
JOB_LOG="$PROJECT_ROOT/scripts/hpc/logs/imbalance/job_ids_multiseed_ratio_$(date +%Y%m%d_%H%M%S).txt"
echo "# Multi-seed + Multi-ratio job submission: $(date)" > "$JOB_LOG"
echo "# Seeds: $SEEDS" >> "$JOB_LOG"
echo "# Ratios: $RATIOS" >> "$JOB_LOG"
echo "# Format: method,seed,ratio,queue,mem,train_jobid,eval_jobid" >> "$JOB_LOG"

# ============================================================
# Part 1: Fixed-ratio methods (各シードで1回ずつ)
# ============================================================
# ============================================================
# Function: Get optimal queue and resources based on method
# ============================================================
get_queue_and_resources() {
    local method="$1"
    local queue mem walltime ncpus
    
    case "$method" in
        # Heavy methods: SMOTE+ENN, SMOTE+Tomek (need more time and memory)
        smote_enn|smote_tomek)
            queue="SMALL"
            mem="6gb"
            walltime="16:00:00"
            ncpus=4
            ;;
        # Medium methods: SMOTE variants
        smote|smote_rus|smote_balanced_rf)
            queue="SINGLE"
            mem="4gb"
            walltime="08:00:00"
            ncpus=4
            ;;
        # Light methods: Undersample only - use DEFAULT (TINY has 30min limit!)
        undersample_rus|undersample_tomek|undersample_enn)
            queue="DEFAULT"
            mem="4gb"
            walltime="04:00:00"
            ncpus=4
            ;;
        # Ensemble methods
        balanced_rf|easy_ensemble)
            queue="SINGLE"
            mem="4gb"
            walltime="04:00:00"
            ncpus=4
            ;;
        # Baseline (fastest) - use DEFAULT
        baseline)
            queue="DEFAULT"
            mem="4gb"
            walltime="02:00:00"
            ncpus=4
            ;;
        *)
            queue="SINGLE"
            mem="4gb"
            walltime="08:00:00"
            ncpus=4
            ;;
    esac
    
    echo "$queue $mem $walltime $ncpus"
}

echo ""
echo "=== Part 1: Fixed-ratio methods ==="
for SEED in $SEEDS; do
    echo ""
    echo "--- SEED=$SEED ---"
    
    for METHOD in $FIXED_METHODS; do
        SCRIPT="pbs_train_${METHOD}.sh"
        
        if [[ ! -f "$SCRIPT" ]]; then
            echo "  [SKIP] $SCRIPT not found"
            continue
        fi
        
        TAG="imbal_v2_${METHOD}_seed${SEED}"
        
        # Get optimal queue and resources
        read QUEUE MEM WALLTIME NCPUS <<< $(get_queue_and_resources "$METHOD")
        
        # Submit training job with SEED and optimized resources
        TRAIN_JOB=$(qsub -q "$QUEUE" \
            -l select=1:ncpus=${NCPUS}:mem=${MEM} \
            -l walltime=${WALLTIME} \
            -v PBS_O_WORKDIR="$PROJECT_ROOT",SEED="$SEED" "$SCRIPT")
        TRAIN_ID="${TRAIN_JOB%%.*}"
        echo "  [$METHOD] seed=$SEED q=$QUEUE mem=$MEM -> $TRAIN_ID"
        
        # Submit evaluation job with dependency
        EVAL_JOB=$(qsub -W depend=afterok:$TRAIN_JOB \
            -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL=RF,TAG="$TAG",TRAIN_JOBID="$TRAIN_ID",SEED="$SEED" \
            pbs_evaluate.sh)
        EVAL_ID="${EVAL_JOB%%.*}"
        echo "    -> Eval: $EVAL_ID"
        
        ALL_JOBS="$ALL_JOBS $TRAIN_JOB $EVAL_JOB"
        JOB_COUNT=$((JOB_COUNT + 2))
        echo "$METHOD,$SEED,default,$QUEUE,$MEM,$TRAIN_JOB,$EVAL_JOB" >> "$JOB_LOG"
        
        sleep 0.5
    done
done

# ============================================================
# Part 2: Variable-ratio methods (各シード × 各ratio)
# ============================================================
echo ""
echo "=== Part 2: Variable-ratio methods ==="
for SEED in $SEEDS; do
    echo ""
    echo "--- SEED=$SEED ---"
    
    for RATIO in $RATIOS; do
        echo "  -- ratio=$RATIO --"
        
        for METHOD in $VARIABLE_METHODS; do
            # Determine model type
            if [[ "$METHOD" == "smote_balanced_rf" ]]; then
                MODEL="BalancedRF"
            else
                MODEL="RF"
            fi
            
            TAG="imbal_v2_${METHOD}_ratio${RATIO//./_}_seed${SEED}"
            
            # Get optimal queue and resources
            read QUEUE MEM WALLTIME NCPUS <<< $(get_queue_and_resources "$METHOD")
            
            # Submit training job using generic ratio script with optimized resources
            TRAIN_JOB=$(qsub -q "$QUEUE" \
                -l select=1:ncpus=${NCPUS}:mem=${MEM} \
                -l walltime=${WALLTIME} \
                -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL="$MODEL",RATIO="$RATIO",METHOD="$METHOD",TAG="$TAG",SEED="$SEED" \
                pbs_train_generic_ratio.sh)
            TRAIN_ID="${TRAIN_JOB%%.*}"
            echo "    [$METHOD] ratio=$RATIO q=$QUEUE mem=$MEM -> $TRAIN_ID"
            
            # Submit evaluation job with dependency
            EVAL_JOB=$(qsub -W depend=afterok:$TRAIN_JOB \
                -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL="$MODEL",TAG="$TAG",TRAIN_JOBID="$TRAIN_ID",SEED="$SEED" \
                pbs_evaluate.sh)
            EVAL_ID="${EVAL_JOB%%.*}"
            echo "      -> Eval: $EVAL_ID"
            
            ALL_JOBS="$ALL_JOBS $TRAIN_JOB $EVAL_JOB"
            JOB_COUNT=$((JOB_COUNT + 2))
            echo "$METHOD,$SEED,$RATIO,$QUEUE,$MEM,$TRAIN_JOB,$EVAL_JOB" >> "$JOB_LOG"
            
            sleep 0.5
        done
    done
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "============================================================"
echo "Total jobs: $JOB_COUNT"
echo ""
echo "Breakdown:"
echo "  Fixed-ratio methods: $(echo $FIXED_METHODS | wc -w) methods × $(echo $SEEDS | wc -w) seeds × 2 (train+eval)"
echo "  Variable-ratio methods: $(echo $VARIABLE_METHODS | wc -w) methods × $(echo $SEEDS | wc -w) seeds × $(echo $RATIOS | wc -w) ratios × 2 (train+eval)"
echo ""
echo "Monitor with: qstat -u \$USER"
echo "Job IDs saved to: $JOB_LOG"
echo "============================================================"

# Display job summary
echo ""
echo "=== Job Summary ==="
cat "$JOB_LOG" | tail -30

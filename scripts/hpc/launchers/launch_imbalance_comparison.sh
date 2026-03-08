#!/bin/bash
# ============================================================
# Imbalance Comparison Launcher (HPC)
# ============================================================
# Replicates run_imbalance_experiments.sh for HPC environment
# Submits 5 parallel jobs (one per experiment)
#
# Usage:
#   ./launch_imbalance_comparison.sh              # seed=42, trials=50
#   ./launch_imbalance_comparison.sh --seed 123   # specific seed
#   ./launch_imbalance_comparison.sh --seeds "42 123"  # multiple seeds
#   ./launch_imbalance_comparison.sh --trials 100 # custom trials
#   ./launch_imbalance_comparison.sh --dry-run    # preview only
#   ./launch_imbalance_comparison.sh --no-eval    # skip evaluation
#
# Experiments (5 per seed):
#   1. Baseline (no oversampling)
#   2. SMOTE (ratio=0.1)
#   3. SMOTE (ratio=0.5)
#   4. Subject-wise SMOTE (ratio=0.1)
#   5. Subject-wise SMOTE (ratio=0.5)
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/imbalance/pbs_imbalance_comparison.sh"

# Defaults
SEEDS="42 123"  # For paper: use multiple seeds for statistical reliability
TRIALS=100  # For paper: 100 trials for statistical significance
DRY_RUN=false
RUN_EVAL=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEEDS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-eval)
            RUN_EVAL=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Define experiments: METHOD|RATIO
# 8 experiments total:
#   - Baseline (no sampling)
#   - Oversampling: SMOTE (0.1, 0.5), Subject-wise SMOTE (0.1, 0.5)
#   - Undersampling: Random Under-Sampler (0.1, 0.5)
#   - Model-based: Balanced Random Forest
EXPERIMENTS=(
    "baseline|0"
    "smote|0.1"
    "smote|0.5"
    "smote_subjectwise|0.1"
    "smote_subjectwise|0.5"
    "undersample_rus|0.1"
    "undersample_rus|0.5"
    "balanced_rf|0"
)

# Resource configuration (optimized for each method)
get_resources() {
    local method="$1"
    case "$method" in
        baseline)
            echo "ncpus=4:mem=8gb 04:00:00 SINGLE"
            ;;
        smote|smote_subjectwise)
            echo "ncpus=4:mem=8gb 06:00:00 SINGLE"
            ;;
        undersample_rus)
            echo "ncpus=4:mem=8gb 03:00:00 SINGLE"
            ;;
        balanced_rf)
            echo "ncpus=8:mem=8gb 06:00:00 DEFAULT"
            ;;
        *)
            echo "ncpus=4:mem=8gb 06:00:00 SINGLE"
            ;;
    esac
}

# Setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/imbalance"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_comparison_${TIMESTAMP}.txt"

# Count jobs
NUM_SEEDS=$(echo "$SEEDS" | wc -w)
TOTAL_JOBS=$((${#EXPERIMENTS[@]} * NUM_SEEDS))

echo "============================================================"
echo "Imbalance Comparison Launcher (HPC)"
echo "============================================================"
echo "Seeds: $SEEDS"
echo "Trials: $TRIALS"
echo "Eval: $RUN_EVAL"
echo "Total Jobs: $TOTAL_JOBS"
echo "Dry run: $DRY_RUN"
echo "Log: $LOG_FILE"
echo "============================================================"
echo ""
echo "Experiments:"
for exp in "${EXPERIMENTS[@]}"; do
    method=$(echo "$exp" | cut -d'|' -f1)
    ratio=$(echo "$exp" | cut -d'|' -f2)
    if [[ "$method" == "baseline" || "$method" == "balanced_rf" ]]; then
        echo "  - $method"
    else
        echo "  - $method (ratio=$ratio)"
    fi
done
echo ""

echo "# Launched at $(date)" > "$LOG_FILE"

JOB_COUNT=0
for SEED in $SEEDS; do
    echo "=== Seed: $SEED ==="
    
    for exp in "${EXPERIMENTS[@]}"; do
        METHOD=$(echo "$exp" | cut -d'|' -f1)
        RATIO=$(echo "$exp" | cut -d'|' -f2)
        
        RESOURCES=$(get_resources "$METHOD")
        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
        
        # Short job name
        case "$METHOD" in
            baseline)
                JOB_NAME="bl_s${SEED}"
                ;;
            smote)
                JOB_NAME="sm${RATIO/./}_s${SEED}"
                ;;
            smote_subjectwise)
                JOB_NAME="sw${RATIO/./}_s${SEED}"
                ;;
            undersample_rus)
                JOB_NAME="rus${RATIO/./}_s${SEED}"
                ;;
            balanced_rf)
                JOB_NAME="brf_s${SEED}"
                ;;
        esac
        
        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
        CMD="$CMD -v METHOD=$METHOD,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$TRIALS,RUN_EVAL=$RUN_EVAL"
        CMD="$CMD $JOB_SCRIPT"
        
        if $DRY_RUN; then
            echo "[DRY-RUN] $METHOD ratio=$RATIO -> $JOB_NAME"
        else
            JOB_ID=$(eval "$CMD" 2>&1)
            echo "[$METHOD] ratio=$RATIO -> $JOB_ID"
            echo "$METHOD:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
            JOB_COUNT=$((JOB_COUNT + 1))
            sleep 0.2
        fi
    done
    echo ""
done

echo "============================================================"
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
else
    echo "Submitted $JOB_COUNT jobs"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Monitor:"
    echo "  qstat -u \$USER"
    echo "  watch 'qstat -u \$USER | grep -E \"(bl_|sm|sw)\"'"
fi
echo "============================================================"

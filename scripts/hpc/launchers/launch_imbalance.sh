#!/bin/bash
# ============================================================
# Unified Imbalance Experiment Launcher
# ============================================================
# Launch training jobs using the unified pbs_train.sh script
#
# Usage:
#   ./launch_imbalance.sh                    # All methods, seed 42
#   ./launch_imbalance.sh --seeds "42 123"   # Multiple seeds
#   ./launch_imbalance.sh --methods "smote smote_tomek"  # Specific methods
#   ./launch_imbalance.sh --dry-run          # Show commands without executing
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/imbalance/pbs_train.sh"

# Default parameters
SEEDS="42"
METHODS="baseline smote smote_subjectwise smote_tomek smote_enn smote_rus balanced_rf easy_ensemble smote_balanced_rf"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Resource configurations per method
get_resources() {
    local method="$1"
    case "$method" in
        balanced_rf|easy_ensemble|smote_balanced_rf|smote_enn)
            echo "ncpus=8:mem=8gb 12:00:00 DEFAULT"
            ;;
        smote_subjectwise)
            echo "ncpus=4:mem=6gb 12:00:00 SINGLE"
            ;;
        baseline|smote_rus)
            echo "ncpus=4:mem=8gb 08:00:00 SINGLE"
            ;;
        *)
            echo "ncpus=4:mem=8gb 10:00:00 SINGLE"
            ;;
    esac
}

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/imbalance"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_${TIMESTAMP}.txt"

echo "============================================================"
echo "Imbalance Experiment Launcher"
echo "============================================================"
echo "Methods: $METHODS"
echo "Seeds: $SEEDS"
echo "Dry run: $DRY_RUN"
echo "Log: $LOG_FILE"
echo "============================================================"
echo ""

echo "# Launched at $(date)" > "$LOG_FILE"

JOB_COUNT=0
for METHOD in $METHODS; do
    for SEED in $SEEDS; do
        RESOURCES=$(get_resources "$METHOD")
        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
        
        JOB_NAME="${METHOD:0:6}_s${SEED}"
        
        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v METHOD=$METHOD,SEED=$SEED $JOB_SCRIPT"
        
        if $DRY_RUN; then
            echo "[DRY-RUN] $CMD"
        else
            JOB_ID=$(eval "$CMD" 2>&1)
            echo "[$METHOD] seed=$SEED -> $JOB_ID"
            echo "$METHOD:$SEED:$JOB_ID" >> "$LOG_FILE"
            ((JOB_COUNT++))
            sleep 0.3
        fi
    done
done

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
else
    echo "Submitted $JOB_COUNT jobs"
    echo "Log file: $LOG_FILE"
fi
echo "============================================================"

#!/bin/bash
# ============================================================
# Class imbalance experiment launcher for paper
# ============================================================
# Experiment conditions:
#   - seeds: 42, 123
#   - Target ratio: 0.1, 0.5
#   - Models: RF (BalancedRF is included as a method)
#   - Imbalance methods: Baseline, Plain SMOTE, Subject-wise SMOTE, RUS, Balanced RF
#   - Optuna trials: 100
#   - Optuna objective: F2 (already implemented)
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/imbalance/pbs_imbalance_comparison.sh"

# Paper settings
SEEDS="42 123"
RATIOS="0.1 0.5"
N_TRIALS=100

# Experiment conditions (for paper)
EXPERIMENTS=(
    "baseline"           # Baseline (no oversampling)
    "smote"              # Plain SMOTE
    "smote_subjectwise"  # Subject-wise SMOTE
    "undersample_rus"    # Random Undersampling
    "balanced_rf"        # BalancedRandomForest
)

# Parse arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
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

# Resource configurations (Optimize based on queue status)
# TINY: not used due to max 30min limit
get_resources() {
    local method="$1"
    case "$method" in
        balanced_rf)
            # BalancedRF: 8cores required, use LONG queue (may run long)
            echo "ncpus=8:mem=8gb 08:00:00 LONG"
            ;;
        smote|smote_subjectwise)
            # SMOTE-family: 4 cores, use SINGLE queue
            echo "ncpus=4:mem=8gb 08:00:00 SINGLE"
            ;;
        baseline|undersample_rus)
            # lightweight experiment: 4cores, use SINGLE queue (LONG has small capacity)
            echo "ncpus=4:mem=8gb 04:00:00 SINGLE"
            ;;
        *)
            # default: SINGLE queue
            echo "ncpus=4:mem=8gb 06:00:00 SINGLE"
            ;;
    esac
}

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/imbalance"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_paper_${TIMESTAMP}.txt"

echo "============================================================"
echo "Class imbalance experiment launcher for paper"
echo "============================================================"
echo "experimentmethod: ${EXPERIMENTS[*]}"
echo "seeds: $SEEDS"
echo "ratio: $RATIOS"
echo "Optuna trials: $N_TRIALS"
echo "Dry run: $DRY_RUN"
echo "Log: $LOG_FILE"
echo ""
echo "Queue status:"
qstat -Q | grep -E "Queue|TINY|SINGLE|DEFAULT|SMALL"
echo "============================================================"
echo ""

echo "# Launched at $(date)" > "$LOG_FILE"

JOB_COUNT=0
TOTAL_JOBS=0

# Calculate total job count
for METHOD in "${EXPERIMENTS[@]}"; do
    for SEED in $SEEDS; do
        if [[ "$METHOD" == "baseline" || "$METHOD" == "balanced_rf" ]]; then
            # Baseline Baseline and BalancedRF do not need ratio
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
        else
            # Run SMOTE-family and RUS for each ratio
            for RATIO in $RATIOS; do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
            done
        fi
    done
done

echo "Total $TOTAL_JOBS jobs to submit"
echo ""

# job(s)submit
for METHOD in "${EXPERIMENTS[@]}"; do
    for SEED in $SEEDS; do
        if [[ "$METHOD" == "baseline" || "$METHOD" == "balanced_rf" ]]; then
            # Baseline Baseline and BalancedRF do not need ratio
            RESOURCES=$(get_resources "$METHOD")
            NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
            WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
            QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
            
            JOB_NAME="${METHOD:0:6}_s${SEED}"
            
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v METHOD=$METHOD,SEED=$SEED,N_TRIALS=$N_TRIALS $JOB_SCRIPT"
            
            if $DRY_RUN; then
                echo "[DRY-RUN] $CMD"
            else
                echo "[$((JOB_COUNT + 1))/$TOTAL_JOBS] Submitting: $METHOD (seed=$SEED)"
                JOBID=$(eval $CMD)
                echo "$JOBID | $METHOD | seed=$SEED | $QUEUE" >> "$LOG_FILE"
                echo "  -> JobID: $JOBID (Queue: $QUEUE)"
            fi
            JOB_COUNT=$((JOB_COUNT + 1))
        else
            # Run SMOTE-family and RUS for each ratio
            for RATIO in $RATIOS; do
                RESOURCES=$(get_resources "$METHOD")
                NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                
                JOB_NAME="${METHOD:0:6}_r${RATIO}_s${SEED}"
                
                CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v METHOD=$METHOD,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS $JOB_SCRIPT"
                
                if $DRY_RUN; then
                    echo "[DRY-RUN] $CMD"
                else
                    echo "[$((JOB_COUNT + 1))/$TOTAL_JOBS] Submitting: $METHOD (ratio=$RATIO, seed=$SEED)"
                    JOBID=$(eval $CMD)
                    echo "$JOBID | $METHOD | ratio=$RATIO | seed=$SEED | $QUEUE" >> "$LOG_FILE"
                    echo "  -> JobID: $JOBID (Queue: $QUEUE)"
                fi
                JOB_COUNT=$((JOB_COUNT + 1))
            done
        fi
    done
done

echo ""
echo "============================================================"
echo "Total $JOB_COUNT jobs submitted"
if ! $DRY_RUN; then
    echo "Log: $LOG_FILE"
    echo ""
    echo "Job status check:"
    echo "  qstat -u s2240011"
    echo ""
    echo "Check specific job logs:"
    echo "  tail -f $LOG_DIR/\${PBS_JOBID}.o*"
fi
echo "============================================================"

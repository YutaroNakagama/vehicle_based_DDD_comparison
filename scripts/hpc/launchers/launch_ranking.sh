#!/bin/bash
# ============================================================
# Ranking-based Training Launcher
# ============================================================
# Launch ranking experiments using pbs_train_ranking.sh
#
# Usage:
#   ./launch_ranking.sh                          # All combinations
#   ./launch_ranking.sh --methods "smote"        # Specific methods
#   ./launch_ranking.sh --rankings "knn"         # Specific rankings
#   ./launch_ranking.sh --dry-run                # Preview commands
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/imbalance/pbs_train_ranking.sh"
RANKING_BASE="$PROJECT_ROOT/results/analysis/domain/distance/subject-wise"

# Default parameters
METHODS="smote smote_subjectwise smote_balanced_rf"
RANKINGS="knn lof"
MODES="source_only target_only"
LEVELS="out_domain in_domain"
METRIC="mmd"
SEED=42
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --rankings)
            RANKINGS="$2"
            shift 2
            ;;
        --modes)
            MODES="$2"
            shift 2
            ;;
        --levels)
            LEVELS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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
        smote_balanced_rf)
            echo "ncpus=8:mem=8gb 12:00:00 DEFAULT"
            ;;
        smote_subjectwise)
            echo "ncpus=4:mem=6gb 12:00:00 SINGLE"
            ;;
        *)
            echo "ncpus=4:mem=8gb 10:00:00 SINGLE"
            ;;
    esac
}

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/ranking"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_${TIMESTAMP}.txt"

echo "============================================================"
echo "Ranking-based Training Launcher"
echo "============================================================"
echo "Methods: $METHODS"
echo "Rankings: $RANKINGS"
echo "Modes: $MODES"
echo "Levels: $LEVELS"
echo "Dry run: $DRY_RUN"
echo "============================================================"
echo ""

echo "# Launched at $(date)" > "$LOG_FILE"

JOB_COUNT=0
SKIP_COUNT=0

for METHOD in $METHODS; do
    for RANKING in $RANKINGS; do
        for MODE in $MODES; do
            for LEVEL in $LEVELS; do
                SUBJECT_FILE="$RANKING_BASE/$METRIC/groups/clustering_ranked/${METRIC}_${RANKING}_${LEVEL}.txt"
                
                if [[ ! -f "$SUBJECT_FILE" ]]; then
                    echo "[SKIP] Subject file not found: $SUBJECT_FILE"
                    ((SKIP_COUNT++))
                    continue
                fi
                
                TAG="rank_${RANKING}_${METRIC}_${LEVEL}_${METHOD}"
                
                RESOURCES=$(get_resources "$METHOD")
                NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                
                JOB_NAME="${METHOD:0:4}_${RANKING:0:3}_${LEVEL:0:3}"
                
                CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                CMD="$CMD -v METHOD=$METHOD,MODE=$MODE,SUBJECT_FILE=$SUBJECT_FILE,TAG=$TAG,SEED=$SEED"
                CMD="$CMD $JOB_SCRIPT"
                
                if $DRY_RUN; then
                    echo "[DRY-RUN] $METHOD/$RANKING/$LEVEL/$MODE"
                else
                    JOB_ID=$(eval "$CMD" 2>&1)
                    echo "[$METHOD] $RANKING/$LEVEL/$MODE -> $JOB_ID"
                    echo "$METHOD:$RANKING:$LEVEL:$MODE:$JOB_ID" >> "$LOG_FILE"
                    ((JOB_COUNT++))
                    sleep 0.3
                fi
            done
        done
    done
done

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
else
    echo "Submitted: $JOB_COUNT jobs"
    echo "Skipped: $SKIP_COUNT (missing subject files)"
    echo "Log: $LOG_FILE"
fi
echo "============================================================"

#!/bin/bash
# ============================================================
# Domain Analysis Training Launcher
# ============================================================
# Launch domain analysis experiments using unified pbs_train.sh
#
# Supports two experiment types:
#   - ranking: Compare ranking methods (which ranking is best?)
#   - smote: Compare SMOTE methods with ranking-based subject selection
#
# Usage:
#   ./launch_domain.sh ranking                   # Ranking comparison
#   ./launch_domain.sh smote                     # SMOTE comparison
#   ./launch_domain.sh smote --methods "smote"   # Specific SMOTE methods
#   ./launch_domain.sh --dry-run ranking         # Preview commands
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_train.sh"
RANKING_BASE="$PROJECT_ROOT/results/analysis/domain/distance/subject-wise"

# Default parameters
EXPERIMENT_TYPE=""
METHODS="smote smote_subjectwise smote_balanced_rf"
RANKINGS="knn lof"
MODES="source_only target_only"
LEVELS="out_domain in_domain"
METRICS="mmd"
SEED=42
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        ranking|smote)
            EXPERIMENT_TYPE="$1"
            shift
            ;;
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
        --metrics)
            METRICS="$2"
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
            echo "Usage: $0 [ranking|smote] [options]"
            exit 1
            ;;
    esac
done

if [[ -z "$EXPERIMENT_TYPE" ]]; then
    echo "Usage: $0 [ranking|smote] [options]"
    echo ""
    echo "Experiment types:"
    echo "  ranking  - Compare ranking methods"
    echo "  smote    - Compare SMOTE methods with ranking-based subjects"
    echo ""
    echo "Options:"
    echo "  --methods    SMOTE methods (smote comparison only)"
    echo "  --rankings   Ranking methods (knn lof ...)"
    echo "  --modes      Training modes (source_only target_only)"
    echo "  --levels     Domain levels (out_domain in_domain)"
    echo "  --metrics    Distance metrics (mmd dtw wasserstein)"
    echo "  --seed       Random seed"
    echo "  --dry-run    Preview without submitting"
    exit 1
fi

# Resource configurations
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
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_domain_${TIMESTAMP}.txt"

echo "============================================================"
echo "Domain Analysis Launcher - ${EXPERIMENT_TYPE^^}"
echo "============================================================"
echo "Dry run: $DRY_RUN"
echo "Log: $LOG_FILE"
echo "============================================================"
echo ""

echo "# Launched at $(date)" > "$LOG_FILE"

JOB_COUNT=0
SKIP_COUNT=0

case "$EXPERIMENT_TYPE" in
    ranking)
        echo "=== Ranking Comparison ==="
        echo "Rankings: $RANKINGS"
        echo "Metrics: $METRICS"
        echo "Levels: $LEVELS"
        echo "Modes: $MODES"
        echo ""
        
        for RANKING in $RANKINGS; do
            for METRIC in $METRICS; do
                for LEVEL in $LEVELS; do
                    for MODE in $MODES; do
                        RESOURCES=$(get_resources "rf")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                        
                        JOB_NAME="r_${RANKING:0:3}_${LEVEL:0:3}"
                        
                        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                        CMD="$CMD -v EXPERIMENT=ranking_comparison,RANKING_METHOD=$RANKING,DISTANCE_METRIC=$METRIC,DOMAIN_LEVEL=$LEVEL,MODE=$MODE,SEED=$SEED"
                        CMD="$CMD $JOB_SCRIPT"
                        
                        if $DRY_RUN; then
                            echo "[DRY-RUN] $RANKING/$METRIC/$LEVEL/$MODE"
                        else
                            JOB_ID=$(eval "$CMD" 2>&1)
                            echo "[$RANKING] $METRIC/$LEVEL/$MODE -> $JOB_ID"
                            echo "ranking:$RANKING:$METRIC:$LEVEL:$MODE:$JOB_ID" >> "$LOG_FILE"
                            ((JOB_COUNT++))
                            sleep 0.3
                        fi
                    done
                done
            done
        done
        ;;
        
    smote)
        echo "=== SMOTE Comparison ==="
        echo "Methods: $METHODS"
        echo "Rankings: $RANKINGS"
        echo "Levels: $LEVELS"
        echo "Modes: $MODES"
        echo ""
        
        for METHOD in $METHODS; do
            for RANKING in $RANKINGS; do
                for LEVEL in $LEVELS; do
                    for MODE in $MODES; do
                        SUBJECT_FILE="$RANKING_BASE/mmd/groups/clustering_ranked/mmd_${RANKING}_${LEVEL}.txt"
                        
                        if [[ ! -f "$SUBJECT_FILE" ]]; then
                            echo "[SKIP] Subject file not found: $SUBJECT_FILE"
                            ((SKIP_COUNT++))
                            continue
                        fi
                        
                        TAG="smote_rank_${RANKING}_${LEVEL}_${METHOD}"
                        
                        RESOURCES=$(get_resources "$METHOD")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                        
                        JOB_NAME="${METHOD:0:4}_${RANKING:0:3}_${LEVEL:0:3}"
                        
                        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                        CMD="$CMD -v EXPERIMENT=smote_comparison,METHOD=$METHOD,MODE=$MODE,SUBJECT_FILE=$SUBJECT_FILE,TAG=$TAG,SEED=$SEED"
                        CMD="$CMD $JOB_SCRIPT"
                        
                        if $DRY_RUN; then
                            echo "[DRY-RUN] $METHOD/$RANKING/$LEVEL/$MODE"
                        else
                            JOB_ID=$(eval "$CMD" 2>&1)
                            echo "[$METHOD] $RANKING/$LEVEL/$MODE -> $JOB_ID"
                            echo "smote:$METHOD:$RANKING:$LEVEL:$MODE:$JOB_ID" >> "$LOG_FILE"
                            ((JOB_COUNT++))
                            sleep 0.3
                        fi
                    done
                done
            done
        done
        ;;
esac

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
else
    echo "Submitted: $JOB_COUNT jobs"
    echo "Skipped: $SKIP_COUNT (missing files)"
    echo "Log: $LOG_FILE"
fi
echo "============================================================"

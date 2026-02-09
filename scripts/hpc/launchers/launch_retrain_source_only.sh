#!/bin/bash
# ============================================================
# Re-train + re-evaluate source_only split2 jobs
# ============================================================
# source_only models were trained with Bug #2 (Dir A ≠ Dir B for
# source subjects), so they need full re-training + re-evaluation.
#
# This launcher re-runs training (which includes eval via RUN_EVAL=true)
# for source_only mode across all conditions.
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODE="source_only"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_retrain_source_only_${TIMESTAMP}.log"

get_resources() {
    local condition="$1"
    case "$condition" in
        balanced_rf)  echo "ncpus=8:mem=12gb 08:00:00 LONG" ;;
        smote|smote_plain) echo "ncpus=4:mem=10gb 08:00:00 SINGLE" ;;
        *)            echo "ncpus=4:mem=8gb 06:00:00 SINGLE" ;;
    esac
}

echo "============================================================"
echo "Re-train source_only split2 (fix Bug #2: Dir A ≠ Dir B)"
echo "============================================================"
echo "Mode: source_only only (re-train + re-eval)"
echo "Distances: ${DISTANCES[*]}"
echo "Domains: ${DOMAINS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Dry run: $DRY_RUN"
echo "============================================================"
echo ""

JOB_COUNT=0

for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            # Baseline
            CONDITION="baseline"
            RESOURCES=$(get_resources "$CONDITION")
            NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
            WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
            QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
            JOB_NAME="fx_bs_${DISTANCE:0:2}${DOMAIN:0:1}_s${SEED}"

            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
            CMD="$CMD -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
            CMD="$CMD $JOB_SCRIPT"

            if $DRY_RUN; then
                echo "[DRY-RUN] baseline | $DISTANCE | $DOMAIN | source_only | s$SEED"
            else
                JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] $CMD"; continue; }
                echo "[SUBMIT] baseline | $DISTANCE | $DOMAIN | source_only | s$SEED → $JOB_ID"
                echo "baseline:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                ((JOB_COUNT++))
                sleep 0.2
            fi

            # Ratio-based conditions
            for RATIO in "${RATIOS[@]}"; do
                for COND in smote_plain smote undersample; do
                    RESOURCES=$(get_resources "$COND")
                    NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                    WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                    QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                    JOB_NAME="fx_${COND:0:2}_${DISTANCE:0:2}${DOMAIN:0:1}_r${RATIO}_s${SEED}"

                    CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                    CMD="$CMD -v CONDITION=$COND,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                    CMD="$CMD $JOB_SCRIPT"

                    if $DRY_RUN; then
                        echo "[DRY-RUN] $COND | $DISTANCE | $DOMAIN | source_only | r=$RATIO | s$SEED"
                    else
                        JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] $CMD"; continue; }
                        echo "[SUBMIT] $COND | $DISTANCE | $DOMAIN | source_only | r=$RATIO | s$SEED → $JOB_ID"
                        echo "$COND:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
                        ((JOB_COUNT++))
                        sleep 0.2
                    fi
                done
            done

            # Balanced RF
            CONDITION="balanced_rf"
            RESOURCES=$(get_resources "$CONDITION")
            NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
            WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
            QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
            JOB_NAME="fx_bf_${DISTANCE:0:2}${DOMAIN:0:1}_s${SEED}"

            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
            CMD="$CMD -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
            CMD="$CMD $JOB_SCRIPT"

            if $DRY_RUN; then
                echo "[DRY-RUN] balanced_rf | $DISTANCE | $DOMAIN | source_only | s$SEED"
            else
                JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] $CMD"; continue; }
                echo "[SUBMIT] balanced_rf | $DISTANCE | $DOMAIN | source_only | s$SEED → $JOB_ID"
                echo "balanced_rf:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                ((JOB_COUNT++))
                sleep 0.2
            fi
        done
    done
done

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "Dry run: $JOB_COUNT jobs would be submitted"
else
    echo "Submitted: $JOB_COUNT jobs"
    echo "Log: $LOG_FILE"
fi
echo "============================================================"

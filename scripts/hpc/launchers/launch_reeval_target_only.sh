#!/bin/bash
# ============================================================
# Re-evaluate target_only split2 jobs (fix: add --target_file)
# ============================================================
# target_only models were trained correctly (using Dir A subjects only).
# Only evaluation was broken (missing --target_file → random split).
# This launcher re-runs evaluation only for all target_only conditions.
#
# source_only and mixed models need re-training (Bug #2: Dir A ≠ Dir B).
# Those will be handled by the full re-train launcher.
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_reeval_split2.sh"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODE="target_only"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_reeval_target_only_${TIMESTAMP}.log"

echo "============================================================"
echo "Re-evaluate target_only split2 (fix --target_file bug)"
echo "============================================================"
echo "Mode: target_only only (re-eval, no re-training needed)"
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
            # Baseline (no ratio)
            JOB_NAME="re_bs_${DISTANCE:0:2}${DOMAIN:0:1}_t_s${SEED}"
            CMD="qsub -N $JOB_NAME -l select=1:ncpus=2:mem=8gb -l walltime=00:30:00 -q SINGLE"
            CMD="$CMD -v CONDITION=baseline,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,RANKING=$RANKING"
            CMD="$CMD $JOB_SCRIPT"

            if $DRY_RUN; then
                echo "[DRY-RUN] baseline | $DISTANCE | $DOMAIN | target_only | s$SEED"
            else
                JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] $CMD"; continue; }
                echo "[SUBMIT] baseline | $DISTANCE | $DOMAIN | target_only | s$SEED → $JOB_ID"
                echo "baseline:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                ((JOB_COUNT++))
                sleep 0.1
            fi

            # Ratio-based conditions
            for RATIO in "${RATIOS[@]}"; do
                for CONDITION in smote_plain smote undersample; do
                    JOB_NAME="re_${CONDITION:0:2}_${DISTANCE:0:2}${DOMAIN:0:1}_t_r${RATIO}_s${SEED}"
                    CMD="qsub -N $JOB_NAME -l select=1:ncpus=2:mem=8gb -l walltime=00:30:00 -q SINGLE"
                    CMD="$CMD -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,RANKING=$RANKING"
                    CMD="$CMD $JOB_SCRIPT"

                    if $DRY_RUN; then
                        echo "[DRY-RUN] $CONDITION | $DISTANCE | $DOMAIN | target_only | r=$RATIO | s$SEED"
                    else
                        JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] $CMD"; continue; }
                        echo "[SUBMIT] $CONDITION | $DISTANCE | $DOMAIN | target_only | r=$RATIO | s$SEED → $JOB_ID"
                        echo "$CONDITION:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
                        ((JOB_COUNT++))
                        sleep 0.1
                    fi
                done
            done

            # Balanced RF (no ratio)
            JOB_NAME="re_bf_${DISTANCE:0:2}${DOMAIN:0:1}_t_s${SEED}"
            CMD="qsub -N $JOB_NAME -l select=1:ncpus=2:mem=8gb -l walltime=00:30:00 -q SINGLE"
            CMD="$CMD -v CONDITION=balanced_rf,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,RANKING=$RANKING"
            CMD="$CMD $JOB_SCRIPT"

            if $DRY_RUN; then
                echo "[DRY-RUN] balanced_rf | $DISTANCE | $DOMAIN | target_only | s$SEED"
            else
                JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] $CMD"; continue; }
                echo "[SUBMIT] balanced_rf | $DISTANCE | $DOMAIN | target_only | s$SEED → $JOB_ID"
                echo "balanced_rf:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                ((JOB_COUNT++))
                sleep 0.1
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

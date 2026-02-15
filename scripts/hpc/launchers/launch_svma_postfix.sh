#!/bin/bash
# =============================================================================
# Launcher: Re-run SvmA (post-fix: MinMax norm + KSS6 + PSO accuracy)
# =============================================================================
# Submits all pooled + split2 jobs for SvmA only.
# Commit: 0152185 fix(SvmA): align with Arefnezhad 2019
#
# Usage:
#   ./scripts/hpc/launchers/launch_svma_postfix.sh --dry-run
#   ./scripts/hpc/launchers/launch_svma_postfix.sh
# =============================================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
POOLED_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research.sh"
SPLIT2_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

# TMPDIR workaround
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# ===== Configuration =====
MODEL="SvmA"
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODES=("source_only" "target_only")

# Queue round-robin across available queues
QUEUES=("SINGLE" "LONG" "DEFAULT" "SMALL")
QUEUE_IDX=0
next_queue() {
    local q="${QUEUES[$((QUEUE_IDX % ${#QUEUES[@]}))]}"
    ((QUEUE_IDX++))
    echo "$q"
}

NCPUS_MEM="ncpus=8:mem=32gb"
WALLTIME="24:00:00"

# ===== Parse arguments =====
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] No jobs will be submitted."
fi

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_svma_postfix_${TIMESTAMP}.log"

echo "============================================================"
echo "  Re-run SvmA (post-fix: MinMax + KSS6 + PSO→accuracy)"
echo "============================================================"
echo "  Commit:    $(cd $PROJECT_ROOT && git log --oneline -1)"
echo "  Seeds:     ${SEEDS[*]}"
echo "  Distances: ${DISTANCES[*]}"
echo "  Domains:   ${DOMAINS[*]}"
echo "  Modes:     ${MODES[*]}"
echo "  Ratios:    ${RATIOS[*]}"
echo "  Queues:    ${QUEUES[*]}"
echo "  Dry run:   $DRY_RUN"
echo "============================================================"
echo ""

{
    echo "# SvmA post-fix launch started at $(date)"
    echo "# Commit: $(cd $PROJECT_ROOT && git log --oneline -1)"
    echo ""
} > "$LOG_FILE"

POOLED_COUNT=0
SPLIT2_COUNT=0
SKIP_COUNT=0

# ===== 1. Pooled jobs (2 seeds) =====
echo "--- Pooled Jobs ---"
for SEED in "${SEEDS[@]}"; do
    JOB_NAME="SvmA_pooled_s${SEED}"
    QUEUE=$(next_queue)

    if $DRY_RUN; then
        echo "[DRY-RUN] qsub -N $JOB_NAME -v MODEL=$MODEL,SEED=$SEED -l walltime=$WALLTIME -q $QUEUE $POOLED_SCRIPT"
        ((POOLED_COUNT++))
    else
        JOB_ID=$(qsub \
            -N "$JOB_NAME" \
            -v "MODEL=$MODEL,SEED=$SEED" \
            -l "select=1:$NCPUS_MEM" \
            -l "walltime=$WALLTIME" \
            -q "$QUEUE" \
            "$POOLED_SCRIPT" 2>&1) || { echo "[ERROR] Failed: $JOB_NAME"; ((SKIP_COUNT++)); continue; }
        echo "[POOLED] SvmA | s$SEED | $QUEUE → $JOB_ID"
        echo "pooled:SvmA:$SEED:$QUEUE:$JOB_ID" >> "$LOG_FILE"
        ((POOLED_COUNT++))
        sleep 0.2
    fi
done
echo ""

# ===== 2. Split2 jobs =====
echo "--- Split2 Jobs ---"

for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                QUEUE=$(next_queue)

                # Baseline (no ratio)
                JOB_NAME="Sa_bs_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_s${SEED}"

                if $DRY_RUN; then
                    echo "[DRY-RUN] baseline | $DISTANCE | $DOMAIN | $MODE | s=$SEED | $QUEUE"
                    ((SPLIT2_COUNT++))
                else
                    JOB_ID=$(qsub -N "$JOB_NAME" \
                        -l "select=1:$NCPUS_MEM" -l "walltime=$WALLTIME" -q "$QUEUE" \
                        -v "MODEL=$MODEL,CONDITION=baseline,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true" \
                        "$SPLIT2_SCRIPT" 2>&1) || { echo "[ERROR] $JOB_NAME"; ((SKIP_COUNT++)); continue; }
                    echo "[SPLIT2] baseline | $DISTANCE | $DOMAIN | $MODE | s$SEED | $QUEUE → $JOB_ID"
                    echo "split2:baseline:$DISTANCE:$DOMAIN:$MODE:$SEED:$QUEUE:$JOB_ID" >> "$LOG_FILE"
                    ((SPLIT2_COUNT++))
                    sleep 0.2
                fi

                # Ratio-based conditions
                for RATIO in "${RATIOS[@]}"; do
                    for COND in smote_plain smote undersample; do
                        QUEUE=$(next_queue)
                        COND_SHORT="${COND:0:2}"
                        JOB_NAME="Sa_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_r${RATIO}_s${SEED}"

                        if $DRY_RUN; then
                            echo "[DRY-RUN] $COND | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s=$SEED | $QUEUE"
                            ((SPLIT2_COUNT++))
                        else
                            JOB_ID=$(qsub -N "$JOB_NAME" \
                                -l "select=1:$NCPUS_MEM" -l "walltime=$WALLTIME" -q "$QUEUE" \
                                -v "MODEL=$MODEL,CONDITION=$COND,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true" \
                                "$SPLIT2_SCRIPT" 2>&1) || { echo "[ERROR] $JOB_NAME"; ((SKIP_COUNT++)); continue; }
                            echo "[SPLIT2] $COND | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s$SEED | $QUEUE → $JOB_ID"
                            echo "split2:$COND:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED:$QUEUE:$JOB_ID" >> "$LOG_FILE"
                            ((SPLIT2_COUNT++))
                            sleep 0.2
                        fi
                    done
                done
            done
        done
    done
done

# Summary
{
    echo ""
    echo "# Launch completed at $(date)"
    echo "# Pooled: $POOLED_COUNT"
    echo "# Split2: $SPLIT2_COUNT"
    echo "# Skipped: $SKIP_COUNT"
    echo "# Total: $((POOLED_COUNT + SPLIT2_COUNT))"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
if $DRY_RUN; then
    echo "  [DRY-RUN] No jobs submitted."
fi
echo "  Pooled jobs:  $POOLED_COUNT"
echo "  Split2 jobs:  $SPLIT2_COUNT"
echo "  Skipped:      $SKIP_COUNT"
echo "  Total:        $((POOLED_COUNT + SPLIT2_COUNT))"
echo ""
echo "  Expected: 2 pooled + 168 split2 = 170 jobs"
echo "  Log: $LOG_FILE"
echo "============================================================"

#!/bin/bash
# ============================================================
# Prior research experiment — resubmit remaining 85 (differential submission)
# ============================================================
# launch_prior_research_resub_missing.sh 74 already submitted in first run.
# per-user limit submit remaining 85 that could not be submitted with.
#
# Breakdown:
#   Lstm mixed:       69 jobs
#   SvmA mixed:       10 jobs
#   SvmA target_only:  1 job
#   SvmW mixed:        5 jobs
#
# Queues: SINGLE/LONG/DEFAULT round-robin to
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
REMAINING_FILE="/tmp/remaining_jobs.txt"

N_TRIALS=100
RANKING="knn"
QUEUES=("SINGLE" "LONG" "DEFAULT")
QUEUE_COUNTER=0

DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *)         echo "Unknown option: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_resub_remaining_${TIMESTAMP}.log"

echo "============================================================"
echo "Prior research experiment — resubmit remaining 85"
echo "============================================================"
echo "Queues: ${QUEUES[*]}"
echo "Dry run: $DRY_RUN"
echo "============================================================"

if [[ ! -f "$REMAINING_FILE" ]]; then
    echo "[ERROR] Remaining jobs file not found: $REMAINING_FILE"
    exit 1
fi

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[ERROR] Job script not found: $JOB_SCRIPT"
    exit 1
fi

{
    echo "# Resub remaining started at $(date)"
    echo "# Command: $0 $*"
    echo ""
} > "$LOG_FILE"

JOB_COUNT=0
SKIP_COUNT=0

while IFS='|' read -r MODEL CONDITION MODE DISTANCE DOMAIN SEED RATIO WALLTIME MEM; do
    QUEUE="${QUEUES[$((QUEUE_COUNTER % 3))]}"
    ((QUEUE_COUNTER++))

    # Generate job name
    case "$MODE" in
        source_only) MODE_SHORT="s" ;;
        target_only) MODE_SHORT="t" ;;
        mixed)       MODE_SHORT="m" ;;
    esac
    COND_SHORT="${CONDITION:0:2}"
    JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE_SHORT}_s${SEED}"

    CMD="qsub -N $JOB_NAME -l select=1:ncpus=8:mem=${MEM} -l walltime=${WALLTIME} -q $QUEUE"
    CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    if [[ -n "$RATIO" ]]; then
        CMD="$CMD,RATIO=$RATIO"
    fi
    CMD="$CMD $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] $MODEL | $CONDITION | $MODE | $DISTANCE | $DOMAIN | s=$SEED | r=${RATIO:-N/A} | $QUEUE"
        ((JOB_COUNT++))
    else
        JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] $MODEL | $CONDITION | $MODE | $DISTANCE | $DOMAIN | s$SEED → FAILED ($QUEUE)"; ((SKIP_COUNT++)); continue; }
        echo "[SUBMIT] $MODEL | $CONDITION | $MODE | $DISTANCE | $DOMAIN | s$SEED | r=${RATIO:-N/A} → $JOB_ID ($QUEUE)"
        echo "$MODEL:$CONDITION:$MODE:$DISTANCE:$DOMAIN:$SEED:${RATIO:-}:$QUEUE:$JOB_ID" >> "$LOG_FILE"
        ((JOB_COUNT++))
        sleep 0.2
    fi
done < "$REMAINING_FILE"

{
    echo ""
    echo "# Launch completed at $(date)"
    echo "# Total jobs submitted: $JOB_COUNT"
    echo "# Skipped: $SKIP_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "Dry run complete. Expected: $JOB_COUNT"
else
    echo "Successfully submitted: $JOB_COUNT"
    echo "Skipped: $SKIP_COUNT"
    echo "Log: $LOG_FILE"
fi
echo "============================================================"

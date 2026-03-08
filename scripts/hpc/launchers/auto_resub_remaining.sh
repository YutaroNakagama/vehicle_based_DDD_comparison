#!/bin/bash
# ============================================================
# Auto-resubmit script — wait for queue slots & submit remaining
# ============================================================
# Read unsubmitted jobs from /tmp/remaining_jobs.txt,
# Submits if queue slots are available.
# Retries every 5 minutes until all jobs are submitted.
#
# Usage:
#   nohup bash scripts/hpc/launchers/auto_resub_remaining.sh &
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
REMAINING_FILE="/tmp/remaining_jobs.txt"
N_TRIALS=100
RANKING="knn"
QUEUES=("SINGLE" "LONG" "DEFAULT")
QUEUE_COUNTER=0
SLEEP_INTERVAL=300  # 5mininterval
MAX_RETRIES=100     # Max 100 rounds (approx 8 hours)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
LOG_FILE="$LOG_DIR/auto_resub_${TIMESTAMP}.log"

echo "============================================================"
echo "Auto resubmit script started: $(date)"
echo "remainingjob(s): $(wc -l < "$REMAINING_FILE") items"
echo "Retry interval: ${SLEEP_INTERVAL}s"
echo "Log: $LOG_FILE"
echo "============================================================"

echo "# Auto resub started at $(date)" > "$LOG_FILE"

RETRY=0
TOTAL_SUBMITTED=0

while [[ -s "$REMAINING_FILE" && $RETRY -lt $MAX_RETRIES ]]; do
    ((RETRY++))
    ROUND_SUBMITTED=0
    ROUND_SKIPPED=0

    # Writing unsubmitted jobs to temp file
    cp "$REMAINING_FILE" "${REMAINING_FILE}.bak"
    > "${REMAINING_FILE}.new"

    while IFS='|' read -r MODEL CONDITION MODE DISTANCE DOMAIN SEED RATIO WALLTIME MEM; do
        QUEUE="${QUEUES[$((QUEUE_COUNTER % 3))]}"
        ((QUEUE_COUNTER++))

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

        JOB_ID=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[$(date +%H:%M:%S)] SUBMIT $MODEL $CONDITION $MODE $DISTANCE $DOMAIN s$SEED r${RATIO:-N/A} → $JOB_ID ($QUEUE)"
            echo "$MODEL:$CONDITION:$MODE:$DISTANCE:$DOMAIN:$SEED:${RATIO:-}:$QUEUE:$JOB_ID" >> "$LOG_FILE"
            ((ROUND_SUBMITTED++))
            ((TOTAL_SUBMITTED++))
            sleep 0.2
        else
            # Submission failed → keep in resubmit list
            echo "$MODEL|$CONDITION|$MODE|$DISTANCE|$DOMAIN|$SEED|$RATIO|$WALLTIME|$MEM" >> "${REMAINING_FILE}.new"
            ((ROUND_SKIPPED++))
        fi
    done < "${REMAINING_FILE}.bak"

    # Update remaining list
    mv "${REMAINING_FILE}.new" "$REMAINING_FILE"
    REMAINING=$(wc -l < "$REMAINING_FILE")

    echo "[$(date +%H:%M:%S)] Round $RETRY: submitted=$ROUND_SUBMITTED skipped=$ROUND_SKIPPED remaining=$REMAINING total=$TOTAL_SUBMITTED"
    echo "# Round $RETRY at $(date): submitted=$ROUND_SUBMITTED skipped=$ROUND_SKIPPED remaining=$REMAINING" >> "$LOG_FILE"

    if [[ $REMAINING -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] All submissions complete!"
        break
    fi

    if [[ $ROUND_SUBMITTED -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] Could not submit in this round.${SLEEP_INTERVAL}retrying after seconds..."
        sleep $SLEEP_INTERVAL
    else
        echo "[$(date +%H:%M:%S)] Some submissions succeeded. Retrying remaining after 30s..."
        sleep 30
    fi
done

{
    echo ""
    echo "# Auto resub completed at $(date)"
    echo "# Total submitted: $TOTAL_SUBMITTED"
    echo "# Remaining: $(wc -l < "$REMAINING_FILE")"
    echo "# Retries: $RETRY"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "Auto re-submission complete: $(date)"
echo "Submitted: $TOTAL_SUBMITTED"
echo "remaining: $(wc -l < "$REMAINING_FILE")"
echo "Retry count: $RETRY"
echo "Log: $LOG_FILE"
echo "============================================================"

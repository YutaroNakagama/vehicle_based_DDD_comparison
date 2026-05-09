#!/bin/bash
# Auto-resubmit remaining exp2 jobs when queue has room.
# Runs as a daemon: checks queue every 30 min, submits up to MAX_PER_WAVE new jobs.

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
SUBMITTED_FILE="$LOG_DIR/submitted_exp2_v2.txt"
REMAINING_LOG="$LOG_DIR/auto_resubmit_exp2_remaining.log"
MAX_QUEUE=48       # Keep at most this many non-interactive jobs in queue
MAX_PER_WAVE=16    # Submit at most this many jobs per cycle
SLEEP_INTERVAL=1800  # 30 minutes

N_TRIALS=100
RANKING="knn"

get_resources() {
    local condition="$1"
    case "$condition" in
        balanced_rf) echo "ncpus=8:mem=12gb 08:00:00 LONG" ;;
        smote|smote_plain) echo "ncpus=4:mem=10gb 08:00:00 SINGLE" ;;
        baseline|undersample) echo "ncpus=4:mem=8gb 06:00:00 SINGLE" ;;
        *) echo "ncpus=4:mem=8gb 08:00:00 SINGLE" ;;
    esac
}

is_submitted() {
    local key="$1"
    grep -qF "$key" "$SUBMITTED_FILE" 2>/dev/null
}

queue_count() {
    qstat -u s2240011 2>/dev/null | grep -v "interactive\|^---\|^hakusan\|^Job\|^ " | wc -l
}

submit_one() {
    local CONDITION="$1" DISTANCE="$2" DOMAIN="$3" MODE="$4" RATIO="$5" SEED="$6"
    local RESOURCES NCPUS_MEM WALLTIME QUEUE JOB_NAME CMD JOB_ID KEY

    RESOURCES=$(get_resources "$CONDITION")
    NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
    WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
    QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)

    if [[ -n "$RATIO" ]]; then
        KEY="${CONDITION}:${DISTANCE}:${DOMAIN}:${MODE}:${RATIO}:${SEED}"
        JOB_NAME="${CONDITION:0:2}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_r${RATIO}_s${SEED}"
        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE \
          -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true \
          $JOB_SCRIPT"
    else
        KEY="${CONDITION}:${DISTANCE}:${DOMAIN}:${MODE}:${SEED}"
        JOB_NAME="${CONDITION:0:2}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_s${SEED}"
        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE \
          -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true \
          $JOB_SCRIPT"
    fi

    is_submitted "$KEY" && return 2   # already submitted, skip (return 2 so callers don't count it)

    JOB_ID=$(eval "$CMD" 2>&1) || { echo "[$(date)] ERROR submitting $KEY: $JOB_ID" >> "$REMAINING_LOG"; return 1; }
    echo "$KEY" >> "$SUBMITTED_FILE"
    echo "[$(date)] SUBMITTED $KEY → $JOB_ID" | tee -a "$REMAINING_LOG"
}

echo "[$(date)] auto_resubmit_exp2_remaining started (PID=$$)" | tee -a "$REMAINING_LOG"

while true; do
    CURRENT_Q=$(queue_count)
    SLOTS=$(( MAX_QUEUE - CURRENT_Q ))
    echo "[$(date)] Queue: $CURRENT_Q jobs, available slots: $SLOTS" >> "$REMAINING_LOG"

    if [[ $SLOTS -le 0 ]]; then
        echo "[$(date)] Queue full, sleeping ${SLEEP_INTERVAL}s" >> "$REMAINING_LOG"
        sleep "$SLEEP_INTERVAL"
        continue
    fi

    SUBMITTED_THIS_WAVE=0

    for DISTANCE in mmd dtw wasserstein; do
      for DOMAIN in out_domain in_domain; do
        for MODE in source_only target_only; do
          for SEED in 42 123; do
            [[ $SUBMITTED_THIS_WAVE -ge $MAX_PER_WAVE ]] && break 4
            [[ $(( MAX_QUEUE - $(queue_count) )) -le 0 ]] && break 4

            # baseline (no ratio)
            submit_one "baseline" "$DISTANCE" "$DOMAIN" "$MODE" "" "$SEED" && ((SUBMITTED_THIS_WAVE++))

            for RATIO in 0.1 0.5; do
              [[ $SUBMITTED_THIS_WAVE -ge $MAX_PER_WAVE ]] && break
              [[ $(( MAX_QUEUE - $(queue_count) )) -le 0 ]] && break
              submit_one "smote_plain" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" && ((SUBMITTED_THIS_WAVE++))
              submit_one "smote" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" && ((SUBMITTED_THIS_WAVE++))
              submit_one "undersample" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" && ((SUBMITTED_THIS_WAVE++))
            done

            submit_one "balanced_rf" "$DISTANCE" "$DOMAIN" "$MODE" "" "$SEED" && ((SUBMITTED_THIS_WAVE++))
          done
        done
      done
    done

    # Check if all 192 are done
    TOTAL_SUBMITTED=$(grep -c . "$SUBMITTED_FILE" 2>/dev/null || echo 0)
    echo "[$(date)] Wave done. Submitted this wave: $SUBMITTED_THIS_WAVE, Total submitted: $TOTAL_SUBMITTED/192" >> "$REMAINING_LOG"

    if [[ $TOTAL_SUBMITTED -ge 192 ]]; then
        echo "[$(date)] All 192 jobs submitted. Daemon exiting." | tee -a "$REMAINING_LOG"
        exit 0
    fi

    sleep "$SLEEP_INTERVAL"
done

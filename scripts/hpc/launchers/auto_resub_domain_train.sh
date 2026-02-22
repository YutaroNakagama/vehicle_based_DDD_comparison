#!/bin/bash
# ============================================================
# Auto-resubmit daemon for failed SvmA + Lstm domain_train jobs
# ============================================================
# Reads FAIL: entries from the launch log and resubmits them
# as queue capacity becomes available. Runs in background.
#
# Usage:
#   nohup bash scripts/hpc/launchers/auto_resub_domain_train.sh &
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
DOMAIN_TRAIN_JOB="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
N_TRIALS=100
RANKING="knn"
MAX_JOB_LIMIT=50           # Target active jobs for this user
BATCH_SIZE=10              # Submit this many per round
CHECK_INTERVAL=300         # Seconds between checks (5 min)

LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DAEMON_LOG="$LOG_DIR/auto_resub_dt_daemon_${TIMESTAMP}.log"

# Source log with FAIL entries
SOURCE_LOG="$LOG_DIR/resubmit_svma_lstm_fixed_20260222_173358.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DAEMON_LOG"; }

# Extract failed entries into a temp file
PENDING_FILE=$(mktemp /tmp/pending_resub_dt_XXXX.txt)
grep "^FAIL:" "$SOURCE_LOG" | sed 's/^FAIL://' | cut -d: -f1-7 > "$PENDING_FILE"
TOTAL=$(wc -l < "$PENDING_FILE")
log "Daemon started. $TOTAL failed jobs to resubmit."
log "Source log: $SOURCE_LOG"
log "Pending file: $PENDING_FILE"

SUBMITTED=0

get_resources_domain_train() {
    local model="$1"
    local queues=("SINGLE" "LONG" "DEFAULT")
    local queue="${queues[$((RANDOM % 3))]}"
    case "$model" in
        SvmA) echo "ncpus=8:mem=32gb 48:00:00 $queue" ;;
        Lstm) echo "ncpus=8:mem=32gb 20:00:00 $queue" ;;
    esac
}

while [[ -s "$PENDING_FILE" ]]; do
    # Check current queue usage
    CURRENT=$(qstat -u s2240011 2>/dev/null | awk 'NR>5{c++}END{print c+0}')
    AVAILABLE=$((MAX_JOB_LIMIT - CURRENT))
    
    if [[ $AVAILABLE -le 0 ]]; then
        REMAINING=$(wc -l < "$PENDING_FILE")
        log "Queue full ($CURRENT jobs). $REMAINING pending. Sleeping ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
        continue
    fi

    # Submit up to BATCH_SIZE or AVAILABLE, whichever is smaller
    TO_SUBMIT=$((AVAILABLE < BATCH_SIZE ? AVAILABLE : BATCH_SIZE))
    ROUND_SUBMITTED=0

    while IFS=: read -r MODEL CONDITION DISTANCE DOMAIN MODE RATIO SEED; do
        [[ $ROUND_SUBMITTED -ge $TO_SUBMIT ]] && break

        RESOURCES=$(get_resources_domain_train "$MODEL")
        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)

        # Build job name
        local_model_short="${MODEL:0:2}"
        case "$CONDITION" in
            baseline)     local_cond_short="bs" ;;
            smote_plain)  local_cond_short="sp" ;;
            smote)        local_cond_short="sm" ;;
            undersample)  local_cond_short="un" ;;
        esac

        if [[ "$CONDITION" == "baseline" || "$RATIO" == "none" ]]; then
            JOB_NAME="${local_model_short}_${local_cond_short}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_s${SEED}"
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
            CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        else
            JOB_NAME="${local_model_short}_${local_cond_short}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_r${RATIO}_s${SEED}"
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
            CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true,RATIO=$RATIO"
        fi
        CMD="$CMD $DOMAIN_TRAIN_JOB"

        JOB_ID=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            log "[SUBMIT] $MODEL:$CONDITION:$DISTANCE:$DOMAIN:$RATIO:s$SEED → $JOB_ID ($QUEUE)"
            ((SUBMITTED++))
            ((ROUND_SUBMITTED++))
            # Remove this entry from pending
            sed -i "1d" "$PENDING_FILE"
            sleep 0.15
        else
            log "[FAIL] $MODEL:$CONDITION:$DISTANCE:$DOMAIN:$RATIO:s$SEED → $JOB_ID"
            # Don't remove — keep for retry
            break
        fi
    done < "$PENDING_FILE"

    REMAINING=$(wc -l < "$PENDING_FILE")
    log "Round: submitted $ROUND_SUBMITTED. Total submitted: $SUBMITTED/$TOTAL. Remaining: $REMAINING."
    
    if [[ $REMAINING -gt 0 ]]; then
        log "Sleeping ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
    fi
done

log "All $TOTAL jobs resubmitted. Daemon exiting."
rm -f "$PENDING_FILE"

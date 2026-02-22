#!/bin/bash
# ============================================================
# Auto-submit daemon for SvmA + Lstm mixed jobs
# ============================================================
# Submits mixed-mode jobs when queue capacity is available.
# Should be started after most domain_train jobs have been submitted.
#
# Usage:
#   nohup bash scripts/hpc/launchers/auto_submit_mixed.sh &
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
MIXED_JOB="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
N_TRIALS=100
RANKING="knn"
MAX_JOB_LIMIT=50
BATCH_SIZE=10
CHECK_INTERVAL=300

SEEDS=(42 123)
RATIOS=(0.1 0.5)
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODELS=("SvmA" "Lstm")
MODE="mixed"

LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DAEMON_LOG="$LOG_DIR/auto_submit_mixed_daemon_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DAEMON_LOG"; }

# Generate all mixed job combinations
PENDING_FILE=$(mktemp /tmp/pending_mixed_XXXX.txt)
for MODEL in "${MODELS[@]}"; do
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                echo "$MODEL:baseline:$DISTANCE:$DOMAIN:$MODE:none:$SEED" >> "$PENDING_FILE"
                for RATIO in "${RATIOS[@]}"; do
                    for COND in "smote_plain" "smote" "undersample"; do
                        echo "$MODEL:$COND:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED" >> "$PENDING_FILE"
                    done
                done
            done
        done
    done
done
TOTAL=$(wc -l < "$PENDING_FILE")
log "Mixed daemon started. $TOTAL jobs to submit."
log "Pending file: $PENDING_FILE"

SUBMITTED=0

get_resources_mixed() {
    local model="$1"
    local queues=("SINGLE" "LONG" "DEFAULT")
    local queue="${queues[$((RANDOM % 3))]}"
    case "$model" in
        SvmA) echo "ncpus=8:mem=48gb 30:00:00 $queue" ;;
        Lstm) echo "ncpus=8:mem=48gb 24:00:00 $queue" ;;
    esac
}

while [[ -s "$PENDING_FILE" ]]; do
    CURRENT=$(qstat -u s2240011 2>/dev/null | awk 'NR>5{c++}END{print c+0}')
    AVAILABLE=$((MAX_JOB_LIMIT - CURRENT))

    if [[ $AVAILABLE -le 0 ]]; then
        REMAINING=$(wc -l < "$PENDING_FILE")
        log "Queue full ($CURRENT jobs). $REMAINING pending. Sleeping ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
        continue
    fi

    TO_SUBMIT=$((AVAILABLE < BATCH_SIZE ? AVAILABLE : BATCH_SIZE))
    ROUND_SUBMITTED=0

    while IFS=: read -r MODEL CONDITION DISTANCE DOMAIN _MODE RATIO SEED; do
        [[ $ROUND_SUBMITTED -ge $TO_SUBMIT ]] && break

        RESOURCES=$(get_resources_mixed "$MODEL")
        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)

        local_model_short="${MODEL:0:2}"
        case "$CONDITION" in
            baseline)     local_cond_short="bs" ;;
            smote_plain)  local_cond_short="sp" ;;
            smote)        local_cond_short="sm" ;;
            undersample)  local_cond_short="un" ;;
        esac

        if [[ "$CONDITION" == "baseline" || "$RATIO" == "none" ]]; then
            JOB_NAME="${local_model_short}_${local_cond_short}_${DISTANCE:0:1}${DOMAIN:0:1}_mi_s${SEED}"
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
            CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,MODE=mixed,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        else
            JOB_NAME="${local_model_short}_${local_cond_short}_${DISTANCE:0:1}${DOMAIN:0:1}_mi_r${RATIO}_s${SEED}"
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
            CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,MODE=mixed,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true,RATIO=$RATIO"
        fi
        CMD="$CMD $MIXED_JOB"

        JOB_ID=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            log "[SUBMIT] $MODEL:$CONDITION:$DISTANCE:$DOMAIN:$RATIO:s$SEED → $JOB_ID ($QUEUE)"
            ((SUBMITTED++))
            ((ROUND_SUBMITTED++))
            sed -i "1d" "$PENDING_FILE"
            sleep 0.15
        else
            log "[FAIL] $MODEL:$CONDITION:$DISTANCE:$DOMAIN:$RATIO:s$SEED → $JOB_ID"
            break
        fi
    done < "$PENDING_FILE"

    REMAINING=$(wc -l < "$PENDING_FILE")
    log "Round: submitted $ROUND_SUBMITTED. Total: $SUBMITTED/$TOTAL. Remaining: $REMAINING."

    if [[ $REMAINING -gt 0 ]]; then
        log "Sleeping ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
    fi
done

log "All $TOTAL mixed jobs submitted. Daemon exiting."
rm -f "$PENDING_FILE"

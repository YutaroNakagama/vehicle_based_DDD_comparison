#!/bin/bash
# ============================================================
# Auto-resubmit daemon v3 - uses all 4 CPU queues
# Submits remaining domain_train jobs as slots open up
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
DOMAIN_TRAIN_JOB="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
N_TRIALS=100
RANKING="knn"
CHECK_INTERVAL=300

LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DAEMON_LOG="$LOG_DIR/auto_resub_dt_v3_daemon_${TIMESTAMP}.log"

PENDING_FILE="/tmp/pending_dt_all.txt"
if [[ ! -f "$PENDING_FILE" ]] || [[ ! -s "$PENDING_FILE" ]]; then
    echo "ERROR: $PENDING_FILE not found or empty" >&2
    exit 1
fi
TOTAL=$(wc -l < "$PENDING_FILE")

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$DAEMON_LOG"; }

log "Daemon v3 started. $TOTAL remaining jobs to submit."

# Per-queue user limits (from qstat -Qf)
declare -A Q_MAX=([DEFAULT]=40 [SINGLE]=40 [SMALL]=30 [LONG]=15)
QUEUES=("DEFAULT" "SINGLE" "SMALL" "LONG")

SUBMITTED=0

while [[ -s "$PENDING_FILE" ]]; do
    # Count current jobs per queue
    declare -A Q_CUR
    for q in "${QUEUES[@]}"; do Q_CUR[$q]=0; done
    while read -r cnt queue; do
        if [[ -n "${Q_MAX[$queue]+x}" ]]; then
            Q_CUR[$queue]=$cnt
        fi
    done < <(qstat -u s2240011 2>/dev/null | awk 'NR>5{print $3}' | sort | uniq -c | awk '{print $1, $2}')

    # Calculate available slots per queue
    TOTAL_AVAILABLE=0
    AVAIL_QUEUES=()
    for q in "${QUEUES[@]}"; do
        avail=$(( ${Q_MAX[$q]} - ${Q_CUR[$q]:-0} ))
        if [[ $avail -gt 0 ]]; then
            AVAIL_QUEUES+=("$q:$avail")
            TOTAL_AVAILABLE=$((TOTAL_AVAILABLE + avail))
        fi
    done

    if [[ $TOTAL_AVAILABLE -le 0 ]]; then
        REMAINING=$(wc -l < "$PENDING_FILE")
        log "All queues full. $REMAINING pending. Sleeping ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
        continue
    fi

    log "Available slots: ${AVAIL_QUEUES[*]} (total: $TOTAL_AVAILABLE)"

    ROUND_SUBMITTED=0
    # Build ordered list of queues to use with their capacity
    Q_ROTATE_IDX=0
    Q_COUNT=${#AVAIL_QUEUES[@]}

    while IFS=: read -r MODEL CONDITION DISTANCE DOMAIN MODE RATIO SEED; do
        [[ $ROUND_SUBMITTED -ge $TOTAL_AVAILABLE ]] && break

        # Find next queue with capacity
        FOUND_QUEUE=""
        for attempt in $(seq 0 $((Q_COUNT - 1))); do
            idx=$(( (Q_ROTATE_IDX + attempt) % Q_COUNT ))
            q_info="${AVAIL_QUEUES[$idx]}"
            q_name="${q_info%%:*}"
            q_slots="${q_info##*:}"
            if [[ $q_slots -gt 0 ]]; then
                FOUND_QUEUE="$q_name"
                AVAIL_QUEUES[$idx]="${q_name}:$((q_slots - 1))"
                Q_ROTATE_IDX=$(( (idx + 1) % Q_COUNT ))
                break
            fi
        done
        [[ -z "$FOUND_QUEUE" ]] && break

        # Resource specs per model
        case "$MODEL" in
            SvmA) NCPUS_MEM="ncpus=8:mem=32gb"; WALLTIME="48:00:00" ;;
            Lstm) NCPUS_MEM="ncpus=8:mem=32gb"; WALLTIME="20:00:00" ;;
            *)    NCPUS_MEM="ncpus=8:mem=32gb"; WALLTIME="48:00:00" ;;
        esac

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
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $FOUND_QUEUE"
            CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        else
            JOB_NAME="${local_model_short}_${local_cond_short}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_r${RATIO}_s${SEED}"
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $FOUND_QUEUE"
            CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true,RATIO=$RATIO"
        fi
        CMD="$CMD $DOMAIN_TRAIN_JOB"

        JOB_ID=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            log "[SUBMIT] $MODEL:$CONDITION:$DISTANCE:$DOMAIN:r$RATIO:s$SEED → $JOB_ID ($FOUND_QUEUE)"
            ((SUBMITTED++))
            ((ROUND_SUBMITTED++))
            sed -i "1d" "$PENDING_FILE"
            sleep 0.1
        else
            log "[FAIL] $FOUND_QUEUE: $JOB_ID"
            break
        fi
    done < "$PENDING_FILE"

    REMAINING=$(wc -l < "$PENDING_FILE")
    log "Round: submitted $ROUND_SUBMITTED. Total submitted: $SUBMITTED. Remaining: $REMAINING."

    if [[ $REMAINING -gt 0 ]]; then
        log "Sleeping ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
    fi
done

log "All jobs submitted ($SUBMITTED total). Daemon exiting."

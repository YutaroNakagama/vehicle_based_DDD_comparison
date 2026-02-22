#!/bin/bash
# ============================================================
# Auto-submit daemon for missing SvmW source_only/target_only/mixed jobs
# Uses pbs_prior_research_split2.sh (MODE-based script)
# Monitors 4 CPU queues and submits when slots are available
# ============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
PENDING_FILE="/tmp/pending_svmw_modes.txt"
PBS_SCRIPT="${PROJECT_ROOT}/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/train"
LOG_FILE="${LOG_DIR}/auto_resub_svmw_modes_daemon_$(date +%Y%m%d_%H%M%S).log"

# Queue limits (per-user max_queued)
declare -A QUEUE_MAX=( [DEFAULT]=40 [SINGLE]=40 [SMALL]=30 [LONG]=15 )
QUEUES=(DEFAULT SINGLE SMALL LONG)

# SvmW settings (fast jobs)
NCPUS_MEM="ncpus=4:mem=8gb"
WALLTIME="03:00:00"
SLEEP_INTERVAL=120  # 2 min (SvmW is fast)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

get_queue_count() {
    local q=$1
    qstat -u s2240011 2>/dev/null | grep " $q " | wc -l
}

submit_job() {
    local line=$1
    # Format: SvmW:mode:condition:distance:domain:ratio:seed
    IFS=':' read -r MODEL MODE CONDITION DISTANCE DOMAIN RATIO SEED <<< "$line"

    # Build short job name
    local cond_short
    case "$CONDITION" in
        baseline) cond_short="bs" ;;
        smote) cond_short="sm" ;;
        smote_plain) cond_short="sp" ;;
        undersample) cond_short="un" ;;
        *) cond_short="${CONDITION:0:2}" ;;
    esac

    local mode_short
    case "$MODE" in
        source_only) mode_short="so" ;;
        target_only) mode_short="to" ;;
        mixed) mode_short="mx" ;;
        *) mode_short="${MODE:0:2}" ;;
    esac

    local JOB_NAME
    if [[ "$CONDITION" == "baseline" ]]; then
        JOB_NAME="Sw_${cond_short}_${DISTANCE:0:1}${DOMAIN:0:1}_${mode_short}_s${SEED}"
    else
        JOB_NAME="Sw_${cond_short}_${DISTANCE:0:1}${DOMAIN:0:1}_${mode_short}_r${RATIO}_s${SEED}"
    fi

    # Find available queue
    local FOUND_QUEUE=""
    for q in "${QUEUES[@]}"; do
        local count
        count=$(get_queue_count "$q")
        local max=${QUEUE_MAX[$q]}
        if (( count < max )); then
            FOUND_QUEUE="$q"
            break
        fi
    done

    if [[ -z "$FOUND_QUEUE" ]]; then
        return 1  # No available queue
    fi

    # Build qsub command
    local CMD="qsub -N $JOB_NAME -l select=1:${NCPUS_MEM} -l walltime=${WALLTIME} -q ${FOUND_QUEUE}"
    CMD="$CMD -v MODEL=SvmW,CONDITION=${CONDITION},MODE=${MODE},DISTANCE=${DISTANCE},DOMAIN=${DOMAIN},SEED=${SEED}"
    if [[ "$RATIO" != "none" ]]; then
        CMD="$CMD,RATIO=${RATIO}"
    fi
    CMD="$CMD ${PBS_SCRIPT}"

    local result
    result=$(eval "$CMD" 2>&1) || true

    if [[ "$result" == *".spcc-adm1"* ]]; then
        log "[SUBMIT] SvmW:${MODE}:${CONDITION}:${DISTANCE}:${DOMAIN}:r${RATIO}:s${SEED} → ${result} (${FOUND_QUEUE})"
        return 0
    else
        log "[FAIL] $JOB_NAME: $result"
        return 1
    fi
}

# Main loop
log "Starting SvmW modes daemon. Pending: $(wc -l < "$PENDING_FILE") jobs."
log "PBS script: $PBS_SCRIPT"
log "Resources: $NCPUS_MEM walltime=$WALLTIME"

TOTAL_SUBMITTED=0
while [[ -s "$PENDING_FILE" ]]; do
    REMAINING=$(wc -l < "$PENDING_FILE")

    # Check available slots across all queues
    TOTAL_AVAIL=0
    AVAIL_MSG=""
    for q in "${QUEUES[@]}"; do
        count=$(get_queue_count "$q")
        max=${QUEUE_MAX[$q]}
        avail=$((max - count))
        if (( avail > 0 )); then
            TOTAL_AVAIL=$((TOTAL_AVAIL + avail))
            AVAIL_MSG="${AVAIL_MSG} ${q}:${avail}"
        fi
    done

    if (( TOTAL_AVAIL == 0 )); then
        log "All queues full. ${REMAINING} pending. Sleeping ${SLEEP_INTERVAL}s..."
        sleep "$SLEEP_INTERVAL"
        continue
    fi

    log "Available slots:${AVAIL_MSG} (total: ${TOTAL_AVAIL})"

    # Submit jobs up to available slots
    SUBMITTED_THIS_ROUND=0
    TEMP_FILE=$(mktemp)
    while IFS= read -r line; do
        if (( SUBMITTED_THIS_ROUND >= TOTAL_AVAIL )); then
            echo "$line" >> "$TEMP_FILE"
            continue
        fi

        if submit_job "$line"; then
            SUBMITTED_THIS_ROUND=$((SUBMITTED_THIS_ROUND + 1))
            TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
        else
            echo "$line" >> "$TEMP_FILE"
        fi
    done < "$PENDING_FILE"
    mv "$TEMP_FILE" "$PENDING_FILE"

    REMAINING=$(wc -l < "$PENDING_FILE")
    log "Round: submitted ${SUBMITTED_THIS_ROUND}. Total submitted: ${TOTAL_SUBMITTED}. Remaining: ${REMAINING}."
    log "Sleeping ${SLEEP_INTERVAL}s..."
    sleep "$SLEEP_INTERVAL"
done

log "All jobs submitted (${TOTAL_SUBMITTED} total). Daemon exiting."

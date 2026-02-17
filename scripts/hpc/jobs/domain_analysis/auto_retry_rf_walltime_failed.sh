#!/bin/bash
# ============================================================
# Auto-retry launcher for 2 RF smote_plain jobs that hit 10h walltime
# ============================================================
# These 2 jobs completed 100 Optuna trials but couldn't finish
# training/eval within DEFAULT queue's 10h limit.
# Resubmits to LONG queue with 15h walltime.
#
# Failed jobs:
#   14850665: smote_plain / mmd / out_domain / mixed / r=0.5 / s123
#             (Exit -29, 10:01:14 elapsed, model saved but no eval)
#   14850667: smote_plain / wasserstein / out_domain / mixed / r=0.5 / s123
#             (Exit -29, 10:00:59 elapsed, killed during calibration, no model.pkl)
#
# Usage:
#   nohup bash scripts/hpc/jobs/domain_analysis/auto_retry_rf_walltime_failed.sh &
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

NCPUS=4
MEM="10gb"
WALLTIME="15:00:00"   # 15h (was 10h in DEFAULT; these need >10h)
SCRIPT_PATH="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"

N_TRIALS=100
RANKING="knn"
RETRY_INTERVAL=300  # 5 minutes

# Target queue: LONG (max 15 per user, supports 15h walltime)
TARGET_QUEUE="LONG"
QUEUE_MAX=15

# Define the 2 failed configs: "CONDITION:DISTANCE:DOMAIN:SEED:RATIO"
declare -a ALL_CONFIGS=(
    "smote_plain:mmd:out_domain:123:0.5"
    "smote_plain:wasserstein:out_domain:123:0.5"
)

# Track submitted jobs
SUBMITTED_FILE="${LOG_DIR}/retry_rf_walltime_failed_submitted.txt"
touch "$SUBMITTED_FILE"

SESSION_LOG="${LOG_DIR}/auto_retry_rf_walltime_$(date +%Y%m%d_%H%M%S).log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$SESSION_LOG"
}

count_queue_jobs() {
    qstat -u "$USER" 2>/dev/null | awk -v q="$TARGET_QUEUE" '$3==q' | wc -l
}

submit_job() {
    local condition=$1 dist=$2 domain=$3 seed=$4 ratio=$5

    local JOB_NAME="rf_r05_sp_${dist:0:4}_od_s${seed}"
    JOB_NAME="${JOB_NAME:0:15}"

    local JOB_ID
    JOB_ID=$(qsub \
        -N "$JOB_NAME" \
        -q "$TARGET_QUEUE" \
        -l select=1:ncpus=${NCPUS}:mem=${MEM} \
        -l walltime=${WALLTIME} \
        -v CONDITION="$condition",MODE=mixed,DISTANCE="$dist",DOMAIN="$domain",RATIO="$ratio",SEED="$seed",N_TRIALS="$N_TRIALS",RANKING="$RANKING",RUN_EVAL=true \
        "$SCRIPT_PATH" 2>&1)

    if [[ $? -eq 0 ]]; then
        echo "${condition}:${dist}:${domain}:${seed}:${ratio}" >> "$SUBMITTED_FILE"
        log "SUBMITTED: $JOB_NAME -> $TARGET_QUEUE ($JOB_ID) [walltime=$WALLTIME]"
        return 0
    else
        log "FAILED to submit: $JOB_NAME -> $TARGET_QUEUE (Error: $JOB_ID)"
        return 1
    fi
}

log "Starting auto-retry for 2 RF walltime-exceeded jobs"
log "Target queue: $TARGET_QUEUE (max ${QUEUE_MAX}), walltime: $WALLTIME"
log "Checking every ${RETRY_INTERVAL}s for available slots"

while true; do
    remaining=0
    for config in "${ALL_CONFIGS[@]}"; do
        if ! grep -qF "$config" "$SUBMITTED_FILE" 2>/dev/null; then
            ((remaining++))
        fi
    done

    if [[ "$remaining" -eq 0 ]]; then
        log "ALL 2 JOBS SUBMITTED. Exiting."
        break
    fi

    log "Remaining: ${remaining}/2"

    current=$(count_queue_jobs)
    if [[ "$current" -ge "$QUEUE_MAX" ]]; then
        log "$TARGET_QUEUE full ($current/$QUEUE_MAX). Waiting ${RETRY_INTERVAL}s..."
        sleep "$RETRY_INTERVAL"
        continue
    fi

    available=$((QUEUE_MAX - current))
    log "$TARGET_QUEUE has $available slots available ($current/$QUEUE_MAX)"

    for config in "${ALL_CONFIGS[@]}"; do
        if grep -qF "$config" "$SUBMITTED_FILE" 2>/dev/null; then
            continue
        fi

        current=$(count_queue_jobs)
        if [[ "$current" -ge "$QUEUE_MAX" ]]; then
            log "Queue filled during submission. Waiting..."
            break
        fi

        IFS=':' read -r condition dist domain seed ratio <<< "$config"
        submit_job "$condition" "$dist" "$domain" "$seed" "$ratio"
        sleep 1
    done

    # Check if all done
    remaining=0
    for config in "${ALL_CONFIGS[@]}"; do
        if ! grep -qF "$config" "$SUBMITTED_FILE" 2>/dev/null; then
            ((remaining++))
        fi
    done

    if [[ "$remaining" -eq 0 ]]; then
        log "ALL 2 JOBS SUBMITTED. Exiting."
        break
    fi

    log "Waiting ${RETRY_INTERVAL}s for next check..."
    sleep "$RETRY_INTERVAL"
done

log "Auto-retry script finished."

#!/bin/bash
# ============================================================
# Auto-retry launcher for failed exp2 ratio=0.5 split2 jobs
# ============================================================
# Resubmits 5 failed jobs (4× smote_plain, 1× sw_smote) to
# available queue slots. Checks every 5 minutes.
#
# Failed jobs:
#   14849058: smote_plain / DTW / in_domain / s123  (Exit 271 - SMALL timeout)
#   14849059: smote_plain / MMD / out_domain / s123  (Exit 271)
#   14849060: smote_plain / Wass / in_domain / s123  (Exit 271)
#   14849061: smote_plain / Wass / out_domain / s123 (Exit 271)
#   14849123: sw_smote / Wass / out_domain / s42     (Exit 143 - preempt)
#
# Usage:
#   nohup bash scripts/hpc/jobs/domain_analysis/auto_retry_r05_failed.sh &
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

NCPUS=4
MEM="10gb"
WALLTIME="10:00:00"
SCRIPT_PATH="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"

N_TRIALS=100
RANKING="knn"
RETRY_INTERVAL=300  # 5 minutes

# Queue list with max_queued per user (avoid SMALL - caused the timeout issue)
declare -A QUEUE_MAX
QUEUE_MAX[DEFAULT]=40
QUEUE_MAX[SINGLE]=40
QUEUE_MAX[LONG]=15

# Define the 5 failed configs: "CONDITION:DISTANCE:DOMAIN:SEED:RATIO"
declare -a ALL_CONFIGS=(
    "smote_plain:dtw:in_domain:123:0.5"
    "smote_plain:mmd:out_domain:123:0.5"
    "smote_plain:wasserstein:in_domain:123:0.5"
    "smote_plain:wasserstein:out_domain:123:0.5"
    "smote:wasserstein:out_domain:42:0.5"
)

# Track submitted jobs
SUBMITTED_FILE="${LOG_DIR}/retry_r05_failed_submitted.txt"
touch "$SUBMITTED_FILE"

SESSION_LOG="${LOG_DIR}/auto_retry_r05_failed_$(date +%Y%m%d_%H%M%S).log"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$SESSION_LOG"
}

count_queue_jobs() {
    local queue=$1
    qstat -u "$USER" 2>/dev/null | awk -v q="$queue" 'NR>5 && $3==q' | wc -l
}

find_available_queue() {
    for queue in DEFAULT SINGLE LONG; do
        local current
        current=$(count_queue_jobs "$queue")
        local max=${QUEUE_MAX[$queue]}
        if [[ "$current" -lt "$max" ]]; then
            echo "$queue"
            return 0
        fi
    done
    return 1
}

submit_job() {
    local condition=$1 dist=$2 domain=$3 seed=$4 ratio=$5 queue=$6

    # Build short job name
    case "$condition" in
        smote_plain) cond_short="sp" ;;
        smote)       cond_short="sm" ;;  # sw_smote
    esac
    case "$domain" in
        in_domain)   dom_short="id" ;;
        out_domain)  dom_short="od" ;;
    esac
    local JOB_NAME="rf_r05_${cond_short}_${dist:0:4}_${dom_short}_s${seed}"
    JOB_NAME="${JOB_NAME:0:15}"

    local JOB_ID
    JOB_ID=$(qsub \
        -N "$JOB_NAME" \
        -q "$queue" \
        -l select=1:ncpus=${NCPUS}:mem=${MEM} \
        -l walltime=${WALLTIME} \
        -v CONDITION="$condition",MODE=mixed,DISTANCE="$dist",DOMAIN="$domain",RATIO="$ratio",SEED="$seed",N_TRIALS="$N_TRIALS",RANKING="$RANKING",RUN_EVAL=true \
        "$SCRIPT_PATH" 2>&1)

    if [[ $? -eq 0 ]]; then
        echo "${condition}:${dist}:${domain}:${seed}:${ratio}" >> "$SUBMITTED_FILE"
        log "SUBMITTED: $JOB_NAME -> $queue ($JOB_ID)"
        return 0
    else
        log "FAILED to submit: $JOB_NAME -> $queue (Error: $JOB_ID)"
        return 1
    fi
}

log "Starting auto-retry for 5 failed exp2 r0.5 jobs"
log "Checking every ${RETRY_INTERVAL}s for available queue slots"
log "Queues: DEFAULT(40) SINGLE(40) LONG(15) [SMALL excluded]"

while true; do
    remaining=0
    for config in "${ALL_CONFIGS[@]}"; do
        if ! grep -qF "$config" "$SUBMITTED_FILE" 2>/dev/null; then
            ((remaining++))
        fi
    done

    if [[ "$remaining" -eq 0 ]]; then
        log "ALL 5 JOBS SUBMITTED. Exiting."
        break
    fi

    log "Remaining: ${remaining}/5"

    queue=$(find_available_queue) || {
        log "All queues full. Waiting ${RETRY_INTERVAL}s..."
        sleep "$RETRY_INTERVAL"
        continue
    }

    for config in "${ALL_CONFIGS[@]}"; do
        if grep -qF "$config" "$SUBMITTED_FILE" 2>/dev/null; then
            continue
        fi

        IFS=':' read -r condition dist domain seed ratio <<< "$config"

        queue=$(find_available_queue) || {
            log "Queue full after partial submission. Waiting..."
            break
        }

        submit_job "$condition" "$dist" "$domain" "$seed" "$ratio" "$queue"
        sleep 1
    done

    # Check if all done after this round
    remaining=0
    for config in "${ALL_CONFIGS[@]}"; do
        if ! grep -qF "$config" "$SUBMITTED_FILE" 2>/dev/null; then
            ((remaining++))
        fi
    done

    if [[ "$remaining" -eq 0 ]]; then
        log "ALL 5 JOBS SUBMITTED. Exiting."
        break
    fi

    log "Waiting ${RETRY_INTERVAL}s for next check..."
    sleep "$RETRY_INTERVAL"
done

log "Auto-retry script finished."

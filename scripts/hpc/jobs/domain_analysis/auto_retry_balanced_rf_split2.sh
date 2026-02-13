#!/bin/bash
# ============================================================
# Auto-retry launcher for BalancedRF split2 rerun
# ============================================================
# Runs in background, checking every 5 minutes for queue slots.
# Submits BalancedRF jobs as soon as capacity becomes available.
# Stops when all 36 jobs are submitted.
#
# Usage:
#   nohup bash scripts/hpc/jobs/domain_analysis/auto_retry_balanced_rf_split2.sh &
#   # or
#   bash scripts/hpc/jobs/domain_analysis/auto_retry_balanced_rf_split2.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

NCPUS=4
MEM="32gb"
WALLTIME="24:00:00"
SCRIPT_PATH="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"

CONDITION="balanced_rf"
N_TRIALS=150
RANKING="knn"
RETRY_INTERVAL=300  # 5 minutes

# Queue list with max_queued per user
declare -A QUEUE_MAX
QUEUE_MAX[DEFAULT]=40
QUEUE_MAX[SINGLE]=40
QUEUE_MAX[SMALL]=30
QUEUE_MAX[LONG]=15

# Build all 36 configs
declare -a ALL_CONFIGS=()
for dist in dtw mmd wasserstein; do
    for domain in in_domain out_domain; do
        for mode in source_only target_only mixed; do
            for seed in 42 123; do
                ALL_CONFIGS+=("${dist}:${domain}:${mode}:${seed}")
            done
        done
    done
done

# Track submitted jobs
SUBMITTED_FILE="${LOG_DIR}/rerun_brf_submitted.txt"
touch "$SUBMITTED_FILE"

# Tracking file for this session
SESSION_LOG="${LOG_DIR}/auto_retry_brf_$(date +%Y%m%d_%H%M%S).log"

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
    for queue in DEFAULT SINGLE SMALL LONG; do
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
    local dist=$1 domain=$2 mode=$3 seed=$4 queue=$5

    case "$mode" in
        source_only) mode_short="so" ;;
        target_only) mode_short="to" ;;
        mixed)       mode_short="mx" ;;
    esac
    case "$domain" in
        in_domain)  dom_short="in" ;;
        out_domain) dom_short="ou" ;;
    esac
    local JOB_NAME="bf_${dist:0:3}_${dom_short}_${mode_short}_s${seed}"

    local JOB_ID
    JOB_ID=$(qsub \
        -N "$JOB_NAME" \
        -q "$queue" \
        -l select=1:ncpus=${NCPUS}:mem=${MEM} \
        -l walltime=${WALLTIME} \
        -v CONDITION="$CONDITION",MODE="$mode",DISTANCE="$dist",DOMAIN="$domain",RATIO="0",SEED="$seed",N_TRIALS="$N_TRIALS",RANKING="$RANKING",RUN_EVAL="true" \
        "$SCRIPT_PATH" 2>&1)

    if [[ $? -eq 0 ]]; then
        echo "${dist}:${domain}:${mode}:${seed}" >> "$SUBMITTED_FILE"
        log "SUBMITTED: $JOB_NAME -> $queue ($JOB_ID)"
        return 0
    else
        log "FAILED: $JOB_NAME -> $queue ($JOB_ID)"
        return 1
    fi
}

log "Starting auto-retry for BalancedRF split2 rerun (36 configs)"
log "Checking every ${RETRY_INTERVAL}s for available queue slots"
log "Queues: DEFAULT(40) SINGLE(40) SMALL(30) LONG(15)"

while true; do
    # Count remaining
    remaining=0
    for config in "${ALL_CONFIGS[@]}"; do
        if ! grep -qF "$config" "$SUBMITTED_FILE" 2>/dev/null; then
            ((remaining++))
        fi
    done

    if [[ "$remaining" -eq 0 ]]; then
        log "ALL 36 JOBS SUBMITTED. Done."
        break
    fi

    log "Remaining: $remaining / ${#ALL_CONFIGS[@]}"

    # Try to submit as many as possible
    submitted_this_round=0
    for config in "${ALL_CONFIGS[@]}"; do
        # Skip already submitted
        if grep -qF "$config" "$SUBMITTED_FILE" 2>/dev/null; then
            continue
        fi

        IFS=':' read -r dist domain mode seed <<< "$config"

        # Find a queue with space
        queue=$(find_available_queue) || true
        if [[ -z "$queue" ]]; then
            log "All queues full. Waiting ${RETRY_INTERVAL}s..."
            break
        fi

        if submit_job "$dist" "$domain" "$mode" "$seed" "$queue"; then
            ((submitted_this_round++))
            sleep 0.5  # Rate limit
        fi
    done

    if [[ "$submitted_this_round" -gt 0 ]]; then
        log "Submitted $submitted_this_round jobs this round"
    fi

    # Check if all done
    remaining=0
    for config in "${ALL_CONFIGS[@]}"; do
        if ! grep -qF "$config" "$SUBMITTED_FILE" 2>/dev/null; then
            ((remaining++))
        fi
    done

    if [[ "$remaining" -eq 0 ]]; then
        log "ALL 36 JOBS SUBMITTED. Done."
        break
    fi

    sleep "$RETRY_INTERVAL"
done

log "Session complete. Submitted file: $SUBMITTED_FILE"
log "Session log: $SESSION_LOG"

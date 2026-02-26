#!/bin/bash
# ============================================================
# Auto-retry launcher for RF source_only split2 rerun
# ============================================================
# Resubmits RF source_only jobs (7 conditions × 12 configs = 84 jobs)
# with corrected source_only split (commit 889c80c).
#
# Memory: 8GB (actual usage ~1GB based on historical data)
# CPUs: 4
# Walltime: 10h (RF source_only typically finishes in <5h)
#
# Usage:
#   nohup bash scripts/hpc/jobs/domain_analysis/auto_retry_rf_source_only_split2.sh &
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

NCPUS=4
MEM="8gb"
WALLTIME="10:00:00"
SCRIPT_PATH="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"

N_TRIALS=100
RANKING="knn"
RETRY_INTERVAL=300  # 5 minutes

# Queue limits
declare -A QUEUE_MAX
QUEUE_MAX[DEFAULT]=40
QUEUE_MAX[SINGLE]=40
QUEUE_MAX[SMALL]=30
QUEUE_MAX[LONG]=15

# 7 RF conditions: condition_name:ratio
declare -a CONDITIONS=(
    "baseline:0"
    "smote_plain:0.1"
    "smote_plain:0.5"
    "smote:0.1"
    "smote:0.5"
    "undersample:0.1"
    "undersample:0.5"
)

# Map condition names to PBS CONDITION values and short names
get_condition_info() {
    local cond=$1
    local ratio=$2
    case "$cond" in
        baseline)    echo "baseline" ;;
        smote_plain) echo "smote_plain" ;;
        smote)       echo "smote" ;;          # sw_smote (subject-wise)
        undersample) echo "undersample" ;;    # undersample_rus
    esac
}

get_short_name() {
    local cond=$1
    local ratio=$2
    case "$cond" in
        baseline)    echo "bl" ;;
        smote_plain) echo "sp" ;;
        smote)       echo "sw" ;;
        undersample) echo "us" ;;
    esac
}

# Build all 84 configs: condition:ratio:distance:domain:mode:seed
declare -a ALL_CONFIGS=()
for cond_ratio in "${CONDITIONS[@]}"; do
    IFS=':' read -r cond ratio <<< "$cond_ratio"
    for dist in dtw mmd wasserstein; do
        for domain in in_domain out_domain; do
            for seed in 42 123; do
                ALL_CONFIGS+=("${cond}:${ratio}:${dist}:${domain}:source_only:${seed}")
            done
        done
    done
done

# Track submitted jobs
SUBMITTED_FILE="${LOG_DIR}/rerun_rf_so_submitted.txt"
touch "$SUBMITTED_FILE"

SESSION_LOG="${LOG_DIR}/auto_retry_rf_so_$(date +%Y%m%d_%H%M%S).log"

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
    local cond=$1 ratio=$2 dist=$3 domain=$4 mode=$5 seed=$6 queue=$7

    local short=$(get_short_name "$cond" "$ratio")
    local ratio_tag=""
    [[ "$ratio" != "0" ]] && ratio_tag="r$(echo $ratio | tr -d '.')"

    case "$domain" in
        in_domain)  dom_short="in" ;;
        out_domain) dom_short="ou" ;;
    esac

    # Job name: rf_<short>_<dist3>_<dom>_so_<ratio>_s<seed>
    local JOB_NAME="rf_${short}_${dist:0:3}_${dom_short}_so${ratio_tag:+_$ratio_tag}_s${seed}"
    # Truncate to 15 chars for PBS
    JOB_NAME="${JOB_NAME:0:15}"

    # Map condition names to PBS values
    local PBS_CONDITION
    case "$cond" in
        baseline)    PBS_CONDITION="baseline" ;;
        smote_plain) PBS_CONDITION="smote_plain" ;;
        smote)       PBS_CONDITION="smote" ;;
        undersample) PBS_CONDITION="undersample" ;;
    esac

    local JOB_ID
    JOB_ID=$(qsub \
        -N "$JOB_NAME" \
        -q "$queue" \
        -l select=1:ncpus=${NCPUS}:mem=${MEM} \
        -l walltime=${WALLTIME} \
        -v CONDITION="$PBS_CONDITION",MODE="source_only",DISTANCE="$dist",DOMAIN="$domain",RATIO="$ratio",SEED="$seed",N_TRIALS="$N_TRIALS",RANKING="$RANKING",RUN_EVAL="true" \
        "$SCRIPT_PATH" 2>&1)

    if [[ $? -eq 0 ]]; then
        echo "${cond}:${ratio}:${dist}:${domain}:source_only:${seed}" >> "$SUBMITTED_FILE"
        log "SUBMITTED: $JOB_NAME -> $queue ($JOB_ID)"
        return 0
    else
        log "FAILED: $JOB_NAME -> $queue ($JOB_ID)"
        return 1
    fi
}

log "Starting auto-retry for RF source_only split2 rerun (${#ALL_CONFIGS[@]} configs)"
log "Checking every ${RETRY_INTERVAL}s for available queue slots"
log "Memory: ${MEM}, CPUs: ${NCPUS}, Walltime: ${WALLTIME}"

while true; do
    # Count remaining
    submitted_count=$(wc -l < "$SUBMITTED_FILE")
    remaining=$((${#ALL_CONFIGS[@]} - submitted_count))

    if [[ "$remaining" -le 0 ]]; then
        log "All ${#ALL_CONFIGS[@]} configs submitted. Exiting."
        break
    fi

    log "Progress: ${submitted_count}/${#ALL_CONFIGS[@]} submitted, ${remaining} remaining"

    # Try to submit remaining configs
    for config in "${ALL_CONFIGS[@]}"; do
        # Skip if already submitted
        if grep -qF "$config" "$SUBMITTED_FILE" 2>/dev/null; then
            continue
        fi

        # Find available queue
        queue=$(find_available_queue) || {
            log "All queues full. Waiting ${RETRY_INTERVAL}s..."
            break
        }

        IFS=':' read -r cond ratio dist domain mode seed <<< "$config"
        submit_job "$cond" "$ratio" "$dist" "$domain" "$mode" "$seed" "$queue"
        sleep 1  # Small delay between submissions
    done

    # Re-check if all submitted
    submitted_count=$(wc -l < "$SUBMITTED_FILE")
    if [[ "$submitted_count" -ge "${#ALL_CONFIGS[@]}" ]]; then
        log "All ${#ALL_CONFIGS[@]} configs submitted. Exiting."
        break
    fi

    sleep "$RETRY_INTERVAL"
done

log "Daemon finished. Total submitted: $(wc -l < "$SUBMITTED_FILE")"

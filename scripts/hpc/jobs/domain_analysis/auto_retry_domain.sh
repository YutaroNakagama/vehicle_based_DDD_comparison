#!/bin/bash
# ============================================================
# Auto-retry domain analysis jobs
# ============================================================
# Monitors job queue and automatically submits remaining jobs
# when slots become available.
#
# Usage:
#   nohup bash scripts/hpc/jobs/domain_analysis/auto_retry_domain.sh &
#   # or use screen/tmux
#
# To stop: kill the process or create /tmp/stop_auto_retry
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

SCRIPT_PATH="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison.sh"
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain"
AUTO_LOG="${LOG_DIR}/auto_retry_$(date +%Y%m%d_%H%M%S).log"
STOP_FILE="/tmp/stop_auto_retry"

# Max jobs per user (conservative estimate)
MAX_JOBS=120

# Check interval (seconds)
CHECK_INTERVAL=300  # 5 minutes

# Available queues (rotate to distribute load)
QUEUES=("SINGLE" "SMALL" "LONG" "DEFAULT")
QUEUE_IDX=0

WALLTIME="168:00:00"
NCPUS=4
MEM="32gb"

DISTANCES=("dtw" "mmd" "wasserstein")
DOMAINS=("in_domain" "mid_domain" "out_domain")
MODES=("source_only" "target_only")
SEEDS=(42 123)

CONDITIONS=(
    "baseline:0"
    "smote_plain:0.1"
    "smote_plain:0.5"
    "smote:0.1"
    "smote:0.5"
    "balanced_rf:0"
    "undersample:0.1"
    "undersample:0.5"
)

N_TRIALS=150
RANKING="knn"
TOTAL_JOBS=288

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$AUTO_LOG"
}

get_current_job_count() {
    qstat -u s2240011 2>/dev/null | grep -cE "^[0-9]" || echo 0
}

get_submitted_job_ids() {
    qstat -u s2240011 2>/dev/null | grep -E "^[0-9]" | awk '{print $1}' | cut -d. -f1 | sort -n
}

# Generate all job combinations
generate_job_key() {
    local dist=$1
    local domain=$2
    local mode=$3
    local seed=$4
    local condition=$5
    local ratio=$6
    echo "${condition}_${dist}_${domain}_${mode}_s${seed}_r${ratio}"
}

submit_job() {
    local dist=$1
    local domain=$2
    local mode=$3
    local seed=$4
    local condition=$5
    local ratio=$6
    
    # Select queue (rotate)
    local queue="${QUEUES[$QUEUE_IDX]}"
    QUEUE_IDX=$(( (QUEUE_IDX + 1) % ${#QUEUES[@]} ))
    
    # Generate job name
    local job_name
    if [[ "$condition" == "baseline" || "$condition" == "balanced_rf" ]]; then
        job_name="dom_${condition:0:4}_${dist:0:3}_${domain:0:3}_${mode:0:3}_s${seed}"
    else
        job_name="dom_${condition:0:4}_r${ratio}_${dist:0:3}_${domain:0:3}_${mode:0:3}_s${seed}"
    fi
    job_name="${job_name:0:15}"
    
    local result
    result=$(qsub \
        -N "$job_name" \
        -q "$queue" \
        -l select=1:ncpus=${NCPUS}:mem=${MEM} \
        -l walltime=${WALLTIME} \
        -v CONDITION="$condition",MODE="$mode",DISTANCE="$dist",DOMAIN="$domain",RATIO="$ratio",SEED="$seed",N_TRIALS="$N_TRIALS",RANKING="$RANKING",RUN_EVAL="true" \
        "$SCRIPT_PATH" 2>&1)
    
    if [[ "$result" == *"spcc-adm1"* ]]; then
        log "  Submitted: $job_name -> $queue ($result)"
        echo "$result" >> "${LOG_DIR}/submitted_jobs_all.txt"
        return 0
    else
        return 1
    fi
}

# Track submitted jobs by checking result files
is_job_complete() {
    local dist=$1
    local domain=$2
    local mode=$3
    local seed=$4
    local condition=$5
    local ratio=$6
    
    # Check if evaluation result exists
    local pattern
    case "$condition" in
        baseline)
            pattern="baseline_domain_${RANKING}_${dist}_${domain}_${mode}_s${seed}"
            ;;
        smote)
            pattern="imbalv3_${RANKING}_${dist}_${domain}_${mode}_subjectwise_ratio${ratio}_s${seed}"
            ;;
        smote_plain)
            pattern="smote_plain_${RANKING}_${dist}_${domain}_${mode}_ratio${ratio}_s${seed}"
            ;;
        undersample)
            pattern="undersample_rus_${RANKING}_${dist}_${domain}_${mode}_ratio${ratio}_s${seed}"
            ;;
        balanced_rf)
            pattern="balanced_rf_${RANKING}_${dist}_${domain}_${mode}_s${seed}"
            ;;
    esac
    
    # Check if result file exists
    local result_file
    result_file=$(find results/outputs/evaluation -name "*${pattern}*.json" 2>/dev/null | head -1)
    [[ -n "$result_file" ]]
}

# Main loop
log "============================================================"
log "AUTO-RETRY DOMAIN ANALYSIS JOBS"
log "============================================================"
log "Total jobs: $TOTAL_JOBS"
log "Max jobs per user: $MAX_JOBS"
log "Check interval: ${CHECK_INTERVAL}s"
log "Log file: $AUTO_LOG"
log "Stop file: $STOP_FILE (create to stop)"
log "============================================================"

rm -f "$STOP_FILE"

while true; do
    # Check stop signal
    if [[ -f "$STOP_FILE" ]]; then
        log "Stop file detected. Exiting."
        break
    fi
    
    # Get current job count
    current_jobs=$(get_current_job_count)
    available_slots=$((MAX_JOBS - current_jobs))
    
    log "Current jobs: $current_jobs, Available slots: $available_slots"
    
    if [[ $available_slots -le 0 ]]; then
        log "No slots available. Waiting ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
        continue
    fi
    
    # Count completed and pending jobs
    completed=0
    pending=0
    submitted_this_round=0
    
    for dist in "${DISTANCES[@]}"; do
        for domain in "${DOMAINS[@]}"; do
            for mode in "${MODES[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    for cond_entry in "${CONDITIONS[@]}"; do
                        condition="${cond_entry%%:*}"
                        ratio="${cond_entry##*:}"
                        
                        if is_job_complete "$dist" "$domain" "$mode" "$seed" "$condition" "$ratio"; then
                            ((completed++))
                            continue
                        fi
                        
                        ((pending++))
                        
                        # Try to submit if slots available
                        if [[ $submitted_this_round -lt $available_slots ]]; then
                            if submit_job "$dist" "$domain" "$mode" "$seed" "$condition" "$ratio"; then
                                ((submitted_this_round++))
                            fi
                            sleep 0.5
                        fi
                    done
                done
            done
        done
    done
    
    log "Status: Completed=$completed, Pending=$pending, Submitted this round=$submitted_this_round"
    
    # Check if all done
    if [[ $pending -eq 0 ]]; then
        log "============================================================"
        log "ALL JOBS COMPLETED!"
        log "============================================================"
        break
    fi
    
    # Wait before next check
    log "Waiting ${CHECK_INTERVAL}s before next check..."
    sleep "$CHECK_INTERVAL"
done

log "Auto-retry finished."

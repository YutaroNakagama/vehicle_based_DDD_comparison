#!/bin/bash
# Monitor job queue and auto-submit when slots available

set -e

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

LOG_DIR="scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/monitor_submit_${TIMESTAMP}.log"
SUBMITTED_FILE="$LOG_DIR/submitted_jobs_${TIMESTAMP}.txt"
touch "$SUBMITTED_FILE"

# User job limit (set to 45 for safety)
MAX_USER_JOBS=45
CHECK_INTERVAL=60  # 1check per minute interval
MIN_SLOTS=5  # Start submission when at least 5 slots available

# Queue list (distribute submissions across more queues)
QUEUES=("SINGLE" "LONG" "DEFAULT" "SMALL" "LARGE" "XLARGE")
QUEUE_INDEX=0

echo "============================================================" | tee -a "$LOG_FILE"
echo "Prior research experiment monitoring auto-submit script" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "User limit: $MAX_USER_JOBS job(s)" | tee -a "$LOG_FILE"
echo "Check interval: ${CHECK_INTERVAL}s" | tee -a "$LOG_FILE"
echo "Min available slots: $MIN_SLOTS" | tee -a "$LOG_FILE"
echo "Queues used: ${QUEUES[*]} (round-robin)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Settings
MODELS="SvmW SvmA Lstm"
DISTANCES="mmd dtw wasserstein"
DOMAINS="out_domain in_domain"
MODES="source_only target_only"
SEEDS="42 123"
RATIOS="0.1 0.5"
RANKING="knn"
N_TRIALS=100

JOB_SCRIPT="$WORKSPACE_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

total_submitted=0
total_failed=0

# Resource acquisition function
get_resources() {
    local model=$1
    
    case $model in
        SvmW)
            echo "ncpus=8:mem=16gb 12:00:00"
            ;;
        SvmA)
            echo "ncpus=8:mem=32gb 24:00:00"
            ;;
        Lstm)
            echo "ncpus=8:mem=32gb 16:00:00"
            ;;
    esac
}

# Job submit function
submit_job() {
    local model=$1
    local condition=$2
    local distance=$3
    local domain=$4
    local mode=$5
    local seed=$6
    local ratio=$7
    
    # Generate job ID (for duplicate checking)
    local job_id="${model}_${condition}_${distance}_${domain}_${mode}_${seed}"
    if [ -n "$ratio" ]; then
        job_id="${job_id}_${ratio}"
    fi
    
    # Check if already submitted
    if grep -q "^${job_id}$" "$SUBMITTED_FILE" 2>/dev/null; then
        return 2  # Already submitted
    fi
    
    # Select queue (round-robin)
    local queue="${QUEUES[$QUEUE_INDEX]}"
    QUEUE_INDEX=$(( (QUEUE_INDEX + 1) % ${#QUEUES[@]} ))
    
    # Generate job name
    local model_abbr="${model:0:2}"
    local cond_abbr="${condition:0:2}"
    local dist_abbr="${distance:0:1}"
    local domain_abbr="${domain:0:1}"
    local mode_abbr="${mode:0:1}"
    
    if [ -n "$ratio" ]; then
        local job_name="${model_abbr}_${cond_abbr}_${dist_abbr}${domain_abbr}_${mode_abbr}_r${ratio}_s${seed}"
    else
        local job_name="${model_abbr}_${cond_abbr}_${dist_abbr}${domain_abbr}_${mode_abbr}_s${seed}"
    fi
    
    # Resource settings
    local resources=$(get_resources "$model")
    local ncpus_mem=$(echo "$resources" | awk '{print $1}')
    local walltime=$(echo "$resources" | awk '{print $2}')
    
    # Prepare environment variables
    local env_vars="MODEL=$model,CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    if [ -n "$ratio" ]; then
        env_vars="${env_vars},RATIO=$ratio"
    fi
    
    # qsubExecute command
    local qsub_cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $env_vars $JOB_SCRIPT"
    
    if output=$($qsub_cmd 2>&1); then
        echo "$job_id" >> "$SUBMITTED_FILE"
        echo "[$(date +%H:%M:%S)] [$queue] OK: $job_name → $output" | tee -a "$LOG_FILE"
        ((total_submitted++))
        return 0
    else
        echo "[$(date +%H:%M:%S)] [$queue] NG: $job_name ($output)" >> "$LOG_FILE"
        ((total_failed++))
        return 1
    fi
}

# Get current user job count
get_current_jobs() {
    qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l || echo "0"
}

# Main loop
echo "[$(date +%H:%M:%S)] job(s)Starting submission..." | tee -a "$LOG_FILE"

for model in $MODELS; do
    # balanced_rfSvmW only
    if [ "$model" = "SvmW" ]; then
        CONDITIONS="baseline smote_plain smote undersample balanced_rf"
    else
        CONDITIONS="baseline smote_plain smote undersample"
    fi
    
    for condition in $CONDITIONS; do
        for distance in $DISTANCES; do
            for domain in $DOMAINS; do
                for mode in $MODES; do
                    for seed in $SEEDS; do
                        # Check available slots before submit
                        while true; do
                            current_jobs=$(get_current_jobs)
                            available_slots=$((MAX_USER_JOBS - current_jobs))
                            
                            if [ $available_slots -ge $MIN_SLOTS ]; then
                                break
                            fi
                            
                            echo "[$(date +%H:%M:%S)] Waiting... (current: $current_jobs/$MAX_USER_JOBS, available: $available_slots)" | tee -a "$LOG_FILE"
                            sleep $CHECK_INTERVAL
                        done
                        
                        if [ "$condition" = "baseline" ]; then
                            # baselinehas no ratio
                            submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" ""
                            sleep 0.2
                        else
                            # Other conditions have a ratio
                            for ratio in $RATIOS; do
                                # Check available slots before each submit
                                current_jobs=$(get_current_jobs)
                                available_slots=$((MAX_USER_JOBS - current_jobs))
                                
                                while [ $available_slots -lt $MIN_SLOTS ]; do
                                    echo "[$(date +%H:%M:%S)] Waiting... (current: $current_jobs/$MAX_USER_JOBS, available: $available_slots)" | tee -a "$LOG_FILE"
                                    sleep $CHECK_INTERVAL
                                    current_jobs=$(get_current_jobs)
                                    available_slots=$((MAX_USER_JOBS - current_jobs))
                                done
                                
                                submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" "$ratio"
                                sleep 0.2
                            done
                        fi
                    done
                done
            done
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "All job submission processing is complete" | tee -a "$LOG_FILE"
echo "Submission succeeded: $total_submitted job(s)" | tee -a "$LOG_FILE"
echo "Submission failed: $total_failed job(s)" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE"
echo "Submitted list: $SUBMITTED_FILE"

# Display submission count per queue
echo "" | tee -a "$LOG_FILE"
echo "Submissions per queue:" | tee -a "$LOG_FILE"
for queue in "${QUEUES[@]}"; do
    count=$(grep -c "\[$queue\] OK:" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "  $queue: $count job(s)" | tee -a "$LOG_FILE"
done

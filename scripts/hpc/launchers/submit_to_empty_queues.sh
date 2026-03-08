#!/bin/bash
# Aggressively submit to available queues

set -e

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

LOG_DIR="scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/submit_empty_queues_${TIMESTAMP}.log"
SUBMITTED_FILE="$LOG_DIR/submitted_jobs_${TIMESTAMP}.txt"
touch "$SUBMITTED_FILE"

echo "============================================================" | tee -a "$LOG_FILE"
echo "Aggressive empty-queue submit script" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
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

# Preferred queues (by availability)
PRIORITY_QUEUES=("DEFAULT" "SMALL" "LARGE" "XLARGE" "LONG" "SINGLE")
queue_index=0

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
    
    # Check if already submitted (search all log files)
    if grep -q "^${job_id}$" "$LOG_DIR"/submitted_jobs_*.txt 2>/dev/null; then
        return 2  # Already submitted
    fi
    
    # Select queue (round-robin)
    local queue="${PRIORITY_QUEUES[$queue_index]}"
    queue_index=$(( (queue_index + 1) % ${#PRIORITY_QUEUES[@]} ))
    
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
        echo "[$(date +%H:%M:%S)] [$queue] ✓ $job_name → $output" | tee -a "$LOG_FILE"
        ((total_submitted++))
        return 0
    else
        if [[ "$output" == *"would exceed"* ]]; then
            echo "[$(date +%H:%M:%S)] [$queue] ✗ $job_name (limit reached)" >> "$LOG_FILE"
            return 1  # Exit on limit error
        else
            echo "[$(date +%H:%M:%S)] [$queue] ✗ $job_name ($output)" >> "$LOG_FILE"
            ((total_failed++))
            return 1
        fi
    fi
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
                        if [ "$condition" = "baseline" ]; then
                            # baselinehas no ratio
                            submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" "" || {
                                ret=$?
                                if [ $ret -eq 1 ]; then
                                    echo "[$(date +%H:%M:%S)] User limit reached. Stopping submission." | tee -a "$LOG_FILE"
                                    break 6
                                fi
                            }
                            sleep 0.1
                        else
                            # Other conditions have a ratio
                            for ratio in $RATIOS; do
                                submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" "$ratio" || {
                                    ret=$?
                                    if [ $ret -eq 1 ]; then
                                        echo "[$(date +%H:%M:%S)] User limit reached. Stopping submission." | tee -a "$LOG_FILE"
                                        break 7
                                    fi
                                }
                                sleep 0.1
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
echo "Submit processing complete" | tee -a "$LOG_FILE"
echo "Submission succeeded: $total_submitted job(s)" | tee -a "$LOG_FILE"
echo "Submission failed: $total_failed job(s)" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Display submission count per queue
echo "" | tee -a "$LOG_FILE"
echo "Submissions per queue:" | tee -a "$LOG_FILE"
for queue in "${PRIORITY_QUEUES[@]}"; do
    count=$(grep -c "\[$queue\] ✓" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "  $queue: $count job(s)" | tee -a "$LOG_FILE"
done

echo ""
echo "Log file: $LOG_FILE"
echo "Submitted list: $SUBMITTED_FILE"

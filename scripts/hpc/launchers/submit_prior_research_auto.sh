#!/bin/bash
# Prior research experiment auto-submission script
# Monitors queue availability and submits sequentially

set -e

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

LOG_DIR="scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/auto_submit_prior_${TIMESTAMP}.log"

# Job limit set (set to 48 for safety)
MAX_JOBS=48
CHECK_INTERVAL=300  # 5check per minute interval

echo "============================================================" | tee -a "$LOG_FILE"
echo "Prior research experiment auto-submit script" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "Job limit: $MAX_JOBS" | tee -a "$LOG_FILE"
echo "Check interval: ${CHECK_INTERVAL}s" | tee -a "$LOG_FILE"
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

# Record submitted jobs
SUBMITTED_JOBS_FILE="$LOG_DIR/submitted_prior_jobs.txt"
touch "$SUBMITTED_JOBS_FILE"

total_submitted=0
total_skipped=0
total_errors=0

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
    if grep -q "^${job_id}$" "$SUBMITTED_JOBS_FILE" 2>/dev/null; then
        return 1  # Already submitted
    fi
    
    # Generate job name
    local model_abbr="${model:0:2}"
    local cond_abbr="${condition:0:2}"
    local dist_abbr="${distance:0:1}"
    local domain_abbr="${domain:0:1}"
    local mode_abbr="${mode:0:1}"
    local job_name="${model_abbr}_${cond_abbr}_${dist_abbr}${domain_abbr}_${mode_abbr}_s${seed}"
    
    # Resource settings
    local walltime mem
    case $model in
        SvmW)
            walltime="12:00:00"
            mem="16gb"
            ;;
        SvmA)
            walltime="24:00:00"
            mem="32gb"
            ;;
        Lstm)
            walltime="16:00:00"
            mem="32gb"
            ;;
    esac
    
    # Prepare environment variables
    local env_vars="MODEL=$model,CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    if [ -n "$ratio" ]; then
        env_vars="${env_vars},RATIO=$ratio"
    fi
    
    # qsubExecute command
    local qsub_cmd="qsub -N $job_name -l select=1:ncpus=8:mem=$mem -l walltime=$walltime -q SINGLE -v $env_vars $JOB_SCRIPT"
    
    if $qsub_cmd >> "$LOG_FILE" 2>&1; then
        echo "$job_id" >> "$SUBMITTED_JOBS_FILE"
        echo "[$(date +%H:%M:%S)] Submission succeeded: $job_id" | tee -a "$LOG_FILE"
        ((total_submitted++))
        return 0
    else
        echo "[$(date +%H:%M:%S)] Submission failed: $job_id" | tee -a "$LOG_FILE"
        ((total_errors++))
        return 1
    fi
}

# Main loop
while true; do
    # Check current job count
    current_jobs=$(qstat -u s2240011 2>/dev/null | grep -E "R|Q" | wc -l || echo "0")
    available_slots=$((MAX_JOBS - current_jobs))
    
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date +%H:%M:%S)] current job count: $current_jobs / $MAX_JOBS (available: $available_slots)" | tee -a "$LOG_FILE"
    
    if [ $available_slots -le 0 ]; then
        echo "[$(date +%H:%M:%S)] Queue is full.${CHECK_INTERVAL}Retrying after seconds..." | tee -a "$LOG_FILE"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # Submit as many jobs as slots allow
    submitted_in_round=0
    
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
                                if submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" ""; then
                                    ((submitted_in_round++))
                                fi
                            else
                                # Other conditions have a ratio
                                for ratio in $RATIOS; do
                                    if submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" "$ratio"; then
                                        ((submitted_in_round++))
                                    fi
                                    
                                    # Break if submit limit reached
                                    if [ $submitted_in_round -ge $available_slots ]; then
                                        break 5
                                    fi
                                done
                            fi
                            
                            # Break if submit limit reached
                            if [ $submitted_in_round -ge $available_slots ]; then
                                break 4
                            fi
                        done
                    done
                done
            done
        done
    done
    
    echo "[$(date +%H:%M:%S)] Submitted in this round: $submitted_in_round job(s)" | tee -a "$LOG_FILE"
    
    # Check whether all jobs have been submitted
    total_expected=552
    total_done=$((total_submitted + total_skipped))
    
    if [ $total_done -ge $total_expected ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "============================================================" | tee -a "$LOG_FILE"
        echo "All job submissions are complete！" | tee -a "$LOG_FILE"
        echo "Submission succeeded: $total_submitted job(s)" | tee -a "$LOG_FILE"
        echo "Skipped: $total_skipped job(s)" | tee -a "$LOG_FILE"
        echo "Error: $total_errors job(s)" | tee -a "$LOG_FILE"
        echo "End time: $(date)" | tee -a "$LOG_FILE"
        echo "============================================================" | tee -a "$LOG_FILE"
        break
    fi
    
    # Wait until next check
    if [ $submitted_in_round -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] No more jobs to submit.${CHECK_INTERVAL}Retrying after seconds..." | tee -a "$LOG_FILE"
        sleep $CHECK_INTERVAL
    else
        # Wait briefly after submitting job
        sleep 10
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE"
echo "Submitted job list: $SUBMITTED_JOBS_FILE"

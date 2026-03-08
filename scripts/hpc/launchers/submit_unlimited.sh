#!/bin/bash
# DEFAULT/SMALLBulk submit to queues (utilizing unlimited queues)

set -e

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

LOG_DIR="scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/submit_unlimited_queues_${TIMESTAMP}.log"
SUBMITTED_FILE="$LOG_DIR/submitted_unlimited_${TIMESTAMP}.txt"
touch "$SUBMITTED_FILE"

echo "============================================================" | tee -a "$LOG_FILE"
echo "Bulk submit to unlimited queues (DEFAULT/SMALL)" | tee -a "$LOG_FILE"
echo "started: $(date)" | tee -a "$LOG_FILE"
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

# Use only unlimited queues
QUEUES=("DEFAULT" "SMALL")
queue_index=0

total_submitted=0
total_failed=0
total_skipped=0

# Resource acquisition
get_resources() {
    local model=$1
    case $model in
        SvmW) echo "ncpus=8:mem=16gb 12:00:00" ;;
        SvmA) echo "ncpus=8:mem=32gb 24:00:00" ;;
        Lstm) echo "ncpus=8:mem=32gb 16:00:00" ;;
    esac
}

# job(s)submit
submit_job() {
    local model=$1 condition=$2 distance=$3 domain=$4 mode=$5 seed=$6 ratio=$7
    
    # Duplicate check (all log files)
    local job_id="${model}_${condition}_${distance}_${domain}_${mode}_${seed}"
    [ -n "$ratio" ] && job_id="${job_id}_${ratio}"
    
    if grep -rq "^${job_id}$" "$LOG_DIR"/submitted_*.txt 2>/dev/null; then
        ((total_skipped++))
        return 2
    fi
    
    # Queue selection (alternating DEFAULT/SMALL)
    local queue="${QUEUES[$queue_index]}"
    queue_index=$(( (queue_index + 1) % 2 ))
    
    # job(s)subjects
    local model_abbr="${model:0:2}"
    local cond_abbr="${condition:0:2}"
    local dist_abbr="${distance:0:1}"
    local domain_abbr="${domain:0:1}"
    local mode_abbr="${mode:0:1}"
    
    local job_name="${model_abbr}_${cond_abbr}_${dist_abbr}${domain_abbr}_${mode_abbr}"
    [ -n "$ratio" ] && job_name="${job_name}_r${ratio}" || job_name="${job_name}"
    job_name="${job_name}_s${seed}"
    
    # Resources
    local resources=$(get_resources "$model")
    local ncpus_mem=$(echo "$resources" | awk '{print $1}')
    local walltime=$(echo "$resources" | awk '{print $2}')
    
    # Environment variables
    local env_vars="MODEL=$model,CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    [ -n "$ratio" ] && env_vars="${env_vars},RATIO=$ratio"
    
    # submit
    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $env_vars $JOB_SCRIPT"
    
    if output=$($cmd 2>&1); then
        echo "$job_id" >> "$SUBMITTED_FILE"
        echo "[$(date +%H:%M:%S)] [$queue] ✓ $job_name → $output" | tee -a "$LOG_FILE"
        ((total_submitted++))
        return 0
    else
        if [[ "$output" == *"would exceed"* ]]; then
            echo "[Limit reached] Pausing (retrying after 60s)" | tee -a "$LOG_FILE"
            sleep 60
            # retry
            if output=$($cmd 2>&1); then
                echo "$job_id" >> "$SUBMITTED_FILE"
                echo "[$(date +%H:%M:%S)] [$queue] ✓ $job_name → $output (retry)" | tee -a "$LOG_FILE"
                ((total_submitted++))
                return 0
            fi
        fi
        echo "[$(date +%H:%M:%S)] [$queue] ✗ $job_name" >> "$LOG_FILE"
        ((total_failed++))
        return 1
    fi
}

# Main loop
echo "[$(date +%H:%M:%S)] Starting submission..." | tee -a "$LOG_FILE"

for model in $MODELS; do
    [ "$model" = "SvmW" ] && CONDITIONS="baseline smote_plain smote undersample balanced_rf" || CONDITIONS="baseline smote_plain smote undersample"
    
    for condition in $CONDITIONS; do
        for distance in $DISTANCES; do
            for domain in $DOMAINS; do
                for mode in $MODES; do
                    for seed in $SEEDS; do
                        if [ "$condition" = "baseline" ]; then
                            submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" ""
                        else
                            for ratio in $RATIOS; do
                                submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" "$ratio"
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
echo "Submission complete" | tee -a "$LOG_FILE"
echo "succeeded: $total_submitted | failed: $total_failed | Skipped: $total_skipped" | tee -a "$LOG_FILE"
echo "Ended: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "DEFAULT: $(grep -c '\[DEFAULT\] ✓' "$LOG_FILE" || echo 0) job(s)" | tee -a "$LOG_FILE"
echo "SMALL: $(grep -c '\[SMALL\] ✓' "$LOG_FILE" || echo 0) job(s)" | tee -a "$LOG_FILE"
echo ""
echo "Log: $LOG_FILE"

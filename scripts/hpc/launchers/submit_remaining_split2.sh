#!/bin/bash
# ============================================================
# Auto-submit remaining split2 jobs script
# ============================================================
# 29 jobs already submitted, remaining 67 jobs to submit
#
# Usage:
#   ./submit_remaining_split2.sh         # Normal execution
#   ./submit_remaining_split2.sh --check  # Only check submittable count
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
LOG_FILE="$PROJECT_ROOT/scripts/hpc/logs/domain/remaining_split2_$(date +%Y%m%d_%H%M%S).log"

# Mode selection
CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=true
fi

# Check current job count
current_jobs=$(qstat -u s2240011 2>/dev/null | grep -c "s2240011" || echo "0")
echo "Current jobs in queue: $current_jobs"

# Get user limit (typically 50)
user_limit=$(qstat -Qf SINGLE 2>/dev/null | grep "max_user_run" | head -1 | awk '{print $NF}' || echo "50")
echo "User job limit: $user_limit"

available_slots=$((user_limit - current_jobs))
echo "Available slots: $available_slots"

if $CHECK_ONLY; then
    echo "Check-only mode. Exiting."
    exit 0
fi

if [[ $available_slots -le 0 ]]; then
    echo "No available slots. Please wait for running jobs to complete."
    exit 1
fi

echo ""
echo "Will submit up to $available_slots jobs..."
echo "Log: $LOG_FILE"
echo ""

# Initialize log
{
    echo "# Remaining split2 jobs submission"
    echo "# Started at $(date)"
    echo "# Available slots: $available_slots"
    echo ""
} > "$LOG_FILE"

submitted=0
skipped=0

# Helper function to submit a job
submit_job() {
    local condition="$1"
    local mode="$2"
    local distance="$3"
    local domain="$4"
    local seed="$5"
    local ratio="${6:-}"
    
    # Resource config
    case "$condition" in
        balanced_rf)
            ncpus_mem="ncpus=8:mem=12gb"
            walltime="08:00:00"
            queue="LONG"
            ;;
        smote|smote_plain)
            ncpus_mem="ncpus=4:mem=10gb"
            walltime="08:00:00"
            queue="SINGLE"
            ;;
        baseline|undersample)
            ncpus_mem="ncpus=4:mem=8gb"
            walltime="06:00:00"
            queue="SINGLE"
            ;;
    esac
    
    # Job name
    cond_short="${condition:0:2}"
    job_name="${cond_short}_${distance:0:1}${domain:0:1}_${mode:0:1}"
    if [[ -n "$ratio" ]]; then
        job_name="${job_name}_r${ratio}_s${seed}"
    else
        job_name="${job_name}_s${seed}"
    fi
    
    # Build command
    cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue"
    if [[ -n "$ratio" ]]; then
        cmd="$cmd -v CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,RATIO=$ratio,SEED=$seed,N_TRIALS=100,RANKING=knn,RUN_EVAL=true"
    else
        cmd="$cmd -v CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=100,RANKING=knn,RUN_EVAL=true"
    fi
    cmd="$cmd $JOB_SCRIPT"
    
    # Submit
    if job_id=$(eval "$cmd" 2>&1); then
        echo "[OK] $condition | $distance | $domain | $mode | r=${ratio:-N/A} | s$seed → $job_id"
        echo "$condition:$distance:$domain:$mode:${ratio:-N/A}:$seed:$job_id" >> "$LOG_FILE"
        ((submitted++))
        return 0
    else
        echo "[FAIL] $condition | $distance | $domain | $mode | r=${ratio:-N/A} | s$seed"
        ((skipped++))
        return 1
    fi
}

# Submit remaining jobs (based on what failed before)
# Priority: mmd/in_domain and dtw/wasserstein jobs that failed

# mmd in_domain target_only r=0.5 s=123 (3 jobs failed)
for cond in "smote_plain" "smote" "undersample"; do
    if [[ $submitted -ge $available_slots ]]; then break; fi
    submit_job "$cond" "target_only" "mmd" "out_domain" "123" "0.5"
    sleep 0.3
done

# All baseline jobs (24 failed)
for distance in "mmd" "dtw" "wasserstein"; do
    for domain in "in_domain" "out_domain"; do
        for mode in "source_only" "target_only"; do
            for seed in "42" "123"; do
                if [[ $submitted -ge $available_slots ]]; then break 4; fi
                submit_job "baseline" "$mode" "$distance" "$domain" "$seed"
                sleep 0.2
            done
        done
    done
done

# All other failed jobs
for distance in "mmd" "dtw" "wasserstein"; do
    for domain in "in_domain" "out_domain"; do
        for mode in "source_only" "target_only"; do
            for seed in "42" "123"; do
                for ratio in "0.1" "0.5"; do
                    for cond in "smote_plain" "smote" "undersample"; do
                        if [[ $submitted -ge $available_slots ]]; then break 6; fi
                        # Skip already submitted mmd/out_domain jobs
                        if [[ "$distance" == "mmd" && "$domain" == "out_domain" ]]; then
                            continue
                        fi
                        submit_job "$cond" "$mode" "$distance" "$domain" "$seed" "$ratio"
                        sleep 0.2
                    done
                done
                
                # balanced_rf
                if [[ $submitted -ge $available_slots ]]; then break 4; fi
                # Skip already submitted mmd/out_domain jobs
                if [[ "$distance" == "mmd" && "$domain" == "out_domain" ]]; then
                    continue
                fi
                submit_job "balanced_rf" "$mode" "$distance" "$domain" "$seed"
                sleep 0.2
            done
        done
    done
done

{
    echo ""
    echo "# Completed at $(date)"
    echo "# Submitted: $submitted"
    echo "# Skipped: $skipped"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "Submitted: $submitted jobs"
echo "Skipped: $skipped jobs"
echo "Log: $LOG_FILE"
echo "============================================================"

#!/bin/bash
# =============================================================================
# monitor_and_eval_imbalv3.sh - Monitor training and submit evaluation
# =============================================================================
# Usage: nohup bash scripts/hpc/domain_analysis/imbalance/monitor_and_eval_imbalv3.sh &
# =============================================================================

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

TRAIN_JOBS=("14609456" "14609457" "14609458")
EVAL_SCRIPT="scripts/hpc/domain_analysis/imbalance/pbs_eval_imbalv3_optimized.sh"

echo "=== imbalv3 Training/Evaluation Monitor ==="
echo "Training Jobs: ${TRAIN_JOBS[*]}"
echo "Started: $(date)"
echo ""

# Function to check if a job array is complete
check_job_complete() {
    local jobid="$1"
    local status=$(qstat -t "${jobid}[]" 2>/dev/null | grep -c "R\|Q\|H\|B")
    echo "$status"
}

# Wait for all training jobs to complete
while true; do
    all_complete=true
    
    for jobid in "${TRAIN_JOBS[@]}"; do
        running=$(check_job_complete "$jobid")
        if [[ "$running" -gt 0 ]]; then
            all_complete=false
            echo "[$(date '+%H:%M:%S')] Job $jobid: $running tasks remaining"
        else
            echo "[$(date '+%H:%M:%S')] Job $jobid: COMPLETE"
        fi
    done
    
    if $all_complete; then
        echo ""
        echo "=== All training complete! Submitting evaluation jobs... ==="
        break
    fi
    
    sleep 300  # Check every 5 minutes
done

# Submit evaluation jobs
echo ""
echo "Submitting evaluation jobs to TINY queue..."

qsub -J 1-12 -q TINY "$EVAL_SCRIPT"
qsub -J 13-24 -q TINY "$EVAL_SCRIPT"
qsub -J 25-36 -q TINY "$EVAL_SCRIPT"

echo ""
echo "Evaluation jobs submitted!"
echo "Finished: $(date)"

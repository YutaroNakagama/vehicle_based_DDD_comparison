#!/bin/bash
# ============================================================
# Auto-submit remaining job when queue space opens
# Runs in background, checks every 5 minutes
# ============================================================

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison.sh"
LOG_FILE="$PROJECT_ROOT/scripts/hpc/launchers/auto_submit.log"
MAX_JOBS=126

echo "[$(date)] Auto-submit daemon started" >> "$LOG_FILE"

while true; do
    current_jobs=$(qstat -u $USER 2>/dev/null | tail -n +6 | wc -l)
    
    if [ "$current_jobs" -lt "$MAX_JOBS" ]; then
        echo "[$(date)] Queue space available ($current_jobs < $MAX_JOBS). Submitting..." >> "$LOG_FILE"
        
        result=$(qsub -N "swdous_s123" -l select=1:ncpus=4:mem=8gb -l walltime=12:00:00 -q SINGLE \
            -v CONDITION=smote,MODE=source_only,DISTANCE=dtw,DOMAIN=out_domain,RATIO=0.5,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true \
            "$JOB_SCRIPT" 2>&1)
        
        if [[ "$result" != *"would exceed"* && "$result" != *"error"* ]]; then
            echo "[$(date)] SUCCESS: $result" >> "$LOG_FILE"
            echo "[$(date)] Daemon completed successfully" >> "$LOG_FILE"
            exit 0
        else
            echo "[$(date)] Submit failed: $result" >> "$LOG_FILE"
        fi
    else
        echo "[$(date)] Queue full ($current_jobs jobs). Waiting..." >> "$LOG_FILE"
    fi
    
    sleep 300  # 5minwaiting
done

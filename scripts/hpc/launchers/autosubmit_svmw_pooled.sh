#!/bin/bash
# =============================================================================
# Auto-submit SvmW pooled retrain+eval when queue has capacity
# Polls every 60 seconds until total job count drops below 125
# =============================================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
PBS_SCRIPT="$PROJECT_ROOT/scripts/hpc/launchers/retrain_eval_svmw_pooled.sh"
LOG="$PROJECT_ROOT/scripts/hpc/logs/train/svmw_pooled_autosubmit.log"

echo "[$(date)] Waiting for queue capacity to submit SvmW pooled retrain..." | tee -a "$LOG"

while true; do
    TOTAL=$(qstat -u s2240011 2>/dev/null | grep -cE " R | Q " || echo 0)
    
    if [[ "$TOTAL" -lt 125 ]]; then
        echo "[$(date)] Queue has $TOTAL jobs (< 125). Submitting..." | tee -a "$LOG"
        
        RESULT=$(qsub "$PBS_SCRIPT" 2>&1)
        EXIT=$?
        
        if [[ $EXIT -eq 0 ]]; then
            echo "[$(date)] SUCCESS: Submitted as $RESULT" | tee -a "$LOG"
            exit 0
        else
            echo "[$(date)] FAILED: $RESULT (exit=$EXIT). Will retry in 60s..." | tee -a "$LOG"
        fi
    else
        echo "[$(date)] Queue full ($TOTAL jobs). Waiting 60s..." >> "$LOG"
    fi
    
    sleep 60
done

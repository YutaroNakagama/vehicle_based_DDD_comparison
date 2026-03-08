#!/bin/bash
# Continuously monitor available queues and submit

set -e

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

LOG_DIR="scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/continuous_submit_${TIMESTAMP}.log"

# User limit (with safety margin)
MAX_JOBS=45
CHECK_INTERVAL=30  # 30per second interval

echo "============================================================" | tee -a "$LOG_FILE"
echo "Continuous monitoring submit script" | tee -a "$LOG_FILE"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo "User limit: $MAX_JOBS" | tee -a "$LOG_FILE"
echo "Check interval: ${CHECK_INTERVAL}s" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

total_rounds=0
total_submitted=0

while true; do
    ((total_rounds++))
    
    # Check current job count
    current_jobs=$(qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l || echo "50")
    available=$((MAX_JOBS - current_jobs))
    
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date +%H:%M:%S)] Round #$total_rounds" | tee -a "$LOG_FILE"
    echo "  current: $current_jobs/$MAX_JOBS, available: $available" | tee -a "$LOG_FILE"
    
    if [ $available -le 0 ]; then
        echo "  → Queue full. Waiting ${CHECK_INTERVAL}s..." | tee -a "$LOG_FILE"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # Attempt job submission if slots available
    echo "  → Attempting submission..." | tee -a "$LOG_FILE"
    
    output=$(bash "$WORKSPACE_ROOT/scripts/hpc/launchers/submit_to_empty_queues.sh" 2>&1)
    submitted=$(echo "$output" | grep "Submission succeeded:" | grep -oE "[0-9]+" | head -1 || echo "0")
    
    if [ "$submitted" -gt 0 ]; then
        echo "  → $submitted job(s)Submission succeeded" | tee -a "$LOG_FILE"
        ((total_submitted += submitted))
        # Wait briefly after submission succeeded
        sleep 5
    else
        echo "  → No new submissions (all submitted or limit reached)" | tee -a "$LOG_FILE"
        
        # Check submitted job count
        total_expected=552
        total_done=$(cat "$LOG_DIR"/submitted_jobs_*.txt 2>/dev/null | sort -u | wc -l || echo "0")
        
        echo "  → Progress: $total_done / $total_expected job(s)" | tee -a "$LOG_FILE"
        
        if [ "$total_done" -ge "$total_expected" ]; then
            echo "" | tee -a "$LOG_FILE"
            echo "============================================================" | tee -a "$LOG_FILE"
            echo "All jobs submitted!" | tee -a "$LOG_FILE"
            echo "Total submitted: $total_submitted jobs (this session)" | tee -a "$LOG_FILE"
            echo "End time: $(date)" | tee -a "$LOG_FILE"
            echo "============================================================" | tee -a "$LOG_FILE"
            break
        fi
        
        sleep $CHECK_INTERVAL
    fi
done

echo ""
echo "Log file: $LOG_FILE"

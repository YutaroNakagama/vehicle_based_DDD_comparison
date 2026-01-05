#!/bin/bash
LOG_FILE="/home/ynakagama/git/work/vehicle_based_DDD_comparison/scripts/local/logs/domain/domain_parallel_20260104_060109.log"
MONITOR_LOG="/home/ynakagama/git/work/vehicle_based_DDD_comparison/scripts/local/logs/domain/monitor.log"

echo "=== Domain Experiment Monitor Started ===" | tee -a "$MONITOR_LOG"
echo "Checking every 30 minutes..." | tee -a "$MONITOR_LOG"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    RUNNING=$(ps aux | grep train.py | grep -v grep | wc -l)
    PARENT=$(pgrep -f "run_domain_parallel" | wc -l)
    DONE_COUNT=$(grep -c "\[DONE\]" "$LOG_FILE" 2>/dev/null || echo 0)
    FAIL_COUNT=$(grep -c "\[FAIL\]" "$LOG_FILE" 2>/dev/null || echo 0)
    LAST_LINE=$(tail -1 "$LOG_FILE" 2>/dev/null)
    
    echo "[$TIMESTAMP] Parent: ${PARENT} | Running: ${RUNNING} | Done: ${DONE_COUNT} | Failed: ${FAIL_COUNT}" | tee -a "$MONITOR_LOG"
    echo "  Last: ${LAST_LINE}" | tee -a "$MONITOR_LOG"
    
    if [[ "$PARENT" -eq 0 && "$RUNNING" -eq 0 ]]; then
        echo "[$TIMESTAMP] === ALL EXPERIMENTS COMPLETED ===" | tee -a "$MONITOR_LOG"
        break
    fi
    
    sleep 1800  # 30 minutes
done

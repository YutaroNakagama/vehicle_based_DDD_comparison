#!/bin/bash
# Simple continuous submission script

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

MAX_JOBS=45
CHECK_INTERVAL=30

echo "========================================="
echo "Continuous submit script started"
echo "started: $(date)"
echo "========================================="

round=0
while true; do
    ((round++))
    
    current=$(qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l || echo "50")
    available=$((MAX_JOBS - current))
    
    echo ""
    echo "[$(date +%H:%M:%S)] #$round current=$current available=$available"
    
    if [ $available -le 0 ]; then
        echo "  full → waiting"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    echo "  Attempting submission..."
    bash "$WORKSPACE_ROOT/scripts/hpc/launchers/submit_to_empty_queues.sh" 2>&1 | grep -E "Submission succeeded|Submission failed|Per queue" | head -5
    
    sleep $CHECK_INTERVAL
done

#!/bin/bash
# シンプル継続投入スクリプト

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

MAX_JOBS=45
CHECK_INTERVAL=30

echo "========================================="
echo "継続投入スクリプト起動"
echo "開始: $(date)"
echo "========================================="

round=0
while true; do
    ((round++))
    
    current=$(qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l || echo "50")
    available=$((MAX_JOBS - current))
    
    echo ""
    echo "[$(date +%H:%M:%S)] #$round 現在=$current 空き=$available"
    
    if [ $available -le 0 ]; then
        echo "  満杯 → 待機"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    echo "  投入試行..."
    bash "$WORKSPACE_ROOT/scripts/hpc/launchers/submit_to_empty_queues.sh" 2>&1 | grep -E "投入成功|投入失敗|キュー別" | head -5
    
    sleep $CHECK_INTERVAL
done

#!/bin/bash
# 継続的に空きキューを監視して投入

set -e

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

LOG_DIR="scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/continuous_submit_${TIMESTAMP}.log"

# ユーザー上限（安全マージン込み）
MAX_JOBS=45
CHECK_INTERVAL=30  # 30秒ごと

echo "============================================================" | tee -a "$LOG_FILE"
echo "継続監視投入スクリプト" | tee -a "$LOG_FILE"
echo "開始時刻: $(date)" | tee -a "$LOG_FILE"
echo "ユーザー上限: $MAX_JOBS" | tee -a "$LOG_FILE"
echo "チェック間隔: ${CHECK_INTERVAL}秒" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

total_rounds=0
total_submitted=0

while true; do
    ((total_rounds++))
    
    # 現在のジョブ数を確認
    current_jobs=$(qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l || echo "50")
    available=$((MAX_JOBS - current_jobs))
    
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date +%H:%M:%S)] ラウンド #$total_rounds" | tee -a "$LOG_FILE"
    echo "  現在: $current_jobs/$MAX_JOBS, 空き: $available" | tee -a "$LOG_FILE"
    
    if [ $available -le 0 ]; then
        echo "  → キュー満杯。${CHECK_INTERVAL}秒待機..." | tee -a "$LOG_FILE"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # 空きがあればジョブ投入を試行
    echo "  → 投入試行..." | tee -a "$LOG_FILE"
    
    output=$(bash "$WORKSPACE_ROOT/scripts/hpc/launchers/submit_to_empty_queues.sh" 2>&1)
    submitted=$(echo "$output" | grep "投入成功:" | grep -oE "[0-9]+" | head -1 || echo "0")
    
    if [ "$submitted" -gt 0 ]; then
        echo "  → $submitted ジョブ投入成功" | tee -a "$LOG_FILE"
        ((total_submitted += submitted))
        # 投入成功したら少し待つ
        sleep 5
    else
        echo "  → 新規投入なし（全て投入済みまたは上限）" | tee -a "$LOG_FILE"
        
        # 投入済みジョブ数を確認
        total_expected=552
        total_done=$(cat "$LOG_DIR"/submitted_jobs_*.txt 2>/dev/null | sort -u | wc -l || echo "0")
        
        echo "  → 進捗: $total_done / $total_expected ジョブ" | tee -a "$LOG_FILE"
        
        if [ "$total_done" -ge "$total_expected" ]; then
            echo "" | tee -a "$LOG_FILE"
            echo "============================================================" | tee -a "$LOG_FILE"
            echo "全ジョブ投入完了！" | tee -a "$LOG_FILE"
            echo "総投入数: $total_submitted ジョブ（このセッション）" | tee -a "$LOG_FILE"
            echo "終了時刻: $(date)" | tee -a "$LOG_FILE"
            echo "============================================================" | tee -a "$LOG_FILE"
            break
        fi
        
        sleep $CHECK_INTERVAL
    fi
done

echo ""
echo "ログファイル: $LOG_FILE"

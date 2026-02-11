#!/bin/bash
# ============================================================
# 自動再投入スクリプト — キュー空き待ち & 差分投入
# ============================================================
# /tmp/remaining_jobs.txt から未投入ジョブを読み取り、
# キューに空きがあれば投入する。
# 全件投入完了するまで5分間隔でリトライする。
#
# 使い方:
#   nohup bash scripts/hpc/launchers/auto_resub_remaining.sh &
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
REMAINING_FILE="/tmp/remaining_jobs.txt"
N_TRIALS=100
RANKING="knn"
QUEUES=("SINGLE" "LONG" "DEFAULT")
QUEUE_COUNTER=0
SLEEP_INTERVAL=300  # 5分間隔
MAX_RETRIES=100     # 最大100回（約8時間）

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
LOG_FILE="$LOG_DIR/auto_resub_${TIMESTAMP}.log"

echo "============================================================"
echo "自動再投入スクリプト開始: $(date)"
echo "残りジョブ: $(wc -l < "$REMAINING_FILE") 件"
echo "リトライ間隔: ${SLEEP_INTERVAL}秒"
echo "ログ: $LOG_FILE"
echo "============================================================"

echo "# Auto resub started at $(date)" > "$LOG_FILE"

RETRY=0
TOTAL_SUBMITTED=0

while [[ -s "$REMAINING_FILE" && $RETRY -lt $MAX_RETRIES ]]; do
    ((RETRY++))
    ROUND_SUBMITTED=0
    ROUND_SKIPPED=0

    # 一時ファイルに未投入分を書き出し
    cp "$REMAINING_FILE" "${REMAINING_FILE}.bak"
    > "${REMAINING_FILE}.new"

    while IFS='|' read -r MODEL CONDITION MODE DISTANCE DOMAIN SEED RATIO WALLTIME MEM; do
        QUEUE="${QUEUES[$((QUEUE_COUNTER % 3))]}"
        ((QUEUE_COUNTER++))

        case "$MODE" in
            source_only) MODE_SHORT="s" ;;
            target_only) MODE_SHORT="t" ;;
            mixed)       MODE_SHORT="m" ;;
        esac
        COND_SHORT="${CONDITION:0:2}"
        JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE_SHORT}_s${SEED}"

        CMD="qsub -N $JOB_NAME -l select=1:ncpus=8:mem=${MEM} -l walltime=${WALLTIME} -q $QUEUE"
        CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        if [[ -n "$RATIO" ]]; then
            CMD="$CMD,RATIO=$RATIO"
        fi
        CMD="$CMD $JOB_SCRIPT"

        JOB_ID=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[$(date +%H:%M:%S)] SUBMIT $MODEL $CONDITION $MODE $DISTANCE $DOMAIN s$SEED r${RATIO:-N/A} → $JOB_ID ($QUEUE)"
            echo "$MODEL:$CONDITION:$MODE:$DISTANCE:$DOMAIN:$SEED:${RATIO:-}:$QUEUE:$JOB_ID" >> "$LOG_FILE"
            ((ROUND_SUBMITTED++))
            ((TOTAL_SUBMITTED++))
            sleep 0.2
        else
            # 投入失敗 → 再投入リストに残す
            echo "$MODEL|$CONDITION|$MODE|$DISTANCE|$DOMAIN|$SEED|$RATIO|$WALLTIME|$MEM" >> "${REMAINING_FILE}.new"
            ((ROUND_SKIPPED++))
        fi
    done < "${REMAINING_FILE}.bak"

    # 残りリストを更新
    mv "${REMAINING_FILE}.new" "$REMAINING_FILE"
    REMAINING=$(wc -l < "$REMAINING_FILE")

    echo "[$(date +%H:%M:%S)] Round $RETRY: submitted=$ROUND_SUBMITTED skipped=$ROUND_SKIPPED remaining=$REMAINING total=$TOTAL_SUBMITTED"
    echo "# Round $RETRY at $(date): submitted=$ROUND_SUBMITTED skipped=$ROUND_SKIPPED remaining=$REMAINING" >> "$LOG_FILE"

    if [[ $REMAINING -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] 全件投入完了！"
        break
    fi

    if [[ $ROUND_SUBMITTED -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] この回は投入できず。${SLEEP_INTERVAL}秒後にリトライ..."
        sleep $SLEEP_INTERVAL
    else
        echo "[$(date +%H:%M:%S)] 一部投入成功。30秒後に残りをリトライ..."
        sleep 30
    fi
done

{
    echo ""
    echo "# Auto resub completed at $(date)"
    echo "# Total submitted: $TOTAL_SUBMITTED"
    echo "# Remaining: $(wc -l < "$REMAINING_FILE")"
    echo "# Retries: $RETRY"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "自動再投入完了: $(date)"
echo "投入済み: $TOTAL_SUBMITTED"
echo "残り: $(wc -l < "$REMAINING_FILE")"
echo "リトライ回数: $RETRY"
echo "ログ: $LOG_FILE"
echo "============================================================"

#!/bin/bash
# ============================================================
# SvmA Arefnezhad2019対応 自動投入スクリプト
# ============================================================
# /tmp/remaining_svma_arefnezhad2019.txt から未投入SvmAジョブを読み取り、
# 各キューの空き枠を確認しながら投入する。
#
# 入力形式 (パイプ区切り, 10列):
#   MODEL|CONDITION|MODE|DISTANCE|DOMAIN|SEED|RATIO|WALLTIME|MEM|N_TRIALS
#
# 使い方:
#   nohup bash scripts/hpc/launchers/auto_resub_svma_arefnezhad2019.sh &
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
REMAINING_FILE="/tmp/remaining_svma_arefnezhad2019.txt"
RANKING="knn"
SLEEP_INTERVAL=300  # 5分間隔
MAX_RETRIES=2000

USER="s2240011"

# Per-user queue limits
declare -A QUEUE_MAX
QUEUE_MAX[SINGLE]=40
QUEUE_MAX[DEFAULT]=40
QUEUE_MAX[LONG]=15
QUEUE_MAX[SMALL]=30

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/auto_resub_svma_arefnezhad2019_${TIMESTAMP}.log"

echo "============================================================"
echo "SvmA Arefnezhad2019対応 自動投入 開始: $(date)"
echo "残りジョブ: $(wc -l < "$REMAINING_FILE") 件"
echo "リトライ間隔: ${SLEEP_INTERVAL}秒"
echo "ログ: $LOG_FILE"
echo "============================================================"

echo "# SvmA Arefnezhad2019 auto resub started at $(date)" > "$LOG_FILE"

RETRY=0
TOTAL_SUBMITTED=0

while [[ -s "$REMAINING_FILE" && $RETRY -lt $MAX_RETRIES ]]; do
    ((RETRY++))
    ROUND_SUBMITTED=0
    ROUND_DEFERRED=0

    # Get per-queue availability
    declare -A AVAIL
    for q in SINGLE DEFAULT LONG SMALL; do
        current=$(qstat -u "$USER" 2>/dev/null | grep "$USER" | awk -v queue="$q" '$3==queue' | wc -l)
        max="${QUEUE_MAX[$q]}"
        avail=$((max - current))
        [[ $avail -lt 0 ]] && avail=0
        AVAIL[$q]=$avail
    done

    TOTAL_AVAIL=$(( ${AVAIL[SINGLE]} + ${AVAIL[DEFAULT]} + ${AVAIL[LONG]} + ${AVAIL[SMALL]} ))
    echo "[$(date +%H:%M:%S)] Round $RETRY: slots SINGLE=${AVAIL[SINGLE]} DEFAULT=${AVAIL[DEFAULT]} LONG=${AVAIL[LONG]} SMALL=${AVAIL[SMALL]} (total=$TOTAL_AVAIL)"

    if [[ $TOTAL_AVAIL -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] 全キュー満杯。${SLEEP_INTERVAL}秒待機..."
        echo "# Round $RETRY: all queues full" >> "$LOG_FILE"
        sleep $SLEEP_INTERVAL
        continue
    fi

    # Build ordered queue list
    ORDERED_QUEUES=()
    for q in SINGLE DEFAULT SMALL LONG; do
        [[ ${AVAIL[$q]} -gt 0 ]] && ORDERED_QUEUES+=("$q")
    done
    QUEUE_IDX=0

    cp "$REMAINING_FILE" "${REMAINING_FILE}.bak"
    > "${REMAINING_FILE}.new"

    while IFS='|' read -r MODEL CONDITION MODE DISTANCE DOMAIN SEED RATIO WALLTIME MEM N_TRIALS; do
        N_TRIALS="${N_TRIALS:-100}"

        # Find a queue with available slots
        SELECTED_QUEUE=""
        TRIED=0
        while [[ $TRIED -lt ${#ORDERED_QUEUES[@]} ]]; do
            local_q="${ORDERED_QUEUES[$((QUEUE_IDX % ${#ORDERED_QUEUES[@]}))]}"
            if [[ ${AVAIL[$local_q]} -gt 0 ]]; then
                SELECTED_QUEUE="$local_q"
                ((QUEUE_IDX++))
                break
            fi
            ((QUEUE_IDX++))
            ((TRIED++))
        done

        if [[ -z "$SELECTED_QUEUE" ]]; then
            echo "$MODEL|$CONDITION|$MODE|$DISTANCE|$DOMAIN|$SEED|$RATIO|$WALLTIME|$MEM|$N_TRIALS" >> "${REMAINING_FILE}.new"
            ((ROUND_DEFERRED++))
            continue
        fi

        case "$MODE" in
            source_only) MODE_SHORT="s" ;;
            target_only) MODE_SHORT="t" ;;
            mixed)       MODE_SHORT="m" ;;
        esac
        COND_SHORT="${CONDITION:0:2}"
        JOB_NAME="Sa_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE_SHORT}_s${SEED}"

        CMD="qsub -N $JOB_NAME -l select=1:ncpus=8:mem=${MEM} -l walltime=${WALLTIME} -q $SELECTED_QUEUE"
        CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        if [[ -n "$RATIO" ]]; then
            CMD="$CMD,RATIO=$RATIO"
        fi
        CMD="$CMD $JOB_SCRIPT"

        JOB_ID=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[$(date +%H:%M:%S)] OK [$SELECTED_QUEUE] $CONDITION $MODE $DISTANCE $DOMAIN s$SEED r${RATIO:-N/A} → $JOB_ID"
            echo "$CONDITION:$MODE:$DISTANCE:$DOMAIN:$SEED:${RATIO:-}:$SELECTED_QUEUE:$JOB_ID" >> "$LOG_FILE"
            ((ROUND_SUBMITTED++))
            ((TOTAL_SUBMITTED++))
            ((AVAIL[$SELECTED_QUEUE]--))
            sleep 0.2
        else
            echo "$MODEL|$CONDITION|$MODE|$DISTANCE|$DOMAIN|$SEED|$RATIO|$WALLTIME|$MEM|$N_TRIALS" >> "${REMAINING_FILE}.new"
            ((ROUND_DEFERRED++))
            ((AVAIL[$SELECTED_QUEUE]--)) 2>/dev/null || true
            ORDERED_QUEUES=()
            for q in SINGLE DEFAULT SMALL LONG; do
                [[ ${AVAIL[$q]} -gt 0 ]] && ORDERED_QUEUES+=("$q")
            done
            [[ ${#ORDERED_QUEUES[@]} -eq 0 ]] && break
        fi
    done < "${REMAINING_FILE}.bak"

    mv "${REMAINING_FILE}.new" "$REMAINING_FILE"
    REMAINING=$(wc -l < "$REMAINING_FILE")

    echo "[$(date +%H:%M:%S)] Round $RETRY: submitted=$ROUND_SUBMITTED deferred=$ROUND_DEFERRED remaining=$REMAINING total=$TOTAL_SUBMITTED"
    echo "# Round $RETRY at $(date): submitted=$ROUND_SUBMITTED deferred=$ROUND_DEFERRED remaining=$REMAINING" >> "$LOG_FILE"

    if [[ $REMAINING -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] 全件投入完了！"
        break
    fi

    if [[ $ROUND_SUBMITTED -eq 0 ]]; then
        sleep $SLEEP_INTERVAL
    else
        sleep 60
    fi
done

{
    echo ""
    echo "# SvmA Arefnezhad2019 auto resub completed at $(date)"
    echo "# Total submitted: $TOTAL_SUBMITTED"
    echo "# Remaining: $(wc -l < "$REMAINING_FILE")"
    echo "# Retries: $RETRY"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "SvmA Arefnezhad2019 自動投入完了: $(date)"
echo "投入済み: $TOTAL_SUBMITTED"
echo "残り: $(wc -l < "$REMAINING_FILE")"
echo "ログ: $LOG_FILE"
echo "============================================================"

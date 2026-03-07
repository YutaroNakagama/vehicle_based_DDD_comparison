#!/bin/bash
# ============================================================
# 自動投入デーモン: 実験3の未投入ジョブを空き次第投入
# ============================================================
# 使い方: nohup bash scripts/hpc/jobs/domain_analysis/auto_submit_exp3_missing.sh &
#
# キューに空きができ次第、以下を順番に投入:
#   1. SvmA 18ジョブ (CPU: SINGLE/LONG/DEFAULT)
#   2. Lstm 38ジョブ (GPU: GPU-1A)
#   3. Lstm eval-only 12タスク (GPU: GPU-1, PBS array)

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

MAX_JOBS=148
CHECK_INTERVAL=300

LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/auto_submit_exp3_missing_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
get_job_count() { qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l; }

log "============================================================"
log "実験3 未投入ジョブ自動投入デーモン開始"
log "ジョブ上限: $MAX_JOBS / チェック間隔: ${CHECK_INTERVAL}秒"
log "============================================================"

# Extract qsub commands from the generated scripts
QSUB_CMDS=()

# SvmA commands
while IFS= read -r line; do
    QSUB_CMDS+=("$line")
done < <(grep '^qsub ' scripts/hpc/jobs/domain_analysis/submit_missing_svma_exp3.sh)

SVMA_COUNT=${#QSUB_CMDS[@]}
log "SvmA コマンド数: $SVMA_COUNT"

# Lstm train commands
while IFS= read -r line; do
    QSUB_CMDS+=("$line")
done < <(grep '^qsub ' scripts/hpc/jobs/domain_analysis/submit_missing_lstm_train_exp3.sh)

LSTM_TRAIN_COUNT=$((${#QSUB_CMDS[@]} - SVMA_COUNT))
log "Lstm train コマンド数: $LSTM_TRAIN_COUNT"

TOTAL=${#QSUB_CMDS[@]}
log "合計投入予定数: $TOTAL"

IDX=0
SUBMITTED=0
FAILED=0

while [[ $IDX -lt $TOTAL ]]; do
    CURRENT=$(get_job_count)
    AVAIL=$((MAX_JOBS - CURRENT))

    if [[ $AVAIL -le 0 ]]; then
        log "キュー満杯: $CURRENT/$MAX_JOBS — ${CHECK_INTERVAL}秒後に再チェック (残り: $((TOTAL - IDX)))"
        sleep "$CHECK_INTERVAL"
        continue
    fi

    # Submit as many as we can
    BATCH=0
    while [[ $IDX -lt $TOTAL ]] && [[ $AVAIL -gt 0 ]]; do
        CMD="${QSUB_CMDS[$IDX]}"
        JOB_NAME=$(echo "$CMD" | grep -oP '(?<=-N )\S+')

        JOB_ID=$(eval "$CMD" 2>&1)
        RC=$?
        if [[ $RC -eq 0 ]]; then
            log "  [OK] $JOB_NAME → $JOB_ID"
            ((SUBMITTED++))
            ((AVAIL--))
            ((BATCH++))
        else
            log "  [FAIL] $JOB_NAME: $JOB_ID"
            ((FAILED++))
            # If quota error, stop this batch and wait
            if echo "$JOB_ID" | grep -q "exceed"; then
                log "  キュー制限到達 — 待機"
                break
            fi
        fi
        ((IDX++))
        sleep 0.3
    done

    log "投入: $SUBMITTED/$TOTAL, 失敗: $FAILED, バッチ: $BATCH"

    if [[ $IDX -lt $TOTAL ]]; then
        sleep "$CHECK_INTERVAL"
    fi
done

# Submit Lstm eval-only PBS array
log ""
log "=== Lstm eval-only PBS array 投入 ==="
EVAL_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_eval_missing_lstm_exp3.sh"
if [[ -f "$EVAL_SCRIPT" ]]; then
    while true; do
        CURRENT=$(get_job_count)
        AVAIL=$((MAX_JOBS - CURRENT))
        if [[ $AVAIL -gt 0 ]]; then
            JOB_ID=$(qsub "$EVAL_SCRIPT" 2>&1)
            RC=$?
            if [[ $RC -eq 0 ]]; then
                log "  [OK] Lstm eval array → $JOB_ID"
            else
                log "  [FAIL] $JOB_ID"
            fi
            break
        fi
        log "  キュー満杯 — ${CHECK_INTERVAL}秒後に再チェック"
        sleep "$CHECK_INTERVAL"
    done
else
    log "  [SKIP] eval script not found: $EVAL_SCRIPT"
fi

log ""
log "============================================================"
log "完了: 投入 $SUBMITTED, 失敗 $FAILED"
log "============================================================"

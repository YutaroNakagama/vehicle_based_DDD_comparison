#!/bin/bash
# ============================================================
# Eval再実行ランチャー (2026-02-10 updated)
# training完了済み・eval失敗のSvmA(91)+SvmW(87)=178件を再評価
# 修正内容:
#   - .str.strip() bug (SvmA_eval) — commit 6c5f14c で修正済
#   - glob bracket escaping [1] dirs — commit e5a499d で修正
#   - scaler None null guard — commit e5a499d で修正
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
EVAL_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/evaluate/pbs_eval_only.sh"
EVAL_LIST="$HOME/tmp/eval_rerun_list_v2.txt"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_eval_rerun_${TIMESTAMP}.log"

JOB_COUNT=0
FAIL_COUNT=0

echo "============================================================"
echo "  Eval再実行 — $(date)"
echo "  Dry run : $DRY_RUN"
echo "============================================================"

{
    echo "# Eval rerun launched at $(date)"
    echo ""
} > "$LOG_FILE"

while IFS='|' read -r JOBID MODEL TAG MODE SEED TARGET_FILE; do
    job_name="ev_${JOBID}"
    vars="MODEL=$MODEL,TAG=$TAG,MODE=$MODE,JOBID=$JOBID,SEED=${SEED:-42}"
    if [[ -n "${TARGET_FILE:-}" ]]; then
        vars="$vars,TARGET_FILE=$TARGET_FILE"
    fi
    cmd="qsub -N $job_name -l select=1:ncpus=2:mem=8gb -l walltime=02:00:00 -q SEMINAR -v $vars $EVAL_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] $JOBID | $TAG | $MODE"
        JOB_COUNT=$((JOB_COUNT+1))
    else
        job_result=$(eval "$cmd" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[SUBMIT] $JOBID | $TAG → $job_result"
            echo "$JOBID|$TAG|$MODE|$job_result" >> "$LOG_FILE"
            JOB_COUNT=$((JOB_COUNT+1))
        else
            echo "[FAIL]   $JOBID | $TAG — $job_result"
            FAIL_COUNT=$((FAIL_COUNT+1))
        fi
        sleep 0.1
    fi
done < "$EVAL_LIST"

{
    echo ""
    echo "# Completed at $(date)"
    echo "# Submitted: $JOB_COUNT  Failed: $FAIL_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Submitted: $JOB_COUNT   Failed: $FAIL_COUNT"
echo "  Log: $LOG_FILE"
echo "============================================================"

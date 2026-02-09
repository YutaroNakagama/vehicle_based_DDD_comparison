#!/bin/bash
# ============================================================
# Eval再実行ランチャー (2026-02-07)
# training完了済み・eval失敗の23件を再評価
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
EVAL_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/evaluate/pbs_eval_only.sh"
EVAL_LIST="/tmp/eval_rerun_list.txt"

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

while IFS='|' read -r JOBID MODEL TAG MODE; do
    job_name="ev_${JOBID}"
    vars="MODEL=$MODEL,TAG=$TAG,MODE=$MODE,JOBID=$JOBID"
    cmd="qsub -N $job_name -l select=1:ncpus=2:mem=4gb -l walltime=00:30:00 -q SEMINAR -v $vars $EVAL_SCRIPT"

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

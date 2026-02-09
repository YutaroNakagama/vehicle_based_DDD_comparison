#!/bin/bash
# ============================================================
# Lstm Eval Batch Submitter (2026-02-07)
# ============================================================
# Submit PBS jobs for all Lstm training results that have no eval
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

PBS_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/eval/pbs_lstm_eval.sh"
SUBMIT_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

echo "============================================================"
echo "  Lstm Eval Batch Submitter"
echo "  $(date)"
echo "============================================================"

while IFS= read -r train_json; do
    [[ -z "$train_json" ]] && continue

    base_dir=$(dirname "$train_json")
    jobid_dir=$(dirname "$base_dir")
    JOBID=$(basename "$jobid_dir")

    fname=$(basename "$train_json" .json)
    rest="${fname#train_results_Lstm_}"
    MODE="${rest%%_prior_*}"
    TAG="${rest#${MODE}_}"

    # Skip if already evaluated
    EVAL_DIR="$PROJECT_ROOT/results/outputs/evaluation/Lstm/$JOBID"
    if find "$EVAL_DIR" -name "eval_results_*.json" -print -quit 2>/dev/null | grep -q .; then
        ((SKIP_COUNT++))
        continue
    fi

    # Truncate job name to 15 chars
    JOB_NAME="eL_${MODE:0:3}_${JOBID: -4}"

    if qsub -N "$JOB_NAME" \
        -v "MODEL=Lstm,MODE=$MODE,TAG=$TAG,JOBID=$JOBID" \
        "$PBS_SCRIPT" 2>&1; then
        ((SUBMIT_COUNT++))
    else
        echo "  [FAIL] Lstm | $MODE | jobid=$JOBID"
        ((FAIL_COUNT++))
    fi

    # Small delay to avoid overwhelming PBS
    sleep 0.1

done < <(find "$PROJECT_ROOT/results/outputs/training/Lstm" -name "*prior*split2*.json" 2>/dev/null)

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Submitted: $SUBMIT_COUNT"
echo "  Skipped:   $SKIP_COUNT"
echo "  Failed:    $FAIL_COUNT"
echo "============================================================"

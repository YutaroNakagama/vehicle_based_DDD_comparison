#!/bin/bash
#PBS -N reeval_all_split2
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/domain/
#PBS -q SMALL
#PBS -l select=1:ncpus=16:mem=64gb
#PBS -l walltime=06:00:00
#PBS -M yutaro.nakagama@bosch.com
#PBS -m ae

# ============================================================
# Bulk re-evaluation of all post-bug split2 results.
# Single job, internal 16-way parallel via xargs -P.
# Reads scripts/hpc/jobs/domain_analysis/reeval_all.tsv.
# Each row: CONDITION MODE DIST DOMAIN RATIO SEED TRAIN_JOBID TAG
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TSV="${PROJECT_ROOT}/scripts/hpc/jobs/domain_analysis/reeval_all.tsv"
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain/reeval_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[$(date)] start; TSV=$TSV LOG_DIR=$LOG_DIR"
echo "rows: $(($(wc -l < "$TSV") - 1))"

# Per-row worker
run_one() {
    local CONDITION="$1" MODE="$2" DIST="$3" DOMAIN="$4" RATIO="$5" SEED="$6" TRAIN_JOBID="$7" TAG="$8"
    [[ "$RATIO" == "NA" ]] && RATIO=""
    local TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/knn/${DIST}_${DOMAIN}.txt"
    local LOG="${LOG_DIR}/eval_${CONDITION}_${MODE}_${DIST}_${DOMAIN}_r${RATIO:-NA}_s${SEED}.log"

    if [[ ! -f "$TARGET_FILE" ]]; then
        echo "MISSING $TARGET_FILE" > "$LOG"
        return 1
    fi

    local MODEL="RF"
    [[ "$CONDITION" == "balanced_rf" ]] && MODEL="BalancedRF"

    python3 scripts/python/evaluation/evaluate.py \
        --model "$MODEL" \
        --tag "$TAG" \
        --mode "$MODE" \
        --target_file "$TARGET_FILE" \
        --seed "$SEED" \
        --jobid "$TRAIN_JOBID" \
        --subject_wise_split \
        > "$LOG" 2>&1
}
export -f run_one
export LOG_DIR

# Drop header, dispatch each row to a worker via xargs (16-way parallel).
# We pass row as a single TAB-separated string (one per -n 1) and split inside the worker.
tail -n +2 "$TSV" | xargs -d '\n' -n 1 -P 16 -I{} bash -c '
    IFS=$'"'"'\t'"'"' read -r C M D DOM R S J T <<< "$1"
    run_one "$C" "$M" "$D" "$DOM" "$R" "$S" "$J" "$T"
' _ {}

echo "[$(date)] all eval invocations done"

# Summary: count successful (lines containing 'EVAL DONE') vs failed
OK=$(grep -l "EVAL DONE" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
TOTAL=$(ls "$LOG_DIR"/*.log 2>/dev/null | wc -l)
FAIL=$((TOTAL - OK))
echo "Result: ok=$OK fail=$FAIL total=$TOTAL"
echo "Logs: $LOG_DIR"

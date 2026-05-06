#!/bin/bash
# Parameterised bulk re-eval (queue + TSV via env vars). Used for multi-queue distribution.
# Env: TSV_PATH (required), PARALLEL (default 16)
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/domain/
#PBS -l walltime=04:00:00
#PBS -l select=1:ncpus=16:mem=64gb
#PBS -M yutaro.nakagama@bosch.com
#PBS -m ae
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

TSV="${TSV_PATH:?TSV_PATH env var required}"
PARALLEL="${PARALLEL:-16}"
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain/reeval_$(basename "${TSV%.tsv}")_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "[$(date)] start; TSV=$TSV PARALLEL=$PARALLEL LOG_DIR=$LOG_DIR"
echo "rows: $(($(wc -l < "$TSV") - 1))"

run_one() {
    local CONDITION="$1" MODE="$2" DIST="$3" DOMAIN="$4" RATIO="$5" SEED="$6" TRAIN_JOBID="$7" TAG="$8"
    [[ "$RATIO" == "NA" ]] && RATIO=""
    local TARGET_FILE="results/analysis/exp2_domain_shift/distance/rankings/split2/knn/${DIST}_${DOMAIN}.txt"
    local LOG="${LOG_DIR}/eval_${CONDITION}_${MODE}_${DIST}_${DOMAIN}_r${RATIO:-NA}_s${SEED}.log"
    if [[ ! -f "$TARGET_FILE" ]]; then echo "MISSING $TARGET_FILE" > "$LOG"; return 1; fi
    local MODEL="RF"
    [[ "$CONDITION" == "balanced_rf" ]] && MODEL="BalancedRF"
    python3 scripts/python/evaluation/evaluate.py \
        --model "$MODEL" --tag "$TAG" --mode "$MODE" \
        --target_file "$TARGET_FILE" --seed "$SEED" --jobid "$TRAIN_JOBID" \
        --subject_wise_split > "$LOG" 2>&1
}
export -f run_one
export LOG_DIR

tail -n +2 "$TSV" | xargs -d '\n' -I{} -P "$PARALLEL" bash -c '
    IFS=$'"'"'\t'"'"' read -r C M D DOM R S J T <<< "$1"
    run_one "$C" "$M" "$D" "$DOM" "$R" "$S" "$J" "$T"
' _ {}

OK=$(grep -l "EVAL DONE" "$LOG_DIR"/*.log 2>/dev/null | wc -l)
TOTAL=$(ls "$LOG_DIR"/*.log 2>/dev/null | wc -l)
echo "[$(date)] done. ok=$OK total=$TOTAL logs=$LOG_DIR"

#!/bin/bash
# ============================================================
# Re-submission daemon — Lstm split2 missing eval conditions
# ============================================================
# Covers the 79 source_only/target_only jobs whose eval failed
# with scaler feature-mismatch errors (old code, Feb 2026).
# Training outputs were moved to:
#   results/outputs/training/Lstm/_invalidated_old_code/
# This daemon resubmits full retrain+eval via split2 GPU script.
#
# Usage:
#   nohup bash scripts/hpc/launchers/rerun_lstm_split2_missing.sh > /tmp/rerun_lstm_split2_missing.log 2>&1 &
#   bash scripts/hpc/launchers/rerun_lstm_split2_missing.sh --dry-run
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2_gpu.sh"
TASK_FILE="$PROJECT_ROOT/scripts/hpc/logs/train/task_files/lstm_split2_missing_resub.tsv"
SUBMITTED_KEYS="/tmp/rerun_lstm_split2_missing_keys.txt"
LOG="/tmp/rerun_lstm_split2_missing.log"
POLL_INTERVAL=300
N_TRIALS=100
RANKING="knn"

trap 'echo "[$(date +%H:%M)] TRAP: daemon exiting (line $LINENO, exit=$?)" >> "$LOG"' EXIT
trap 'echo "[$(date +%H:%M)] TRAP: signal received, exiting" >> "$LOG"; exit 1' INT TERM HUP

declare -A GPU_QUEUE_MAX=( [GPU-1]=15 [GPU-1A]=10 [GPU-S]=10 [GPU-L]=2 [GPU-LA]=2 )
GPU_QUEUES=("GPU-L" "GPU-LA" "GPU-1" "GPU-1A" "GPU-S")
declare -A QUEUE_CURRENT=()

touch "$SUBMITTED_KEYS"

# ---- Check if a canonical eval result already exists ----
has_eval_result() {
    local cond="$1" dist="$2" dom="$3" mode="$4" seed="$5" ratio="$6"
    local eval_dir="results/outputs/evaluation/Lstm"

    local patterns=()
    if [[ "$cond" == "baseline" ]]; then
        patterns+=("eval_results_Lstm_${mode}_prior_Lstm_baseline_knn_${dist}_${dom}_${mode}_split2_s${seed}.json")
    elif [[ "$cond" == "smote_plain" ]]; then
        patterns+=("eval_results_Lstm_${mode}_prior_Lstm_smote_plain_knn_${dist}_${dom}_${mode}_split2_ratio${ratio}_s${seed}.json")
    elif [[ "$cond" == "smote" ]]; then
        patterns+=("eval_results_Lstm_${mode}_prior_Lstm_imbalv3_knn_${dist}_${dom}_${mode}_split2_subjectwise_ratio${ratio}_s${seed}.json")
    elif [[ "$cond" == "undersample" ]]; then
        patterns+=("eval_results_Lstm_${mode}_prior_Lstm_undersample_rus_knn_${dist}_${dom}_${mode}_split2_ratio${ratio}_s${seed}.json")
    fi

    for pat in "${patterns[@]}"; do
        find "$eval_dir" -name "$pat" 2>/dev/null | grep -v _invalidated | grep -q . && return 0
    done
    return 1
}

get_resources() {
    local mode="$1"
    if [[ "$mode" == "mixed" ]]; then
        echo "ncpus=4:ngpus=1:mem=8gb 12:00:00"
    else
        echo "ncpus=4:ngpus=1:mem=8gb 08:00:00"
    fi
}

get_queue_counts() {
    local qstat_output
    qstat_output=$(qstat -u s2240011 2>/dev/null | tail -n +6 || true)
    for q in "${GPU_QUEUES[@]}"; do
        QUEUE_CURRENT[$q]=$(echo "$qstat_output" | awk -v q="$q" '$3==q' | wc -l || echo 0)
    done
}

find_available_gpu_queue() {
    for q in "${GPU_QUEUES[@]}"; do
        local current="${QUEUE_CURRENT[$q]:-0}"
        local max="${GPU_QUEUE_MAX[$q]:-0}"
        if (( current < max )); then
            echo "$q"; return 0
        fi
    done
    return 1
}

# ---- Load conditions from TSV (skip header) ----
mapfile -t CONDITION_ROWS < <(tail -n +2 "$TASK_FILE")
TOTAL=${#CONDITION_ROWS[@]}

echo "[$(date +%H:%M)] rerun_lstm_split2_missing daemon started. Total: $TOTAL conditions" | tee -a "$LOG"
$DRY_RUN && echo "[DRY-RUN mode]" | tee -a "$LOG"

while true; do
    get_queue_counts || true
    SUBMITTED_THIS_ROUND=0

    for row in "${CONDITION_ROWS[@]}"; do
        IFS=$'\t' read -r COND DIST DOM MODE SEED RATIO <<< "$row"
        [[ -z "$COND" ]] && continue

        KEY="Lstm:${COND}:${DIST}:${DOM}:${MODE}:s${SEED}:r${RATIO:-none}"

        if has_eval_result "$COND" "$DIST" "$DOM" "$MODE" "$SEED" "${RATIO:-0.5}"; then
            continue
        fi

        if grep -qF "$KEY" "$SUBMITTED_KEYS" 2>/dev/null; then
            continue
        fi

        QUEUE=""
        QUEUE=$(find_available_gpu_queue) || true
        if [[ -z "$QUEUE" ]]; then
            continue
        fi

        RES=$(get_resources "$MODE")
        NCPUS_MEM=$(echo "$RES" | cut -d' ' -f1)
        WALLTIME=$(echo "$RES" | cut -d' ' -f2)

        # Job name: Ls_<cond2>_<d><o>_<m2>_s<SEED>
        C2="${COND:0:2}"; M2="${MODE:0:2}"; D1="${DIST:0:1}"; O1="${DOM:0:1}"
        JOB_NAME="Ls_${C2}_${D1}${O1}_${M2}_s${SEED}"
        [[ -n "${RATIO:-}" ]] && JOB_NAME="${JOB_NAME}_r${RATIO}"

        VARS="MODEL=Lstm,CONDITION=$COND,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        [[ -n "${RATIO:-}" ]] && VARS="${VARS},RATIO=$RATIO"

        if $DRY_RUN; then
            echo "[DRY] $KEY → $QUEUE | $WALLTIME" | tee -a "$LOG"
            echo "$KEY" >> "$SUBMITTED_KEYS"
            ((SUBMITTED_THIS_ROUND++)) || true
            continue
        fi

        JOB_OUT=$(qsub -N "$JOB_NAME" \
            -l "select=1:${NCPUS_MEM}" \
            -l "walltime=${WALLTIME}" \
            -q "$QUEUE" \
            -v "$VARS" \
            "$JOB_SCRIPT" 2>&1) || {
            echo "  [ERR] qsub failed for $KEY: $JOB_OUT" | tee -a "$LOG"
            continue
        }

        JOB_NUM="${JOB_OUT%%.*}"
        echo "$KEY" >> "$SUBMITTED_KEYS"
        QUEUE_CURRENT[$QUEUE]=$(( ${QUEUE_CURRENT[$QUEUE]:-0} + 1 ))
        ((SUBMITTED_THIS_ROUND++)) || true
        echo "  [SUB] $KEY → $QUEUE $JOB_NUM" | tee -a "$LOG"
        sleep 0.3
    done

    # Count truly remaining
    REMAINING=0
    for row in "${CONDITION_ROWS[@]}"; do
        IFS=$'\t' read -r COND DIST DOM MODE SEED RATIO <<< "$row"
        [[ -z "$COND" ]] && continue
        has_eval_result "$COND" "$DIST" "$DOM" "$MODE" "$SEED" "${RATIO:-0.5}" || ((REMAINING++)) || true
    done

    if (( REMAINING == 0 )); then
        echo "[$(date +%H:%M)] ALL DONE – all $TOTAL conditions have eval results." | tee -a "$LOG"
        break
    fi

    GPU_TOTAL=0
    for q in "${GPU_QUEUES[@]}"; do GPU_TOTAL=$(( GPU_TOTAL + ${QUEUE_CURRENT[$q]:-0} )); done
    echo "[POLL $(date +%H:%M)] gpu_q=$GPU_TOTAL submitted=$SUBMITTED_THIS_ROUND remaining=$REMAINING" | tee -a "$LOG"

    $DRY_RUN && break
    sleep "$POLL_INTERVAL"
done

echo "[$(date +%H:%M)] Daemon finished." | tee -a "$LOG"

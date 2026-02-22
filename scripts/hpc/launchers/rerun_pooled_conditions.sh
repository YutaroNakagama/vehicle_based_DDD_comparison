#!/bin/bash
# ============================================================
# 再実行デーモン — Pooled 条件別ジョブ投入
# ============================================================
# 問題: pbs_prior_research.sh (pooled) は CONDITION パラメータがなく、
#        常に baseline として訓練していた。そのため smote_plain,
#        sw_smote (imbalv3), undersample_rus の pooled 結果が
#        全て baseline と同一だった。
#
# 修正: pbs_prior_research.sh に CONDITION 対応追加済み。
#        このデーモンで全モデル × 全条件 × 全シードの pooled ジョブを投入。
#
# 対象:
#   - 3モデル (SvmW, SvmA, Lstm) × 4条件 × 2シード = 24 configs
#   - baseline は新しいタグ形式で再訓練 (旧ファイルは無効化済み)
#
# Usage:
#   nohup bash scripts/hpc/launchers/rerun_pooled_conditions.sh &
#   # ログ: /tmp/rerun_pooled_conditions.log
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research.sh"
LOG="/tmp/rerun_pooled_conditions.log"
SUBMITTED_KEYS="/tmp/rerun_pooled_conditions_keys.txt"
POLL_INTERVAL=300  # 5 minutes

# ---- エラートラップ ----
trap 'echo "[$(date +%H:%M)] TRAP: daemon exiting (line $LINENO, exit=$?)" >> "$LOG"' EXIT
trap 'echo "[$(date +%H:%M)] TRAP: received signal, exiting" >> "$LOG"; exit 1' INT TERM HUP

# ---- キュー制限 ----
declare -A QUEUE_MAX=( [SINGLE]=40 [DEFAULT]=40 [SMALL]=30 [LONG]=15 )
declare -A QUEUE_CURRENT=()
CPU_QUEUES=("SINGLE" "DEFAULT" "SMALL" "LONG")

# GPU queues for Lstm
declare -A GPU_QUEUE_MAX=( [GPU-1]=4 [GPU-1A]=4 [GPU-S]=4 [GPU-L]=4 [GPU-LA]=4 )
declare -A GPU_QUEUE_CURRENT=()
GPU_QUEUES=("GPU-1" "GPU-1A" "GPU-S" "GPU-L" "GPU-LA")

touch "$SUBMITTED_KEYS"

# ---- 新しい評価結果が存在するか確認 ----
has_eval_result() {
    local model="$1" cond="$2" seed="$3" ratio="$4"
    local eval_dir="results/outputs/evaluation/${model}"

    local pattern
    case "$cond" in
        baseline)
            pattern="eval_results_${model}_pooled_prior_${model}_baseline_s${seed}"
            ;;
        smote_plain)
            pattern="eval_results_${model}_pooled_prior_${model}_smote_plain_ratio${ratio}_s${seed}"
            ;;
        smote)
            pattern="eval_results_${model}_pooled_prior_${model}_imbalv3_subjectwise_ratio${ratio}_s${seed}"
            ;;
        undersample)
            pattern="eval_results_${model}_pooled_prior_${model}_undersample_rus_ratio${ratio}_s${seed}"
            ;;
    esac

    # Check if matching file exists (excluding invalidated)
    find "$eval_dir" -name "${pattern}*.json" 2>/dev/null | grep -v _invalidated | grep -q .
}

# ---- キュー状態確認 ----
get_queue_counts() {
    local qstat_output
    qstat_output=$(qstat -u s2240011 2>/dev/null | tail -n +6 || true)

    for q in "${CPU_QUEUES[@]}"; do
        QUEUE_CURRENT[$q]=$(echo "$qstat_output" | awk -v q="$q" '$3==q' | wc -l || echo 0)
    done
    for q in "${GPU_QUEUES[@]}"; do
        GPU_QUEUE_CURRENT[$q]=$(echo "$qstat_output" | awk -v q="$q" '$3==q' | wc -l || echo 0)
    done
}

find_available_queue() {
    for q in "${CPU_QUEUES[@]}"; do
        local current="${QUEUE_CURRENT[$q]:-0}"
        local max="${QUEUE_MAX[$q]:-0}"
        if (( current < max )); then
            echo "$q"
            return 0
        fi
    done
    return 1
}

find_available_gpu_queue() {
    for q in "${GPU_QUEUES[@]}"; do
        local current="${GPU_QUEUE_CURRENT[$q]:-0}"
        local max="${GPU_QUEUE_MAX[$q]:-0}"
        if (( current < max )); then
            echo "$q"
            return 0
        fi
    done
    return 1
}

# ---- 全実験条件を列挙 ----
ALL_JOBS=()
MODELS=("SvmW" "SvmA" "Lstm")
CONDITIONS=("baseline" "smote_plain" "smote" "undersample")
SEEDS=(42 123)
RATIO="0.5"

for MODEL in "${MODELS[@]}"; do
    for COND in "${CONDITIONS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            ALL_JOBS+=("${MODEL}|${COND}|${SEED}")
        done
    done
done

echo "[$(date +%H:%M)] Pooled-conditions daemon started. Total configs: ${#ALL_JOBS[@]}" >> "$LOG"
echo "[$(date +%H:%M)] Models: ${MODELS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Conditions: ${CONDITIONS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Seeds: ${SEEDS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Polling every ${POLL_INTERVAL}s" >> "$LOG"

# ---- メインループ ----
while true; do
    get_queue_counts || true

    SUBMITTED_THIS_ROUND=0
    REMAINING=0

    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r MODEL COND SEED <<< "$job_spec"

        KEY="${MODEL}:${COND}:s${SEED}"

        # Skip if eval result already exists
        if has_eval_result "$MODEL" "$COND" "$SEED" "$RATIO"; then
            continue
        fi

        # Skip if already submitted
        if grep -qF "$KEY" "$SUBMITTED_KEYS" 2>/dev/null; then
            ((REMAINING++)) || true
            continue
        fi

        # Find available queue (GPU for Lstm, CPU for others)
        QUEUE=""
        if [[ "$MODEL" == "Lstm" ]]; then
            QUEUE=$(find_available_gpu_queue) || true
        else
            QUEUE=$(find_available_queue) || true
        fi
        if [[ -z "$QUEUE" ]]; then
            ((REMAINING++)) || true
            continue
        fi

        # Resources: Lstm needs GPU, others CPU-only
        if [[ "$MODEL" == "Lstm" ]]; then
            NCPUS_MEM="ncpus=4:ngpus=1:mem=8gb"
        else
            NCPUS_MEM="ncpus=4:mem=16gb"
        fi
        WALLTIME="24:00:00"

        # Generate job name (compact)
        COND_SHORT="${COND:0:3}"
        JOB_NAME="P_${MODEL:0:2}_${COND_SHORT}_s${SEED}"

        # Submit
        JID=$(qsub -N "$JOB_NAME" \
            -q "$QUEUE" \
            -l "select=1:${NCPUS_MEM}" \
            -l "walltime=${WALLTIME}" \
            -v "MODEL=${MODEL},CONDITION=${COND},SEED=${SEED},RATIO=${RATIO}" \
            "$JOB_SCRIPT" 2>&1) || {
            echo "[$(date +%H:%M)] SUBMIT FAILED: $KEY → $JID" >> "$LOG"
            continue
        }

        JID_NUM="${JID%%.*}"
        echo "$KEY" >> "$SUBMITTED_KEYS"
        if [[ "$MODEL" == "Lstm" ]]; then
            GPU_QUEUE_CURRENT[$QUEUE]=$(( ${GPU_QUEUE_CURRENT[$QUEUE]:-0} + 1 ))
        else
            QUEUE_CURRENT[$QUEUE]=$(( ${QUEUE_CURRENT[$QUEUE]:-0} + 1 ))
        fi
        ((SUBMITTED_THIS_ROUND++)) || true

        echo "[$(date +%H:%M)] SUBMITTED: $KEY → $JID_NUM ($QUEUE)" >> "$LOG"
    done

    # Count truly remaining
    TOTAL_REMAINING=0
    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r MODEL COND SEED <<< "$job_spec"
        if ! has_eval_result "$MODEL" "$COND" "$SEED" "$RATIO"; then
            ((TOTAL_REMAINING++)) || true
        fi
    done

    if (( TOTAL_REMAINING == 0 )); then
        echo "[$(date +%H:%M)] ALL DONE – all ${#ALL_JOBS[@]} pooled configs have eval results." >> "$LOG"
        break
    fi

    # Status
    TOTAL_Q=0
    for q in "${CPU_QUEUES[@]}"; do TOTAL_Q=$(( TOTAL_Q + ${QUEUE_CURRENT[$q]:-0} )); done
    echo "[$(date +%H:%M)] Poll: submitted=$SUBMITTED_THIS_ROUND remaining=$TOTAL_REMAINING queue=$TOTAL_Q" >> "$LOG"

    sleep "$POLL_INTERVAL"
done

echo "[$(date +%H:%M)] Daemon finished." >> "$LOG"

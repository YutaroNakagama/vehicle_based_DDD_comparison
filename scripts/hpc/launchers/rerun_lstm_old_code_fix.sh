#!/bin/bash
# ============================================================
# 再実行デーモン — Lstm 旧コード結果の再投入 (baseline + smote)
# ============================================================
# 問題: jid < 14800000 の Lstm 結果が無効:
#   - baseline: recall=0 (全件) — コード修正前の結果
#   - imbalv3/smote: oversampling 未適用バグ
#
# 重複のある 63 ディレクトリは _invalidated_old_duplicates へ移動済み。
# 残り 72 ディレクトリは _invalidated_old_code へ移動済み。
#
# 対象:
#   - Lstm baseline:  3モード × 3距離 × 2ドメイン × 2シード = 36 configs
#   - Lstm smote:     3モード × 3距離 × 2ドメイン × 2シード × 2比率 = 72 configs
#   合計: 108 configs (valid eval 既存分を除く)
#
# Usage:
#   nohup bash scripts/hpc/launchers/rerun_lstm_old_code_fix.sh &
#   # ログ: /tmp/rerun_lstm_old_code_fix.log
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh"
LOG="/tmp/rerun_lstm_old_code_fix.log"
SUBMITTED_KEYS="/tmp/rerun_lstm_old_code_fix_keys.txt"
POLL_INTERVAL=300  # 5 minutes
N_TRIALS=100
RANKING="knn"

# ---- エラートラップ ----
trap 'echo "[$(date +%H:%M)] TRAP: daemon exiting (line $LINENO, exit=$?)" >> "$LOG"' EXIT
trap 'echo "[$(date +%H:%M)] TRAP: received signal, exiting" >> "$LOG"; exit 1' INT TERM HUP

# ---- GPU キュー制限 ----
declare -A GPU_QUEUE_MAX=( [GPU-1]=15 [GPU-1A]=10 [GPU-S]=10 [GPU-L]=2 [GPU-LA]=2 )
GPU_QUEUES=("GPU-L" "GPU-LA" "GPU-1" "GPU-1A" "GPU-S")
declare -A QUEUE_CURRENT=()

touch "$SUBMITTED_KEYS"

# ---- リソース定義 ----
get_resources() {
    local mode="$1"
    # source_only/target_only needs ~4h train + eval → 8h
    # mixed needs ~8h train + eval → 12h
    # actual RSS max ~2GB; reduced from 32gb
    if [[ "$mode" == "mixed" ]]; then
        echo "ncpus=4:ngpus=1:mem=8gb 12:00:00"
    else
        echo "ncpus=4:ngpus=1:mem=8gb 08:00:00"
    fi
}

# ---- 新しい評価結果が存在するか確認 ----
has_eval_result() {
    local cond="$1" dist="$2" dom="$3" mode="$4" seed="$5" ratio="$6"
    local eval_dir="results/outputs/evaluation/Lstm"

    local pattern
    if [[ "$cond" == "baseline" ]]; then
        pattern="eval_results_Lstm_${mode}_prior_Lstm_baseline_knn_${dist}_${dom}_${mode}_split2_s${seed}.json"
    else
        # smote → imbalv3 in filename, subjectwise
        pattern="eval_results_Lstm_${mode}_prior_Lstm_imbalv3_knn_${dist}_${dom}_${mode}_split2_subjectwise_ratio${ratio}_s${seed}.json"
    fi

    # Check if matching file exists (excluding invalidated)
    find "$eval_dir" -name "$pattern" 2>/dev/null | grep -v _invalidated | grep -q .
}

# ---- キュー状態確認 ----
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
            echo "$q"
            return 0
        fi
    done
    return 1
}

# ---- 全実験条件を列挙 ----
ALL_JOBS=()
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("in_domain" "out_domain")
SEEDS=(42 123)
RATIOS=(0.1 0.5)
MODES=("source_only" "target_only" "mixed")
CONDITIONS=("baseline" "smote")

for MODE in "${MODES[@]}"; do
    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # baseline (no ratio)
                ALL_JOBS+=("baseline|$DIST|$DOM|$MODE|$SEED|none")
                # smote (with ratios)
                for RATIO in "${RATIOS[@]}"; do
                    ALL_JOBS+=("smote|$DIST|$DOM|$MODE|$SEED|$RATIO")
                done
            done
        done
    done
done

echo "[$(date +%H:%M)] Lstm old-code-fix daemon started. Total configs: ${#ALL_JOBS[@]}" >> "$LOG"
echo "[$(date +%H:%M)] Conditions: ${CONDITIONS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Modes: ${MODES[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Polling every ${POLL_INTERVAL}s" >> "$LOG"

# ---- メインループ ----
while true; do
    get_queue_counts || true

    SUBMITTED_THIS_ROUND=0
    REMAINING=0

    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r COND DIST DOM MODE SEED RATIO <<< "$job_spec"

        KEY="Lstm:${COND}:${DIST}:${DOM}:${MODE}:s${SEED}:r${RATIO}"

        # Skip if eval result already exists
        if has_eval_result "$COND" "$DIST" "$DOM" "$MODE" "$SEED" "$RATIO"; then
            continue
        fi

        # Skip if already submitted
        if grep -qF "$KEY" "$SUBMITTED_KEYS" 2>/dev/null; then
            ((REMAINING++)) || true
            continue
        fi

        # Find GPU queue
        QUEUE=""
        QUEUE=$(find_available_gpu_queue) || true
        if [[ -z "$QUEUE" ]]; then
            ((REMAINING++)) || true
            continue
        fi

        # Get resources
        RES=$(get_resources "$MODE")
        NCPUS_MEM=$(echo "$RES" | cut -d' ' -f1)
        WALLTIME=$(echo "$RES" | cut -d' ' -f2)

        # Generate job name
        COND_SHORT="${COND:0:2}"
        MODE_SHORT="${MODE:0:2}"
        DIST_SHORT="${DIST:0:1}"
        DOM_SHORT="${DOM:0:1}"
        JOB_NAME="Ls_${COND_SHORT}_${DIST_SHORT}${DOM_SHORT}_${MODE_SHORT}_s${SEED}"
        if [[ "$RATIO" != "none" ]]; then
            JOB_NAME="${JOB_NAME}_r${RATIO}"
        fi

        # Build qsub variables
        VARS="MODEL=Lstm,CONDITION=$COND,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        if [[ "$RATIO" != "none" ]]; then
            VARS="${VARS},RATIO=$RATIO"
        fi

        JOB_ID=$(qsub -N "$JOB_NAME" \
            -l "select=1:${NCPUS_MEM}" \
            -l "walltime=${WALLTIME}" \
            -q "$QUEUE" \
            -v "$VARS" \
            "$JOB_SCRIPT" 2>&1) || {
            echo "  [ERR] Failed: $KEY → $JOB_ID" >> "$LOG"
            ((REMAINING++)) || true
            continue
        }

        JOB_ID_NUM="${JOB_ID%%.*}"
        echo "$KEY" >> "$SUBMITTED_KEYS"
        QUEUE_CURRENT[$QUEUE]=$(( ${QUEUE_CURRENT[$QUEUE]:-0} + 1 ))
        ((SUBMITTED_THIS_ROUND++)) || true

        echo "  [SUB] Lstm | $COND | $MODE | $DIST | $DOM | r=${RATIO} | s$SEED | $QUEUE → $JOB_ID_NUM" >> "$LOG"

        sleep 0.3
    done

    # Count truly remaining (not yet having eval results)
    TOTAL_REMAINING=0
    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r COND DIST DOM MODE SEED RATIO <<< "$job_spec"
        if ! has_eval_result "$COND" "$DIST" "$DOM" "$MODE" "$SEED" "$RATIO"; then
            ((TOTAL_REMAINING++)) || true
        fi
    done

    if (( TOTAL_REMAINING == 0 )); then
        echo "[$(date +%H:%M)] ALL DONE – all ${#ALL_JOBS[@]} configs have eval results." >> "$LOG"
        break
    fi

    TOTAL_GPU=0
    for q in "${GPU_QUEUES[@]}"; do TOTAL_GPU=$(( TOTAL_GPU + ${QUEUE_CURRENT[$q]:-0} )); done
    echo "[POLL] $(date +%H:%M) | gpu_q=$TOTAL_GPU | submitted=$SUBMITTED_THIS_ROUND | remaining=$TOTAL_REMAINING" >> "$LOG"

    if [[ "$SUBMITTED_THIS_ROUND" -eq 0 ]]; then
        echo "  (waiting for GPU slots or results, sleeping...)" >> "$LOG"
    fi

    sleep "$POLL_INTERVAL"
done

echo "[$(date +%H:%M)] Daemon finished." >> "$LOG"

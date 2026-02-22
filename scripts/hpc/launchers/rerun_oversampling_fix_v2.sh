#!/bin/bash
# ============================================================
# 再実行デーモン v2 — oversampling バグ修正後のジョブ再投入
# ============================================================
# v1 からの変更点:
#   - Lstm を GPU キューに投入 (GPU-1, GPU-1A, GPU-S, GPU-L, GPU-LA)
#   - GPU 版 PBS スクリプト使用 (pbs_prior_research_unified_gpu.sh)
#   - SvmA は引き続き CPU キューに投入
#
# 対象:
#   - Lstm: smote_plain, undersample (全モード × 全configs)
#   - SvmA: smote_plain, undersample (全モード × 全configs)
#   各 144 configs = 合計最大 288 jobs
#
# Usage:
#   nohup bash scripts/hpc/launchers/rerun_oversampling_fix_v2.sh &
#   # ログ: /tmp/rerun_oversampling_fix_v2.log
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

JOB_SCRIPT_CPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
JOB_SCRIPT_GPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh"
LOG="/tmp/rerun_oversampling_fix_v2.log"
SUBMITTED_KEYS="/tmp/rerun_oversampling_fix_v2_keys.txt"
POLL_INTERVAL=300  # 5 minutes
N_TRIALS=100
RANKING="knn"

# ---- エラートラップ ----
trap 'echo "[$(date +%H:%M)] TRAP: daemon exiting (line $LINENO, exit=$?)" >> "$LOG"' EXIT
trap 'echo "[$(date +%H:%M)] TRAP: received signal, exiting" >> "$LOG"; exit 1' INT TERM HUP

# ---- キュー制限 ----
# CPU queues
declare -A CPU_QUEUE_MAX=( [SINGLE]=40 [DEFAULT]=40 [SMALL]=30 [LONG]=15 )
CPU_QUEUES=("SINGLE" "DEFAULT" "SMALL" "LONG")

# GPU queues (per-queue limits based on cluster config)
#   GPU-1: 4 GPUs, GPU-1A: 2, GPU-S: 2, GPU-L: 1, GPU-LA: 1 → total ~10
declare -A GPU_QUEUE_MAX=( [GPU-1]=15 [GPU-1A]=10 [GPU-S]=10 [GPU-L]=2 [GPU-LA]=2 )
GPU_QUEUES=("GPU-L" "GPU-LA" "GPU-1" "GPU-1A" "GPU-S")

declare -A QUEUE_CURRENT=()

touch "$SUBMITTED_KEYS"

# ---- リソース定義 ----
get_resources() {
    local model="$1"
    local mode="$2"

    case "$model" in
        Lstm)
            # GPU: 1 GPU + 4 CPUs (actual RSS max ~2GB)
            # source_only/target_only needs ~4h train + eval → 8h
            # mixed needs ~8h train + eval → 12h
            if [[ "$mode" == "mixed" ]]; then
                echo "ncpus=4:ngpus=1:mem=8gb 12:00:00"
            else
                echo "ncpus=4:ngpus=1:mem=8gb 08:00:00"
            fi
            ;;
        SvmA)
            # CPU: PSO optimization takes significant time (actual RSS max ~1.3GB)
            if [[ "$mode" == "mixed" ]]; then
                echo "ncpus=4:mem=8gb 48:00:00"
            else
                echo "ncpus=4:mem=8gb 30:00:00"
            fi
            ;;
    esac
}

# ---- 新しい評価結果が存在するか確認 ----
has_eval_result() {
    local model="$1" cond="$2" dist="$3" dom="$4" mode="$5" seed="$6" ratio="$7"
    local eval_dir="results/outputs/evaluation/$model"

    local tag
    case "$cond" in
        smote_plain) tag="smote_plain" ;;
        undersample) tag="undersample_rus" ;;
    esac

    # domain_train pattern (new format with _within/_cross)
    local dt_pattern="eval_results_${model}_domain_train_prior_${model}_${tag}_knn_${dist}_${dom}_*ratio${ratio}_s${seed}_within.json"

    # Legacy pattern (source_only/target_only/mixed)
    local legacy_pattern="eval_results_${model}_${mode}_prior_${model}_${tag}_knn_${dist}_${dom}_*ratio${ratio}_s${seed}.json"

    # Check if matching file exists (excluding invalidated)
    if find "$eval_dir" -name "$dt_pattern" 2>/dev/null | grep -v _invalidated | grep -q .; then
        return 0
    fi
    if find "$eval_dir" -name "$legacy_pattern" 2>/dev/null | grep -v _invalidated | grep -q .; then
        return 0
    fi
    return 1
}

# ---- キュー状態確認 ----
get_queue_counts() {
    local qstat_output
    qstat_output=$(qstat -u s2240011 2>/dev/null | tail -n +6 || true)

    for q in "${CPU_QUEUES[@]}" "${GPU_QUEUES[@]}"; do
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

find_available_cpu_queue() {
    # Also check total CPU count doesn't exceed 125
    local total_cpu=0
    for q in "${CPU_QUEUES[@]}"; do
        total_cpu=$(( total_cpu + ${QUEUE_CURRENT[$q]:-0} ))
    done
    if (( total_cpu >= 125 )); then
        return 1
    fi

    for q in "${CPU_QUEUES[@]}"; do
        local current="${QUEUE_CURRENT[$q]:-0}"
        local max="${CPU_QUEUE_MAX[$q]:-0}"
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
CONDITIONS=("smote_plain" "undersample")
# Lstm first (GPU, faster), then SvmA (CPU)
MODELS=("Lstm" "SvmA")

for MODEL in "${MODELS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for DIST in "${DISTANCES[@]}"; do
            for DOM in "${DOMAINS[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    for COND in "${CONDITIONS[@]}"; do
                        for RATIO in "${RATIOS[@]}"; do
                            ALL_JOBS+=("$MODEL|$COND|$DIST|$DOM|$MODE|$SEED|$RATIO")
                        done
                    done
                done
            done
        done
    done
done

echo "[$(date +%H:%M)] Daemon v2 started (GPU for Lstm). Total configs: ${#ALL_JOBS[@]}" >> "$LOG"
echo "[$(date +%H:%M)] Models: ${MODELS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Conditions: ${CONDITIONS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Lstm → GPU queues: ${GPU_QUEUES[*]}" >> "$LOG"
echo "[$(date +%H:%M)] SvmA → CPU queues: ${CPU_QUEUES[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Polling every ${POLL_INTERVAL}s" >> "$LOG"

# ---- メインループ ----
while true; do
    get_queue_counts || true

    SUBMITTED_THIS_ROUND=0
    REMAINING=0

    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r MODEL COND DIST DOM MODE SEED RATIO <<< "$job_spec"

        KEY="${MODEL}:${COND}:${DIST}:${DOM}:${MODE}:r${RATIO}:s${SEED}"

        # Skip if eval result already exists
        if has_eval_result "$MODEL" "$COND" "$DIST" "$DOM" "$MODE" "$SEED" "$RATIO"; then
            continue
        fi

        # Skip if already submitted
        if grep -qF "$KEY" "$SUBMITTED_KEYS" 2>/dev/null; then
            ((REMAINING++)) || true
            continue
        fi

        # Route by model
        QUEUE=""
        JOB_SCRIPT=""
        if [[ "$MODEL" == "Lstm" ]]; then
            QUEUE=$(find_available_gpu_queue) || true
            JOB_SCRIPT="$JOB_SCRIPT_GPU"
        else
            QUEUE=$(find_available_cpu_queue) || true
            JOB_SCRIPT="$JOB_SCRIPT_CPU"
        fi

        if [[ -z "$QUEUE" ]]; then
            ((REMAINING++)) || true
            continue
        fi

        # Get resources
        RES=$(get_resources "$MODEL" "$MODE")
        NCPUS_MEM=$(echo "$RES" | cut -d' ' -f1)
        WALLTIME=$(echo "$RES" | cut -d' ' -f2)

        # Generate job name (compact)
        MODEL_SHORT="${MODEL:0:2}"
        COND_SHORT="${COND:0:2}"
        MODE_SHORT="${MODE:0:2}"
        DIST_SHORT="${DIST:0:1}"
        DOM_SHORT="${DOM:0:1}"
        JOB_NAME="${MODEL_SHORT}_${COND_SHORT}_${DIST_SHORT}${DOM_SHORT}_${MODE_SHORT}_r${RATIO}_s${SEED}"

        # Build qsub command
        VARS="MODEL=$MODEL,CONDITION=$COND,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true,RATIO=$RATIO"

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

        echo "  [SUB] $MODEL | $COND | $MODE | $DIST | $DOM | r=${RATIO} | s$SEED | $QUEUE → $JOB_ID_NUM" >> "$LOG"

        sleep 0.3
    done

    # Count truly remaining (not yet having eval results)
    TOTAL_REMAINING=0
    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r MODEL COND DIST DOM MODE SEED RATIO <<< "$job_spec"
        if ! has_eval_result "$MODEL" "$COND" "$DIST" "$DOM" "$MODE" "$SEED" "$RATIO"; then
            ((TOTAL_REMAINING++)) || true
        fi
    done

    if (( TOTAL_REMAINING == 0 )); then
        echo "[$(date +%H:%M)] ALL DONE – all ${#ALL_JOBS[@]} configs have eval results." >> "$LOG"
        break
    fi

    # Status
    TOTAL_CPU=0; TOTAL_GPU=0
    for q in "${CPU_QUEUES[@]}"; do TOTAL_CPU=$(( TOTAL_CPU + ${QUEUE_CURRENT[$q]:-0} )); done
    for q in "${GPU_QUEUES[@]}"; do TOTAL_GPU=$(( TOTAL_GPU + ${QUEUE_CURRENT[$q]:-0} )); done
    echo "[POLL] $(date +%H:%M) | cpu_q=$TOTAL_CPU | gpu_q=$TOTAL_GPU | submitted=$SUBMITTED_THIS_ROUND | remaining=$TOTAL_REMAINING" >> "$LOG"

    if [[ "$SUBMITTED_THIS_ROUND" -eq 0 ]]; then
        echo "  (waiting for slots or results, sleeping...)" >> "$LOG"
    fi

    sleep "$POLL_INTERVAL"
done

echo "[$(date +%H:%M)] Daemon v2 finished." >> "$LOG"

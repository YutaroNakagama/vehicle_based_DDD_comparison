#!/bin/bash
# ============================================================
# Re-execution daemon — submit pooled jobs for 15 seeds × all conditions × 2 ratios
# ============================================================
# Submits pooled training+eval for all models × conditions × seeds × ratios.
#
# Target:
#   - 3 models (SvmW, SvmA, Lstm)
#   - 4 conditions (baseline, smote_plain, smote/sw_smote, undersample)
#   - 15 seeds
#   - 2 ratios for non-baseline (0.1, 0.5)
#   - Total: 3 × (15 + 15×2 ×3) = 3 × 105 = 315 configs
#
# Usage:
#   nohup bash scripts/hpc/launchers/rerun_pooled_15seeds.sh > /dev/null 2>&1 &
#   tail -f /tmp/rerun_pooled_15seeds.log
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research.sh"
LOG="/tmp/rerun_pooled_15seeds.log"
SUBMITTED_KEYS="/tmp/rerun_pooled_15seeds_keys.txt"
POLL_INTERVAL=300  # 5 minutes

# ---- Error trap ----
trap 'echo "[$(date +%H:%M)] TRAP: daemon exiting (line $LINENO, exit=$?)" >> "$LOG"' EXIT
trap 'echo "[$(date +%H:%M)] TRAP: received signal, exiting" >> "$LOG"; exit 1' INT TERM HUP

# ---- Queue limits ----
declare -A QUEUE_MAX=( [SINGLE]=40 [DEFAULT]=40 [SMALL]=30 [LONG]=15 )
declare -A QUEUE_CURRENT=()
CPU_QUEUES=("SINGLE" "DEFAULT" "SMALL" "LONG")

# GPU queues for Lstm
declare -A GPU_QUEUE_MAX=( [GPU-1]=4 [GPU-1A]=4 [GPU-S]=4 [GPU-L]=4 [GPU-LA]=4 )
declare -A GPU_QUEUE_CURRENT=()
GPU_QUEUES=("GPU-1" "GPU-1A" "GPU-S" "GPU-L" "GPU-LA")

# ---- Max total jobs in queue ----
MAX_TOTAL_JOBS=165

touch "$SUBMITTED_KEYS"

# ---- Check if evaluation result exists ----
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

# ---- Queue helpers ----
get_queue_counts() {
    local qstat_output
    qstat_output=$(qstat -u s2240011 2>/dev/null | tail -n +6 || true)

    TOTAL_IN_QUEUE=$(echo "$qstat_output" | grep -c "s2240011" || echo 0)

    for q in "${CPU_QUEUES[@]}"; do
        QUEUE_CURRENT[$q]=$(echo "$qstat_output" | awk -v q="$q" '$3==q' | wc -l || echo 0)
    done
    for q in "${GPU_QUEUES[@]}"; do
        GPU_QUEUE_CURRENT[$q]=$(echo "$qstat_output" | awk -v q="$q" '$3==q' | wc -l || echo 0)
    done
}

find_available_queue() {
    # Check total limit first
    if (( TOTAL_IN_QUEUE >= MAX_TOTAL_JOBS )); then
        return 1
    fi
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
    # Check total limit first
    if (( TOTAL_IN_QUEUE >= MAX_TOTAL_JOBS )); then
        return 1
    fi
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

# ---- Enumerate all experiment conditions ----
ALL_JOBS=()
MODELS=("SvmW" "SvmA" "Lstm")
CONDITIONS=("baseline" "smote_plain" "smote" "undersample")
SEEDS=(42 123 0 1 3 7 13 99 256 512 777 999 1337 2024 1234)

for MODEL in "${MODELS[@]}"; do
    for COND in "${CONDITIONS[@]}"; do
        if [[ "$COND" == "baseline" ]]; then
            for SEED in "${SEEDS[@]}"; do
                ALL_JOBS+=("${MODEL}|${COND}|${SEED}|")
            done
        else
            for RATIO in 0.1 0.5; do
                for SEED in "${SEEDS[@]}"; do
                    ALL_JOBS+=("${MODEL}|${COND}|${SEED}|${RATIO}")
                done
            done
        fi
    done
done

echo "[$(date +%H:%M)] Pooled-15seeds daemon started. Total configs: ${#ALL_JOBS[@]}" >> "$LOG"
echo "[$(date +%H:%M)] Models: ${MODELS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Conditions: ${CONDITIONS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Seeds: ${SEEDS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Ratios: 0.1, 0.5 (for non-baseline)" >> "$LOG"
echo "[$(date +%H:%M)] Polling every ${POLL_INTERVAL}s" >> "$LOG"

# ---- Main loop ----
while true; do
    get_queue_counts || true

    SUBMITTED_THIS_ROUND=0
    REMAINING=0

    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r MODEL COND SEED RATIO <<< "$job_spec"

        # Default ratio for baseline
        RATIO="${RATIO:-0.5}"

        KEY="${MODEL}:${COND}:r${RATIO}:s${SEED}"
        [[ "$COND" == "baseline" ]] && KEY="${MODEL}:${COND}:s${SEED}"

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

        # Resources
        if [[ "$MODEL" == "Lstm" ]]; then
            NCPUS_MEM="ncpus=4:ngpus=1:mem=8gb"
        else
            NCPUS_MEM="ncpus=4:mem=16gb"
        fi
        WALLTIME="24:00:00"

        # Compact job name
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
        TOTAL_IN_QUEUE=$(( TOTAL_IN_QUEUE + 1 ))
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
        IFS='|' read -r MODEL COND SEED RATIO <<< "$job_spec"
        RATIO="${RATIO:-0.5}"
        if ! has_eval_result "$MODEL" "$COND" "$SEED" "$RATIO"; then
            ((TOTAL_REMAINING++)) || true
        fi
    done

    if (( TOTAL_REMAINING == 0 )); then
        echo "[$(date +%H:%M)] ALL DONE – all ${#ALL_JOBS[@]} pooled configs have eval results." >> "$LOG"
        break
    fi

    echo "[$(date +%H:%M)] Poll: submitted=$SUBMITTED_THIS_ROUND remaining=$TOTAL_REMAINING queue=$TOTAL_IN_QUEUE" >> "$LOG"

    sleep "$POLL_INTERVAL"
done

echo "[$(date +%H:%M)] Daemon finished." >> "$LOG"

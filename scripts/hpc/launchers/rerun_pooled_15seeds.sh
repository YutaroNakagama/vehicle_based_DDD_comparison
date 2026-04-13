#!/bin/bash
# ============================================================
# Re-execution daemon — batch-submit pooled jobs (15 seeds per batch)
# ============================================================
# Groups configs by (model, condition, ratio) and submits ONE batch
# PBS job per group.  Each batch runs up to 15 tasks in parallel.
#
# Target: 3 models × 7 condition/ratio combos × 15 seeds = 315 configs
#         → 21 batch jobs (vs 315 individual jobs)
#
# Usage:
#   nohup bash scripts/hpc/launchers/rerun_pooled_15seeds.sh > /dev/null 2>&1 &
#   tail -f /tmp/rerun_pooled_15seeds.log
# ============================================================

set -u

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

BATCH_JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_batch.sh"
TASK_DIR="$PROJECT_ROOT/scripts/hpc/logs/train/task_files/pooled_batch"
LOG="/tmp/rerun_pooled_15seeds.log"
SUBMITTED_KEYS="/tmp/rerun_pooled_15seeds_keys.txt"
POLL_INTERVAL=300   # 5 min

mkdir -p "$TASK_DIR"
touch "$SUBMITTED_KEYS"

# ---- Error trap ----
trap 'echo "[$(date +%H:%M)] TRAP: daemon exiting (line $LINENO, exit=$?)" >> "$LOG"' EXIT
trap 'echo "[$(date +%H:%M)] TRAP: received signal, exiting" >> "$LOG"; exit 1' INT TERM HUP

# ---- Queue limits (CPU only) ----
declare -A QUEUE_MAX=( [SINGLE]=40 [DEFAULT]=40 [SMALL]=30 [LONG]=15 )
declare -A QUEUE_CURRENT=()
CPU_QUEUES=("SINGLE" "DEFAULT" "SMALL" "LONG")
MAX_TOTAL_JOBS=165
TOTAL_IN_QUEUE=0

# ---- Check if evaluation result exists ----
has_eval_result() {
    local model="$1" cond="$2" seed="$3" ratio="$4"
    local eval_dir="results/outputs/evaluation/${model}"
    local pattern
    case "$cond" in
        baseline)    pattern="eval_results_${model}_pooled_prior_${model}_baseline_s${seed}" ;;
        smote_plain) pattern="eval_results_${model}_pooled_prior_${model}_smote_plain_ratio${ratio}_s${seed}" ;;
        smote)       pattern="eval_results_${model}_pooled_prior_${model}_imbalv3_subjectwise_ratio${ratio}_s${seed}" ;;
        undersample) pattern="eval_results_${model}_pooled_prior_${model}_undersample_rus_ratio${ratio}_s${seed}" ;;
    esac
    find "$eval_dir" -name "${pattern}*.json" 2>/dev/null | grep -v _invalidated | grep -q .
}

# ---- Queue helpers ----
get_queue_counts() {
    local qstat_output
    qstat_output=$(qstat -u s2240011 2>/dev/null | tail -n +6 || true)
    TOTAL_IN_QUEUE=$(echo "$qstat_output" | grep -c "s2240011" 2>/dev/null || echo 0)
    for q in "${CPU_QUEUES[@]}"; do
        QUEUE_CURRENT[$q]=$(echo "$qstat_output" | awk -v q="$q" '$3==q' | wc -l 2>/dev/null || echo 0)
    done
}

find_available_queue() {
    if (( TOTAL_IN_QUEUE >= MAX_TOTAL_JOBS )); then return 1; fi
    for q in "${CPU_QUEUES[@]}"; do
        local current="${QUEUE_CURRENT[$q]:-0}"
        local max="${QUEUE_MAX[$q]:-0}"
        if (( current < max )); then echo "$q"; return 0; fi
    done
    return 1
}

# ---- All experiment configs ----
MODELS=("SvmW" "SvmA" "Lstm")
CONDITIONS=("baseline" "smote_plain" "smote" "undersample")
SEEDS=(42 123 0 1 3 7 13 99 256 512 777 999 1337 2024 1234)

echo "[$(date +%H:%M)] Pooled-15seeds BATCH daemon started." >> "$LOG"
echo "[$(date +%H:%M)] Models: ${MODELS[*]}  Conditions: ${CONDITIONS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Batch job script: $BATCH_JOB_SCRIPT" >> "$LOG"

# ---- Main loop ----
while true; do
    get_queue_counts || true

    # ---- Phase 1: collect pending configs, grouped by (model,cond,ratio) ----
    declare -A PENDING_SEEDS       # group_key → "seed1 seed2 ..."
    TOTAL_REMAINING=0

    for MODEL in "${MODELS[@]}"; do
        for COND in "${CONDITIONS[@]}"; do
            if [[ "$COND" == "baseline" ]]; then
                RATIOS=("")
            else
                RATIOS=("0.1" "0.5")
            fi
            for RATIO in "${RATIOS[@]}"; do
                RATIO="${RATIO:-0.5}"
                for SEED in "${SEEDS[@]}"; do
                    # Build tracking key (same format as before for compatibility)
                    KEY=""
                    if [[ "$COND" == "baseline" ]]; then
                        KEY="${MODEL}:${COND}:s${SEED}"
                    else
                        KEY="${MODEL}:${COND}:r${RATIO}:s${SEED}"
                    fi

                    # Skip if eval exists
                    if has_eval_result "$MODEL" "$COND" "$SEED" "$RATIO"; then
                        continue
                    fi

                    ((TOTAL_REMAINING++)) || true

                    # Skip if already submitted
                    if grep -qF "$KEY" "$SUBMITTED_KEYS" 2>/dev/null; then
                        continue
                    fi

                    # Add to pending group
                    GROUP_KEY="${MODEL}|${COND}|${RATIO}"
                    PENDING_SEEDS[$GROUP_KEY]+="${SEED}|${KEY} "
                done
            done
        done
    done

    # ---- Phase 2: submit batch jobs for pending groups ----
    SUBMITTED_THIS_ROUND=0

    for GROUP_KEY in "${!PENDING_SEEDS[@]}"; do
        IFS='|' read -r MODEL COND RATIO <<< "$GROUP_KEY"
        ENTRIES="${PENDING_SEEDS[$GROUP_KEY]}"

        # Check queue availability
        QUEUE
        QUEUE=$(find_available_queue) || {
            echo "[$(date +%H:%M)] Queue full — deferring group ${GROUP_KEY}" >> "$LOG"
            continue
        }

        # Write task file
        TIMESTAMP
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        COND_SHORT="${COND:0:5}"
        TASK_FILE="${TASK_DIR}/${MODEL}_${COND_SHORT}_r${RATIO}_${TIMESTAMP}.txt"
        BATCH_COUNT=0

        for entry in $ENTRIES; do
            SEED="${entry%%|*}"
            echo "${MODEL}|${COND}|${SEED}|${RATIO}" >> "$TASK_FILE"
            ((BATCH_COUNT++)) || true
        done

        # Compute resources: 1 core per task + 1 spare, 4 GB per task
        NCPUS=$((BATCH_COUNT + 1))
        MEM_GB=$((BATCH_COUNT * 4 + 4))

        # Job name
        JOB_NAME="PB_${MODEL:0:2}_${COND_SHORT}"

        # Submit batch job
        JID
        JID=$(qsub -N "$JOB_NAME" \
            -q "$QUEUE" \
            -l "select=1:ncpus=${NCPUS}:mem=${MEM_GB}gb" \
            -l "walltime=24:00:00" \
            -v "TASK_FILE=${TASK_FILE}" \
            "$BATCH_JOB_SCRIPT" 2>&1) || {
            echo "[$(date +%H:%M)] SUBMIT FAILED: ${GROUP_KEY} (${BATCH_COUNT} tasks) → $JID" >> "$LOG"
            rm -f "$TASK_FILE"
            continue
        }

        JID_NUM="${JID%%.*}"

        # Record all individual keys as submitted
        for entry in $ENTRIES; do
            TRACK_KEY="${entry#*|}"
            echo "$TRACK_KEY" >> "$SUBMITTED_KEYS"
        done

        TOTAL_IN_QUEUE=$(( TOTAL_IN_QUEUE + 1 ))
        QUEUE_CURRENT[$QUEUE]=$(( ${QUEUE_CURRENT[$QUEUE]:-0} + 1 ))
        ((SUBMITTED_THIS_ROUND++)) || true

        echo "[$(date +%H:%M)] BATCH SUBMITTED: ${GROUP_KEY} → $JID_NUM ($QUEUE) [${BATCH_COUNT} tasks]" >> "$LOG"
    done

    unset PENDING_SEEDS

    # ---- Check completion ----
    if (( TOTAL_REMAINING == 0 )); then
        echo "[$(date +%H:%M)] ALL DONE — all pooled configs have eval results." >> "$LOG"
        break
    fi

    echo "[$(date +%H:%M)] Poll: batch_submitted=$SUBMITTED_THIS_ROUND remaining=$TOTAL_REMAINING queue=$TOTAL_IN_QUEUE" >> "$LOG"

    sleep "$POLL_INTERVAL"
done

echo "[$(date +%H:%M)] Daemon finished." >> "$LOG"

#!/bin/bash
# ============================================================
# Batch Parallel Launcher: SvmW (unified) + SvmA (split2)
# ============================================================
# Groups multiple tasks into batch jobs (4 tasks per job),
# each running in parallel within a single PBS allocation.
#
# Usage:
#   bash scripts/hpc/launchers/launch_batch_parallel.sh --dry-run
#   nohup bash scripts/hpc/launchers/launch_batch_parallel.sh \
#     > scripts/hpc/logs/train/batch_parallel_output.log 2>&1 &
# ============================================================
set -o pipefail

trap 'echo "[KILLED] Received SIGHUP at $(date)" | tee -a "${LOG_FILE:-/dev/null}"; exit 130' HUP
trap 'echo "[KILLED] Received SIGTERM at $(date)" | tee -a "${LOG_FILE:-/dev/null}"; exit 143' TERM
trap 'echo "[KILLED] Received SIGINT at $(date)" | tee -a "${LOG_FILE:-/dev/null}"; exit 130' INT
trap 'echo "[EXIT] Script exiting with code $? at $(date)" | tee -a "${LOG_FILE:-/dev/null}"' EXIT

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
BATCH_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_batch_parallel.sh"
EVAL_DIR="$PROJECT_ROOT/results/outputs/evaluation"

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# ===== Configuration =====
TASKS_PER_JOB=4          # parallel tasks per PBS job
CPUS_PER_TASK=8
MEM_PER_TASK_GB=32        # GB per task

# Queue limits
declare -A QUEUE_LIMITS=( [SINGLE]=40 [LONG]=15 [DEFAULT]=40 [SMALL]=30 )
MAX_TOTAL_JOBS=160
POLL_INTERVAL=120

# SvmW config
SVMW_SEEDS=(42 123 0 1 3 7 13 99 256 512 777 999 1337 2024 1234)
SVMW_WALLTIME="14:00:00"    # longest single SvmW task ~12h, +2h margin

# SvmA config
SVMA_SEEDS=(42 123)
SVMA_WALLTIME="48:00:00"    # longest SvmA task ~43h, +5h margin

RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")

# ===== Parse arguments =====
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/batch_parallel_${TIMESTAMP}.log"
TASK_DIR="$PROJECT_ROOT/scripts/hpc/logs/train/task_files"
mkdir -p "$TASK_DIR"

echo "============================================================"
echo "[BATCH-LAUNCHER] Parallel batch submission"
echo "============================================================"
echo "  Tasks per job: $TASKS_PER_JOB"
echo "  CPUs per task: $CPUS_PER_TASK"
echo "  Dry run: $DRY_RUN"
echo "============================================================"

# ============================================================
# Completion cache
# ============================================================
echo "Building completion caches..."

SVMW_CACHE=$(mktemp)
find "$EVAL_DIR" -name "eval_results_SvmW*domain_train*_within.json" 2>/dev/null \
    | xargs -I{} basename {} > "$SVMW_CACHE"
echo "  SvmW completed: $(wc -l < "$SVMW_CACHE")"

SVMA_CACHE=$(mktemp)
find "$EVAL_DIR" -name "eval_results_SvmA*split2*.json" 2>/dev/null \
    | xargs -I{} basename {} > "$SVMA_CACHE"
echo "  SvmA completed: $(wc -l < "$SVMA_CACHE")"
trap 'rm -f "$SVMW_CACHE" "$SVMA_CACHE"' EXIT

is_svmw_completed() {
    local cond="$1" dist="$2" dom="$3" seed="$4" ratio="$5"
    if [[ "$cond" == "baseline" ]]; then
        grep -q "SvmW_domain_train.*_${dist}_${dom}_.*_s${seed}_within\.json" "$SVMW_CACHE"
    else
        grep -q "SvmW_domain_train.*_${cond}_.*_${dist}_${dom}_.*ratio${ratio}_s${seed}_within\.json" "$SVMW_CACHE"
    fi
}

is_svma_completed() {
    local cond="$1" dist="$2" dom="$3" mode="$4" seed="$5" ratio="$6"
    if [[ "$cond" == "baseline" ]]; then
        grep -q "SvmA.*_${dist}_${dom}_${mode}_split2_s${seed}" "$SVMA_CACHE"
    else
        grep -q "SvmA.*_${cond}.*_${dist}_${dom}_${mode}_split2.*ratio${ratio}_s${seed}" "$SVMA_CACHE"
    fi
}

# ============================================================
# Queue management
# ============================================================
get_total_queue_count() {
    qstat -u "$USER" 2>/dev/null | awk '$1 ~ /^[0-9]+\./ {c++} END {print c+0}'
}

get_per_queue_counts() {
    declare -gA QUEUE_COUNTS
    QUEUE_COUNTS=( [SINGLE]=0 [LONG]=0 [DEFAULT]=0 [SMALL]=0 )
    while IFS=' ' read -r count queue; do
        if [[ -n "$queue" ]]; then
            QUEUE_COUNTS["$queue"]=$count
        fi
    done < <(qstat -u "$USER" 2>/dev/null | awk '$1 ~ /^[0-9]+\./ {print $3}' | sort | uniq -c | awk '{print $1, $2}')
}

pick_queue() {
    get_per_queue_counts
    local best_queue="SINGLE"
    local best_avail=-1
    for q in SINGLE LONG DEFAULT SMALL; do
        local cnt=${QUEUE_COUNTS[$q]:-0}
        local lim=${QUEUE_LIMITS[$q]:-40}
        local avail=$((lim - cnt))
        if [[ $avail -gt $best_avail ]]; then
            best_avail=$avail
            best_queue="$q"
        fi
    done
    echo "$best_queue"
}

wait_for_slot() {
    while true; do
        local total
        total=$(get_total_queue_count)
        if [[ $total -lt $MAX_TOTAL_JOBS ]]; then
            get_per_queue_counts
            for q in SINGLE LONG DEFAULT SMALL; do
                local cnt=${QUEUE_COUNTS[$q]:-0}
                local lim=${QUEUE_LIMITS[$q]:-40}
                if [[ $cnt -lt $lim ]]; then
                    return 0
                fi
            done
        fi
        echo "[WAIT] $total/$MAX_TOTAL_JOBS — waiting ${POLL_INTERVAL}s ($(date +%H:%M:%S))" | tee -a "$LOG_FILE"
        sleep "$POLL_INTERVAL"
    done
}

# ============================================================
# Collect pending tasks
# ============================================================
echo ""
echo "Scanning for pending tasks..."

declare -a SVMW_TASKS=()
declare -a SVMA_TASKS=()

# --- SvmW (unified: domain_train) ---
for DIST in "${DISTANCES[@]}"; do
    for DOM in "${DOMAINS[@]}"; do
        for SEED in "${SVMW_SEEDS[@]}"; do
            if ! is_svmw_completed "baseline" "$DIST" "$DOM" "$SEED" ""; then
                # FORMAT: MODEL|CONDITION|MODE|DISTANCE|DOMAIN|RATIO|SEED|N_TRIALS|RANKING|RUN_EVAL|SCRIPT_TYPE
                SVMW_TASKS+=("SvmW|baseline|domain_train|$DIST|$DOM||$SEED|$N_TRIALS|$RANKING|true|unified")
            fi
            for RATIO in "${RATIOS[@]}"; do
                for COND in smote_plain smote undersample; do
                    if ! is_svmw_completed "$COND" "$DIST" "$DOM" "$SEED" "$RATIO"; then
                        SVMW_TASKS+=("SvmW|$COND|domain_train|$DIST|$DOM|$RATIO|$SEED|$N_TRIALS|$RANKING|true|unified")
                    fi
                done
            done
        done
    done
done

# --- SvmA (split2: source_only / target_only) ---
SVMA_MODES=("source_only" "target_only")
for DIST in "${DISTANCES[@]}"; do
    for DOM in "${DOMAINS[@]}"; do
        for MODE in "${SVMA_MODES[@]}"; do
            for SEED in "${SVMA_SEEDS[@]}"; do
                if ! is_svma_completed "baseline" "$DIST" "$DOM" "$MODE" "$SEED" ""; then
                    SVMA_TASKS+=("SvmA|baseline|$MODE|$DIST|$DOM||$SEED|$N_TRIALS|$RANKING|true|split2")
                fi
                for RATIO in "${RATIOS[@]}"; do
                    for COND in smote_plain smote undersample; do
                        if ! is_svma_completed "$COND" "$DIST" "$DOM" "$MODE" "$SEED" "$RATIO"; then
                            SVMA_TASKS+=("SvmA|$COND|$MODE|$DIST|$DOM|$RATIO|$SEED|$N_TRIALS|$RANKING|true|split2")
                        fi
                    done
                done
            done
        done
    done
done

echo "  SvmW pending: ${#SVMW_TASKS[@]}"
echo "  SvmA pending: ${#SVMA_TASKS[@]}"

# Merge all tasks
ALL_TASKS=("${SVMW_TASKS[@]}" "${SVMA_TASKS[@]}")
TOTAL=${#ALL_TASKS[@]}
echo "  Total pending: $TOTAL"

if [[ $TOTAL -eq 0 ]]; then
    echo "[DONE] No pending tasks."
    exit 0
fi

# ============================================================
# Group into batches and submit
# ============================================================
echo ""
echo "Submitting batches of $TASKS_PER_JOB tasks..."
echo ""

SUBMITTED=0
BATCH_IDX=0

# Process SvmW and SvmA separately (different walltimes)
submit_batches() {
    local -n TASK_ARRAY=$1
    local MODEL_NAME="$2"
    local WALLTIME="$3"
    local COUNT=${#TASK_ARRAY[@]}

    if [[ $COUNT -eq 0 ]]; then
        echo "[$MODEL_NAME] No pending tasks."
        return
    fi

    local TOTAL_CPUS=$((CPUS_PER_TASK * TASKS_PER_JOB))
    local TOTAL_MEM=$((MEM_PER_TASK_GB * TASKS_PER_JOB))

    echo "[$MODEL_NAME] $COUNT tasks → $((( COUNT + TASKS_PER_JOB - 1 ) / TASKS_PER_JOB)) batch jobs"
    echo "[$MODEL_NAME] Resources per job: ${TOTAL_CPUS} CPUs, ${TOTAL_MEM}GB, walltime=${WALLTIME}"

    local i=0
    while [[ $i -lt $COUNT ]]; do
        ((BATCH_IDX++))

        # Write task file for this batch
        local TASK_FILE="$TASK_DIR/batch_${TIMESTAMP}_${MODEL_NAME}_${BATCH_IDX}.txt"
        local BATCH_SIZE=0
        for (( j=i; j < i + TASKS_PER_JOB && j < COUNT; j++ )); do
            echo "${TASK_ARRAY[$j]}" >> "$TASK_FILE"
            ((BATCH_SIZE++))
        done

        # Actual resources for this batch (last batch may be smaller)
        local ACTUAL_CPUS=$((CPUS_PER_TASK * BATCH_SIZE))
        local ACTUAL_MEM=$((MEM_PER_TASK_GB * BATCH_SIZE))
        local JOB_NAME="${MODEL_NAME}_batch${BATCH_IDX}"

        if $DRY_RUN; then
            echo "[DRY-RUN] $JOB_NAME: $BATCH_SIZE tasks (${ACTUAL_CPUS}CPU/${ACTUAL_MEM}GB) — $TASK_FILE"
        else
            local submitted=false
            local attempts=0
            while [[ "$submitted" == "false" ]]; do
                wait_for_slot
                local QUEUE
                QUEUE=$(pick_queue)

                JOB_ID=$(qsub -N "$JOB_NAME" \
                    -l "select=1:ncpus=${ACTUAL_CPUS}:mem=${ACTUAL_MEM}gb" \
                    -l "walltime=${WALLTIME}" \
                    -q "$QUEUE" \
                    -v "TASK_FILE=${TASK_FILE},PARALLEL=${BATCH_SIZE}" \
                    "$BATCH_SCRIPT" 2>&1)
                if [[ $? -eq 0 ]]; then
                    echo "[SUBMIT] $JOB_NAME ($BATCH_SIZE tasks) → $QUEUE → $JOB_ID" | tee -a "$LOG_FILE"
                    ((SUBMITTED += BATCH_SIZE))
                    submitted=true
                    sleep 0.5
                else
                    ((attempts++))
                    echo "[RETRY] $JOB_NAME failed ($QUEUE): $JOB_ID — attempt $attempts, waiting..." | tee -a "$LOG_FILE"
                    sleep "$POLL_INTERVAL"
                fi
            done
        fi

        i=$((i + TASKS_PER_JOB))
    done
}

submit_batches SVMW_TASKS "SvmW" "$SVMW_WALLTIME"
submit_batches SVMA_TASKS "SvmA" "$SVMA_WALLTIME"

echo ""
echo "============================================================"
echo "Submission Summary"
echo "============================================================"
echo "Total pending: $TOTAL"
echo "Submitted: $SUBMITTED"
echo "Batch jobs: $BATCH_IDX"
echo "Tasks per job: $TASKS_PER_JOB"
echo "Log: $LOG_FILE"
echo "Task files: $TASK_DIR"
echo "============================================================"

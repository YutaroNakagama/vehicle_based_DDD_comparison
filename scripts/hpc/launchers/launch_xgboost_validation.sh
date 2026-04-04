#!/bin/bash
# ============================================================
# XGBoost Validation — Batched launcher (A1 resolution)
# ============================================================
# Replicates the RF 4-factor factorial experiment with XGBoost
# to validate that Sobol-Hoeffding results are classifier-independent.
#
# Groups multiple experiments into batch jobs via pbs_batch_parallel.sh.
# Each batch contains 28 tasks (7 conditions × 4 seeds), running
# 4 in parallel within a single PBS allocation.
#
# Factorial: 7R × 3M × 3D × 2G × 12 seeds = 1,512 experiments
# Batching:  ~54 batch jobs (28 tasks/job, 4 parallel)
#
# Usage:
#   bash scripts/hpc/launchers/launch_xgboost_validation.sh --dry-run
#   bash scripts/hpc/launchers/launch_xgboost_validation.sh
#   bash scripts/hpc/launchers/launch_xgboost_validation.sh --seeds "42 123 256"
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

BATCH_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_batch_parallel.sh"
EVAL_DIR="$PROJECT_ROOT/results/outputs/evaluation"

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# ===== Factorial parameters (same as RF experiment) =====
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("in_domain" "out_domain")
MODES=("source_only" "target_only" "mixed")
RATIOS=("0.1" "0.5")
RANKING="knn"
N_TRIALS=100

DEFAULT_SEEDS=(0 1 3 7 13 42 123 256 512 999 1337 2024)

# ===== Batch configuration =====
TASKS_PER_JOB=28          # tasks per PBS job (7 conds × 4 seeds)
PARALLEL=1                # sequential (reduces resource footprint for queuing)
CPUS_PER_TASK=4
MEM_PER_TASK_GB=8
WALLTIME="18:00:00"       # 28 tasks × ~30min = ~14h → +4h margin

# Queue management (per-user limits observed empirically)
# SINGLE: max 7 queued per user
# DEFAULT: max 5 queued per user
# SMALL/LONG excluded: hit generic per-user limit with 1-2 jobs
declare -A QUEUE_LIMITS=( [SINGLE]=7 [DEFAULT]=5 )
MAX_TOTAL_JOBS=12
POLL_INTERVAL=120

# ===== Parse arguments =====
DRY_RUN=false
SEEDS=("${DEFAULT_SEEDS[@]}")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --seeds) read -ra SEEDS <<< "$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

SEED_COUNT=${#SEEDS[@]}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
TASK_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain/task_files"
mkdir -p "$LOG_DIR" "$TASK_DIR"
LOG_FILE="$LOG_DIR/launch_xgboost_batch_${TIMESTAMP}.log"

# ============================================================
# Completion cache — skip already-evaluated experiments
# ============================================================
echo "Building completion cache..."
XGB_CACHE=$(mktemp)
find "$EVAL_DIR" -name "eval_results_XGBoost*xgb_*.json" 2>/dev/null \
    | xargs -I{} basename {} > "$XGB_CACHE"
XGB_DONE=$(wc -l < "$XGB_CACHE")
echo "  XGBoost already completed: $XGB_DONE"
trap 'rm -f "$XGB_CACHE"' EXIT

is_completed() {
    local cond="$1" dist="$2" dom="$3" mode="$4" seed="$5" ratio="$6"
    if [[ "$cond" == "baseline" ]]; then
        grep -q "XGBoost.*xgb_baseline.*_${dist}_${dom}_${mode}_split2_s${seed}" "$XGB_CACHE"
    else
        grep -q "XGBoost.*xgb_${cond}.*_${dist}_${dom}_${mode}_split2.*ratio${ratio}_s${seed}" "$XGB_CACHE"
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
    QUEUE_COUNTS=( [SINGLE]=0 [DEFAULT]=0 )
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
    for q in SINGLE DEFAULT; do
        local cnt=${QUEUE_COUNTS[$q]:-0}
        local lim=${QUEUE_LIMITS[$q]:-10}
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
            for q in SINGLE DEFAULT; do
                local cnt=${QUEUE_COUNTS[$q]:-0}
                local lim=${QUEUE_LIMITS[$q]:-10}
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
echo "Scanning for pending XGBoost experiments..."

declare -a ALL_TASKS=()

for MODE in "${MODES[@]}"; do
    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # FORMAT: MODEL|CONDITION|MODE|DISTANCE|DOMAIN|RATIO|SEED|N_TRIALS|RANKING|RUN_EVAL|SCRIPT_TYPE

                # 1. baseline
                if ! is_completed "baseline" "$DIST" "$DOM" "$MODE" "$SEED" ""; then
                    ALL_TASKS+=("XGBoost|baseline|${MODE}|${DIST}|${DOM}||${SEED}|${N_TRIALS}|${RANKING}|true|xgb")
                fi

                for RATIO in "${RATIOS[@]}"; do
                    # 2-3. smote_plain
                    if ! is_completed "smote_plain" "$DIST" "$DOM" "$MODE" "$SEED" "$RATIO"; then
                        ALL_TASKS+=("XGBoost|smote_plain|${MODE}|${DIST}|${DOM}|${RATIO}|${SEED}|${N_TRIALS}|${RANKING}|true|xgb")
                    fi
                    # 4-5. smote (subject-wise)
                    if ! is_completed "smote" "$DIST" "$DOM" "$MODE" "$SEED" "$RATIO"; then
                        ALL_TASKS+=("XGBoost|smote|${MODE}|${DIST}|${DOM}|${RATIO}|${SEED}|${N_TRIALS}|${RANKING}|true|xgb")
                    fi
                    # 6-7. undersample
                    if ! is_completed "undersample" "$DIST" "$DOM" "$MODE" "$SEED" "$RATIO"; then
                        ALL_TASKS+=("XGBoost|undersample|${MODE}|${DIST}|${DOM}|${RATIO}|${SEED}|${N_TRIALS}|${RANKING}|true|xgb")
                    fi
                done
            done
        done
    done
done

TOTAL=${#ALL_TASKS[@]}
TOTAL_JOBS=$(( (TOTAL + TASKS_PER_JOB - 1) / TASKS_PER_JOB ))

echo ""
echo "============================================================"
echo "  XGBoost Validation — Batched Launcher"
echo "  $(date)"
echo "============================================================"
echo "  MODEL: XGBoost"
echo "  FACTORIAL: 7R × 3M × 3D × 2G = 126 cells"
echo "  SEEDS: $SEED_COUNT seeds"
echo "  ALREADY DONE: $XGB_DONE"
echo "  PENDING: $TOTAL experiments"
echo "  BATCHING: $TOTAL_JOBS jobs × up to $TASKS_PER_JOB tasks/job (${PARALLEL} parallel)"
echo "  RESOURCES: ${CPUS_PER_TASK}CPU × ${PARALLEL} = $((CPUS_PER_TASK * PARALLEL)) CPUs, ${MEM_PER_TASK_GB}GB × ${PARALLEL} = $((MEM_PER_TASK_GB * PARALLEL))GB"
echo "  WALLTIME: $WALLTIME"
echo "  DRY_RUN: $DRY_RUN"
echo ""

{
    echo "# XGBoost Validation batch launch: $(date)"
    echo "# SEEDS: ${SEEDS[*]}"
    echo "# PENDING: $TOTAL experiments → $TOTAL_JOBS batch jobs"
    echo ""
} > "$LOG_FILE"

if [[ $TOTAL -eq 0 ]]; then
    echo "[DONE] All XGBoost experiments already completed."
    exit 0
fi

# ============================================================
# Group into batches and submit
# ============================================================
echo "Submitting batches of $TASKS_PER_JOB tasks ($PARALLEL parallel)..."
echo ""

SUBMITTED=0
BATCH_IDX=0
i=0

while [[ $i -lt $TOTAL ]]; do
    ((BATCH_IDX++))

    # Write task file for this batch
    TASK_FILE="$TASK_DIR/xgb_batch_${TIMESTAMP}_${BATCH_IDX}.txt"
    BATCH_SIZE=0
    for (( j=i; j < i + TASKS_PER_JOB && j < TOTAL; j++ )); do
        echo "${ALL_TASKS[$j]}" >> "$TASK_FILE"
        ((BATCH_SIZE++))
    done

    # Resources scaled to actual batch size (last batch may be smaller)
    ACTUAL_PAR=$((BATCH_SIZE < PARALLEL ? BATCH_SIZE : PARALLEL))
    ACTUAL_CPUS=$((CPUS_PER_TASK * ACTUAL_PAR))
    ACTUAL_MEM=$((MEM_PER_TASK_GB * ACTUAL_PAR))
    JOB_NAME="xgb_b${BATCH_IDX}"

    if $DRY_RUN; then
        echo "[DRY] $JOB_NAME: $BATCH_SIZE tasks (${ACTUAL_CPUS}CPU/${ACTUAL_MEM}GB) — $TASK_FILE"
    else
        submitted=false
        while [[ "$submitted" == "false" ]]; do
            wait_for_slot
            QUEUE=$(pick_queue)

            JOB_ID=$(qsub -N "$JOB_NAME" \
                -l "select=1:ncpus=${ACTUAL_CPUS}:mem=${ACTUAL_MEM}gb" \
                -l "walltime=${WALLTIME}" \
                -q "$QUEUE" \
                -v "TASK_FILE=${TASK_FILE},PARALLEL=${ACTUAL_PAR}" \
                "$BATCH_SCRIPT" 2>&1)
            if [[ $? -eq 0 ]]; then
                echo "[SUBMIT] $JOB_NAME ($BATCH_SIZE tasks, ${ACTUAL_PAR}par) → $QUEUE → $JOB_ID" | tee -a "$LOG_FILE"
                ((SUBMITTED += BATCH_SIZE))
                submitted=true
                sleep 0.5
            else
                echo "[RETRY] $JOB_NAME failed ($QUEUE): $JOB_ID — waiting..." | tee -a "$LOG_FILE"
                sleep "$POLL_INTERVAL"
            fi
        done
    fi

    i=$((i + TASKS_PER_JOB))
done

{
    echo ""
    echo "# Completed: $(date)"
    echo "# Submitted: $SUBMITTED experiments in $BATCH_IDX batch jobs"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Pending experiments: $TOTAL"
echo "  Batch jobs:          $BATCH_IDX"
echo "  Tasks/job:           $TASKS_PER_JOB ($PARALLEL parallel)"
echo "  Submitted:           $SUBMITTED"
echo "  Walltime/job:        $WALLTIME"
echo "  Task files:          $TASK_DIR/xgb_batch_${TIMESTAMP}_*.txt"
echo "  Log: $LOG_FILE"
echo "============================================================"

#!/bin/bash
# ============================================================
# Experiment 3: 15-seed prior research launcher (domain_train)
# ============================================================
# Extends exp3 from 2 seeds to 15 seeds for statistical robustness.
# Only submits seeds that don't already have completed results.
#
# Seed list (15 common ML seeds):
#   42, 123         — already completed (exp3 original)
#   0, 1, 3, 7, 13  — used in exp2
#   99, 256, 512     — used in exp2
#   777, 999, 1337   — used in exp2
#   2024, 1234, 2025 — common ML seeds
#
# Models: SvmA, Lstm (SvmW excluded — TN=0 degenerate issue)
# Per model: 3 distances × 2 domains × 15 seeds × 7 conditions = 630 jobs
# Total: 2 models × 630 = 1260 jobs
# New jobs (13 seeds only): 2 models × 546 = 1092 jobs
#
# Usage:
#   bash scripts/hpc/launchers/launch_exp3_15seeds.sh --dry-run
#   nohup bash scripts/hpc/launchers/launch_exp3_15seeds.sh \
#     > scripts/hpc/logs/train/exp3_15seeds_output.log 2>&1 &
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
CPU_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
GPU_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh"
EVAL_DIR="$PROJECT_ROOT/results/outputs/evaluation"

# ============================================================
# 15 common ML seeds
# ============================================================
ALL_SEEDS=(42 123 0 1 3 7 13 99 256 512 777 999 1337 2024 1234)

RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"

DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")

# SvmW excluded: all domain_train results are degenerate (TN=0)
MODELS=("SvmA" "Lstm")

# Queue limits (KAGAYAKI per-user limits)
MAX_TOTAL_JOBS=170
POLL_INTERVAL=120  # seconds between queue checks when waiting for slots
MAX_PER_BATCH=40   # max jobs to submit in one batch before re-checking queue

# Parse arguments
DRY_RUN=false
DAEMON_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --daemon)
            DAEMON_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/exp3_15seeds_${TIMESTAMP}.log"

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# ============================================================
# Pre-cache completed eval files for fast lookup
# ============================================================
echo "Building completion cache..."
CACHE_FILE=$(mktemp)
find "$EVAL_DIR" -name "eval_results_*domain_train*_within.json" 2>/dev/null \
    | xargs -I{} basename {} > "$CACHE_FILE"
CACHE_COUNT=$(wc -l < "$CACHE_FILE")
echo "Cached $CACHE_COUNT completed eval files."
trap 'rm -f "$CACHE_FILE"' EXIT

is_completed() {
    local model="$1" condition="$2" distance="$3" domain="$4" seed="$5" ratio="$6"

    if [[ "$condition" == "baseline" ]]; then
        grep -q "${model}_domain_train.*_${distance}_${domain}_.*_s${seed}_within\.json" "$CACHE_FILE"
    else
        grep -q "${model}_domain_train.*_${condition}_.*_${distance}_${domain}_.*ratio${ratio}_s${seed}_within\.json" "$CACHE_FILE"
    fi
}

# ============================================================
# Queue capacity management
# ============================================================
# Per-queue limits (KAGAYAKI)
declare -A QUEUE_LIMITS=(
    [SINGLE]=40 [LONG]=15 [DEFAULT]=10 [SMALL]=10
    [GPU-1]=40 [GPU-1A]=40 [GPU-S]=40 [GPU-L]=40 [GPU-LA]=40
)

get_total_queue_count() {
    qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l
}

get_per_queue_counts() {
    # Populates QUEUE_COUNTS associative array
    declare -gA QUEUE_COUNTS=()
    while IFS=' ' read -r count queue; do
        QUEUE_COUNTS["$queue"]=$count
    done < <(qstat -u s2240011 2>/dev/null | tail -n +6 | awk '{print $3}' | sort | uniq -c | awk '{print $1, $2}')
}

wait_for_slots() {
    local queue="$1"
    local limit="${QUEUE_LIMITS[$queue]:-40}"
    while true; do
        local total
        total=$(get_total_queue_count)
        if [[ $total -ge $MAX_TOTAL_JOBS ]]; then
            echo "[WAIT] Total: $total/$MAX_TOTAL_JOBS — waiting ($(date +%H:%M:%S))" | tee -a "$LOG_FILE"
            sleep "$POLL_INTERVAL"
            continue
        fi
        get_per_queue_counts
        local qcount="${QUEUE_COUNTS[$queue]:-0}"
        if [[ $qcount -lt $limit ]]; then
            return 0
        fi
        echo "[WAIT] $queue: $qcount/$limit full — waiting ($(date +%H:%M:%S))" | tee -a "$LOG_FILE"
        sleep "$POLL_INTERVAL"
    done
}

# ============================================================
# Helper: pick least-loaded queue from a list
# ============================================================
pick_queue() {
    local -a candidates=("$@")
    get_per_queue_counts
    local best_queue="${candidates[0]}"
    local best_count=99999
    for q in "${candidates[@]}"; do
        local cnt="${QUEUE_COUNTS[$q]:-0}"
        local lim="${QUEUE_LIMITS[$q]:-40}"
        local avail=$((lim - cnt))
        if [[ $avail -gt 0 && $cnt -lt $best_count ]]; then
            best_count=$cnt
            best_queue="$q"
        fi
    done
    echo "$best_queue"
}

get_queue_cpu() {
    pick_queue "SINGLE" "LONG" "DEFAULT" "SMALL"
}

get_queue_gpu() {
    pick_queue "GPU-1" "GPU-1A" "GPU-S" "GPU-L" "GPU-LA"
}

# ============================================================
# Build job list (skip already completed)
# ============================================================
echo "============================================================"
echo "Experiment 3: 15-seed launcher (domain_train)"
echo "============================================================"
echo "Seeds: ${ALL_SEEDS[*]}"
echo "Models: ${MODELS[*]}"
echo "Distances: ${DISTANCES[*]}"
echo "Domains: ${DOMAINS[*]}"
echo "Dry run: $DRY_RUN"
echo "Daemon mode: $DAEMON_MODE"
echo ""
echo "Scanning for already completed conditions..."
echo ""

declare -a PENDING_JOBS=()

for MODEL in "${MODELS[@]}"; do
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for SEED in "${ALL_SEEDS[@]}"; do
                # Baseline
                if ! is_completed "$MODEL" "baseline" "$DISTANCE" "$DOMAIN" "$SEED" ""; then
                    PENDING_JOBS+=("$MODEL|baseline|$DISTANCE|$DOMAIN|$SEED|")
                fi
                # Ratio-based conditions
                for RATIO in "${RATIOS[@]}"; do
                    for COND in smote_plain smote undersample; do
                        if ! is_completed "$MODEL" "$COND" "$DISTANCE" "$DOMAIN" "$SEED" "$RATIO"; then
                            PENDING_JOBS+=("$MODEL|$COND|$DISTANCE|$DOMAIN|$SEED|$RATIO")
                        fi
                    done
                done
            done
        done
    done
done

TOTAL_PENDING=${#PENDING_JOBS[@]}
echo "Pending jobs: $TOTAL_PENDING"
echo ""

{
    echo "# Exp3 15-seed launch started at $(date)"
    echo "# Seeds: ${ALL_SEEDS[*]}"
    echo "# Models: ${MODELS[*]}"
    echo "# Pending: $TOTAL_PENDING"
    echo ""
} > "$LOG_FILE"

if [[ $TOTAL_PENDING -eq 0 ]]; then
    echo "All jobs already completed!"
    exit 0
fi

# ============================================================
# Submit jobs
# ============================================================
SUBMITTED=0
SKIPPED=0
FAILED=0

for job_spec in "${PENDING_JOBS[@]}"; do
    IFS='|' read -r MODEL CONDITION DISTANCE DOMAIN SEED RATIO <<< "$job_spec"

    # Build job name
    local_cond_short="${CONDITION:0:2}"
    local_dist_short="${DISTANCE:0:2}"
    local_dom_short="${DOMAIN:0:1}"
    JOB_NAME="${MODEL:0:2}_${local_cond_short}_${local_dist_short}_${local_dom_short}_s${SEED}"

    # Select script, resources, and queue (pick least-loaded)
    if [[ "$MODEL" == "Lstm" ]]; then
        SCRIPT="$GPU_SCRIPT"
        NCPUS_MEM="ncpus=4:mem=16gb:ngpus=1"
        WALLTIME="20:00:00"
        if $DRY_RUN; then
            QUEUE="GPU-auto"
        else
            QUEUE=$(get_queue_gpu)
        fi
    else
        # SvmA (72h: smote/smote_plain + mmd + ratio=0.5 can exceed 48h)
        SCRIPT="$CPU_SCRIPT"
        NCPUS_MEM="ncpus=8:mem=32gb"
        WALLTIME="72:00:00"
        if $DRY_RUN; then
            QUEUE="CPU-auto"
        else
            QUEUE=$(get_queue_cpu)
        fi
    fi

    # Wait for queue slots (unless dry run)
    if ! $DRY_RUN; then
        wait_for_slots "$QUEUE"
    fi

    # Build qsub command
    VARS="MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    if [[ -n "$RATIO" ]]; then
        VARS="$VARS,RATIO=$RATIO"
    fi

    CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v $VARS $SCRIPT"

    if $DRY_RUN; then
        local_ratio_str=""
        [[ -n "$RATIO" ]] && local_ratio_str=" r=$RATIO"
        echo "[DRY-RUN] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | s$SEED${local_ratio_str} → $QUEUE"
        ((SUBMITTED++))
    else
        # Retry with backoff if queue is temporarily full
        local attempts=0
        local max_attempts=5
        while true; do
            JOB_ID=$(eval "$CMD" 2>&1)
            if [[ $? -eq 0 ]]; then
                echo "[SUBMIT] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | s$SEED | r=$RATIO → $JOB_ID ($QUEUE)"
                echo "OK:$MODEL:$CONDITION:$DISTANCE:$DOMAIN:$SEED:$RATIO:$JOB_ID" >> "$LOG_FILE"
                ((SUBMITTED++))
                break
            fi
            ((attempts++))
            if [[ $attempts -ge $max_attempts ]]; then
                echo "[ERROR] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | s$SEED | r=$RATIO → FAILED after $max_attempts attempts: $JOB_ID"
                echo "FAIL:$MODEL:$CONDITION:$DISTANCE:$DOMAIN:$SEED:$RATIO:$JOB_ID" >> "$LOG_FILE"
                ((FAILED++))
                break
            fi
            echo "[RETRY] $QUEUE full, waiting $POLL_INTERVAL s... (attempt $attempts/$max_attempts)" | tee -a "$LOG_FILE"
            sleep "$POLL_INTERVAL"
            # Re-pick queue on retry (might find a less loaded one)
            if [[ "$MODEL" == "Lstm" ]]; then
                QUEUE=$(get_queue_gpu)
            else
                QUEUE=$(get_queue_cpu)
            fi
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v $VARS $SCRIPT"
            wait_for_slots "$QUEUE"
        done
        sleep 0.3

        # Pause periodically if submitting many jobs
        if [[ $((SUBMITTED % MAX_PER_BATCH)) -eq 0 && $SUBMITTED -gt 0 ]]; then
            echo "[INFO] Batch of $MAX_PER_BATCH submitted, checking queue..." | tee -a "$LOG_FILE"
            sleep 2
        fi
    fi
done

# ============================================================
# Summary
# ============================================================
{
    echo ""
    echo "# Completed at $(date)"
    echo "# Submitted: $SUBMITTED"
    echo "# Failed: $FAILED"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "Submission Summary"
echo "============================================================"
echo "Total pending: $TOTAL_PENDING"
echo "Submitted: $SUBMITTED"
echo "Failed: $FAILED"
echo ""
echo "Job breakdown per model:"
echo "  SvmA: 3 dist × 2 dom × 15 seeds × 7 cond = 630 jobs"
echo "  Lstm: 3 dist × 2 dom × 15 seeds × 7 cond = 630 jobs"
echo "  Total: 1260 jobs (new seeds only: ~1092)"
echo ""
echo "Log file: $LOG_FILE"
echo "============================================================"

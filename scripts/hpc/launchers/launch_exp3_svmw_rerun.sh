#!/bin/bash
# ============================================================
# Experiment 3: SvmW re-run with tunable C (domain_train)
# ============================================================
# Re-runs SvmW after fixing the TN=0 degenerate issue:
#   - C was previously hardcoded at 300.0
#   - Now C is tuned by Optuna in [0.01, 1000] (log scale)
#
# Old degenerate eval files have been quarantined to:
#   results/outputs/evaluation/SvmW_quarantine_c300_fixed/
#
# Per model: 3 distances × 2 domains × 15 seeds × 7 conditions = 630 jobs
# SvmW uses CPU queues: SINGLE, LONG, DEFAULT, SMALL
#
# Usage:
#   bash scripts/hpc/launchers/launch_exp3_svmw_rerun.sh --dry-run
#   nohup bash scripts/hpc/launchers/launch_exp3_svmw_rerun.sh \
#     > scripts/hpc/logs/train/exp3_svmw_rerun_output.log 2>&1 &
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
CPU_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
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

MODELS=("SvmW")

# Queue limits (KAGAYAKI per-user limits)
MAX_TOTAL_JOBS=170
POLL_INTERVAL=120  # seconds between queue checks when waiting for slots
MAX_PER_BATCH=40   # max jobs to submit in one batch before re-checking queue

# Parse arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
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
LOG_FILE="$LOG_DIR/exp3_svmw_rerun_${TIMESTAMP}.log"

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# ============================================================
# Pre-cache completed eval files for fast lookup
# ============================================================
echo "Building completion cache..."
CACHE_FILE=$(mktemp)
find "$EVAL_DIR" -name "eval_results_SvmW*domain_train*_within.json" 2>/dev/null \
    | xargs -I{} basename {} > "$CACHE_FILE"
CACHE_COUNT=$(wc -l < "$CACHE_FILE")
echo "Cached $CACHE_COUNT completed SvmW eval files."
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
declare -A QUEUE_LIMITS=(
    [SINGLE]=40 [LONG]=15 [DEFAULT]=10 [SMALL]=10
)

get_total_queue_count() {
    qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l
}

get_per_queue_counts() {
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

# ============================================================
# Build job list (skip already completed)
# ============================================================
echo "============================================================"
echo "Exp3: SvmW re-run (C tunable, domain_train)"
echo "============================================================"
echo "Seeds: ${ALL_SEEDS[*]}"
echo "Distances: ${DISTANCES[*]}"
echo "Domains: ${DOMAINS[*]}"
echo "Dry run: $DRY_RUN"
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
    echo "# Exp3 SvmW re-run (C tunable) started at $(date)"
    echo "# Seeds: ${ALL_SEEDS[*]}"
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
FAILED=0

for job_spec in "${PENDING_JOBS[@]}"; do
    IFS='|' read -r MODEL CONDITION DISTANCE DOMAIN SEED RATIO <<< "$job_spec"

    # Build job name
    local_cond_short="${CONDITION:0:2}"
    local_dist_short="${DISTANCE:0:2}"
    local_dom_short="${DOMAIN:0:1}"
    JOB_NAME="Sw_${local_cond_short}_${local_dist_short}_${local_dom_short}_s${SEED}"

    # SvmW: CPU queue, 8 cores, 32GB RAM, 48h walltime
    SCRIPT="$CPU_SCRIPT"
    NCPUS_MEM="ncpus=8:mem=32gb"
    WALLTIME="48:00:00"
    if $DRY_RUN; then
        QUEUE="CPU-auto"
    else
        QUEUE=$(get_queue_cpu)
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
        echo "[DRY-RUN] SvmW | $CONDITION | $DISTANCE | $DOMAIN | s$SEED${local_ratio_str} → $QUEUE"
        ((SUBMITTED++))
    else
        local attempts=0
        local max_attempts=5
        while true; do
            JOB_ID=$(eval "$CMD" 2>&1)
            if [[ $? -eq 0 ]]; then
                echo "[SUBMIT] SvmW | $CONDITION | $DISTANCE | $DOMAIN | s$SEED | r=$RATIO → $JOB_ID ($QUEUE)"
                echo "OK:SvmW:$CONDITION:$DISTANCE:$DOMAIN:$SEED:$RATIO:$JOB_ID" >> "$LOG_FILE"
                ((SUBMITTED++))
                break
            fi
            ((attempts++))
            if [[ $attempts -ge $max_attempts ]]; then
                echo "[ERROR] SvmW | $CONDITION | $DISTANCE | $DOMAIN | s$SEED | r=$RATIO → FAILED after $max_attempts attempts: $JOB_ID"
                echo "FAIL:SvmW:$CONDITION:$DISTANCE:$DOMAIN:$SEED:$RATIO:$JOB_ID" >> "$LOG_FILE"
                ((FAILED++))
                break
            fi
            echo "[RETRY] $QUEUE full, waiting $POLL_INTERVAL s... (attempt $attempts/$max_attempts)" | tee -a "$LOG_FILE"
            sleep "$POLL_INTERVAL"
            QUEUE=$(get_queue_cpu)
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v $VARS $SCRIPT"
            wait_for_slots "$QUEUE"
        done
        sleep 0.3

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
echo "Log: $LOG_FILE"

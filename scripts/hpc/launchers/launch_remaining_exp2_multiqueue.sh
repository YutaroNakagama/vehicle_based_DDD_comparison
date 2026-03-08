#!/bin/bash
# ============================================================
# Experiment 2 remaining jobs — multi-queue distribution submit (2026-02-07)
# ============================================================
# SINGLE Queue full, distributing submissions to DEFAULT / SMALL.
#
# Queue selection rules:
#   DEFAULT : max_user_queued=40, ncpus≤64, walltime≤168h  → available
#   SMALL   : max_user_queued=30, ncpus≤768, walltime≤168h → available
#   LONG    : max_user_queued=15, ncpus≤128, walltime≤504h → if remaining
#
# remaining 108 job(s):
#   smote_plain remaining:  10 domain + 2 pooled = 12
#   smote:             48 domain + 2 pooled = 50
#   undersample:       48 domain + 2 pooled = 50
#                                       total = 112 (4 items retried in previous rounds)
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
DOMAIN_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
POOLED_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/imbalance/pbs_imbalance_comparison.sh"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("in_domain" "out_domain")
MODES=("source_only" "target_only")

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_remaining_exp2_multiqueue_${TIMESTAMP}.log"

JOB_COUNT=0
FAIL_COUNT=0

# Queue counters (track per-queue submissions to respect limits)
DEFAULT_COUNT=0
DEFAULT_LIMIT=40
SMALL_COUNT=0
SMALL_LIMIT=30
LONG_COUNT=0
LONG_LIMIT=15  # actual per-user max
SEMINAR_COUNT=0
SEMINAR_LIMIT=25  # conservative (no explicit limit found)

# --- Pick an available queue ---
pick_queue() {
    # Prefer DEFAULT first, then SMALL, then LONG, then SEMINAR
    if [[ $DEFAULT_COUNT -lt $DEFAULT_LIMIT ]]; then
        echo "DEFAULT"
    elif [[ $SMALL_COUNT -lt $SMALL_LIMIT ]]; then
        echo "SMALL"
    elif [[ $LONG_COUNT -lt $LONG_LIMIT ]]; then
        echo "LONG"
    elif [[ $SEMINAR_COUNT -lt $SEMINAR_LIMIT ]]; then
        echo "SEMINAR"
    else
        echo "FULL"
    fi
}

increment_queue_count() {
    local q="$1"
    case "$q" in
        DEFAULT) ((DEFAULT_COUNT++)) ;;
        SMALL)   ((SMALL_COUNT++))   ;;
        LONG)    ((LONG_COUNT++))    ;;
        SEMINAR) ((SEMINAR_COUNT++)) ;;
    esac
}

# --- Helper: submit domain-split job ---
submit_domain() {
    local condition="$1" mode="$2" distance="$3" domain="$4" seed="$5" ratio="${6:-}"

    local ncpus_mem walltime
    case "$condition" in
        smote|smote_plain) ncpus_mem="ncpus=4:mem=10gb"; walltime="08:00:00" ;;
        undersample)       ncpus_mem="ncpus=4:mem=8gb";  walltime="06:00:00" ;;
        baseline)          ncpus_mem="ncpus=4:mem=8gb";  walltime="06:00:00" ;;
        balanced_rf)       ncpus_mem="ncpus=8:mem=12gb"; walltime="08:00:00" ;;
        *)                 ncpus_mem="ncpus=4:mem=8gb";  walltime="08:00:00" ;;
    esac

    local queue
    queue=$(pick_queue)
    if [[ "$queue" == "FULL" ]]; then
        echo "[SKIP] All queues full after $JOB_COUNT submitted."
        ((FAIL_COUNT++))
        return 1
    fi

    local cond_short="${condition:0:2}"
    local job_name="${cond_short}_${distance:0:1}${domain:0:1}_${mode:0:1}"
    [[ -n "$ratio" ]] && job_name="${job_name}_r${ratio}_s${seed}" || job_name="${job_name}_s${seed}"

    local vars="CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    [[ -n "$ratio" ]] && vars="$vars,RATIO=$ratio"

    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $vars $DOMAIN_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] [$queue] $condition | $distance | $domain | $mode | r=${ratio:-N/A} | s$seed"
        increment_queue_count "$queue"
        ((JOB_COUNT++))
    else
        local job_id
        if job_id=$(eval "$cmd" 2>&1); then
            echo "[SUBMIT] [$queue] $condition | $distance | $domain | $mode | r=${ratio:-N/A} | s$seed → $job_id"
            echo "$queue:$condition:$distance:$domain:$mode:${ratio:-N/A}:$seed:$job_id" >> "$LOG_FILE"
            increment_queue_count "$queue"
            ((JOB_COUNT++))
        else
            echo "[FAIL]   [$queue] $condition | r=${ratio:-N/A} | s$seed — $job_id"
            ((FAIL_COUNT++))
        fi
        sleep 0.2
    fi
}

# --- Helper: submit pooled job ---
submit_pooled() {
    local method="$1" seed="$2" ratio="${3:-0.5}"

    local ncpus_mem walltime
    case "$method" in
        smote|smote_plain) ncpus_mem="ncpus=4:mem=10gb"; walltime="08:00:00" ;;
        undersample*|baseline) ncpus_mem="ncpus=4:mem=8gb"; walltime="06:00:00" ;;
        *)                 ncpus_mem="ncpus=4:mem=8gb"; walltime="08:00:00" ;;
    esac

    local queue
    queue=$(pick_queue)
    if [[ "$queue" == "FULL" ]]; then
        echo "[SKIP] All queues full."
        ((FAIL_COUNT++))
        return 1
    fi

    local pbs_method="$method"
    case "$method" in
        smote)       pbs_method="smote_subjectwise" ;;
        undersample) pbs_method="undersample_rus"   ;;
    esac

    local job_name="pl_${method:0:2}_s${seed}"
    local vars="METHOD=$pbs_method,SEED=$seed,RATIO=$ratio,N_TRIALS=$N_TRIALS,RUN_EVAL=true"
    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $vars $POOLED_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] [$queue] pooled | $method | s$seed"
        increment_queue_count "$queue"
        ((JOB_COUNT++))
    else
        local job_id
        if job_id=$(eval "$cmd" 2>&1); then
            echo "[SUBMIT] [$queue] pooled | $method | s$seed → $job_id"
            echo "$queue:pooled:$method:$seed:$job_id" >> "$LOG_FILE"
            increment_queue_count "$queue"
            ((JOB_COUNT++))
        else
            echo "[FAIL]   [$queue] pooled | $method | s$seed — $job_id"
            ((FAIL_COUNT++))
        fi
        sleep 0.2
    fi
}

# ============================================================
echo "============================================================"
echo "  Experiment 2 remaining jobs — multi-queue distribution submit"
echo "  $(date)"
echo "============================================================"
echo "  Dry run : $DRY_RUN"
echo "  Queues  : DEFAULT (≤$DEFAULT_LIMIT), SMALL (≤$SMALL_LIMIT), LONG (≤$LONG_LIMIT), SEMINAR (≤$SEMINAR_LIMIT)"
echo "  Expected: ~108 jobs"
echo ""

{
    echo "# Multi-queue launch started at $(date)"
    echo "# Queues: DEFAULT/$DEFAULT_LIMIT, SMALL/$SMALL_LIMIT, LONG/$LONG_LIMIT, SEMINAR/$SEMINAR_LIMIT"
    echo ""
} > "$LOG_FILE"

# --- smote_plain remaining (wasserstein partial + pooled) ---
echo "--- smote_plain remaining (6 domain + 2 pooled) ---"
# wasserstein/out_domain/source_only s123 remaining (s42 already submitted in retry)
for RATIO in "${RATIOS[@]}"; do
    submit_domain "smote_plain" "source_only" "wasserstein" "out_domain" "123" "$RATIO"
done
# wasserstein/out_domain/target_only all 4
for SEED in "${SEEDS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        submit_domain "smote_plain" "target_only" "wasserstein" "out_domain" "$SEED" "$RATIO"
    done
done
# pooled
for SEED in "${SEEDS[@]}"; do
    submit_pooled "smote_plain" "$SEED"
done

# --- smote: all 48 domain + 2 pooled ---
echo ""
echo "--- smote all 50 jobs ---"
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                for RATIO in "${RATIOS[@]}"; do
                    submit_domain "smote" "$MODE" "$DISTANCE" "$DOMAIN" "$SEED" "$RATIO"
                done
            done
        done
    done
done
for SEED in "${SEEDS[@]}"; do
    submit_pooled "smote" "$SEED"
done

# --- undersample: all 48 domain + 2 pooled ---
echo ""
echo "--- undersample all 50 jobs ---"
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                for RATIO in "${RATIOS[@]}"; do
                    submit_domain "undersample" "$MODE" "$DISTANCE" "$DOMAIN" "$SEED" "$RATIO"
                done
            done
        done
    done
done
for SEED in "${SEEDS[@]}"; do
    submit_pooled "undersample" "$SEED"
done

# --- Summary ---
{
    echo ""
    echo "# Completed at $(date)"
    echo "# Submitted: $JOB_COUNT (DEFAULT=$DEFAULT_COUNT, SMALL=$SMALL_COUNT, LONG=$LONG_COUNT, SEMINAR=$SEMINAR_COUNT)"
    echo "# Failed: $FAIL_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "  Dry run: would submit $JOB_COUNT jobs"
else
    echo "  Submitted: $JOB_COUNT jobs"
    echo "  Failed:    $FAIL_COUNT jobs"
fi
echo "  DEFAULT: $DEFAULT_COUNT / $DEFAULT_LIMIT"
echo "  SMALL:   $SMALL_COUNT / $SMALL_LIMIT"
echo "  LONG:    $LONG_COUNT / $LONG_LIMIT"
echo "  SEMINAR: $SEMINAR_COUNT / $SEMINAR_LIMIT"
echo "  Log:     $LOG_FILE"
echo "============================================================"

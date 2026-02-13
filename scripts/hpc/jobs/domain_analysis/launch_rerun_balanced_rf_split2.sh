#!/bin/bash
# ============================================================
# Rerun BalancedRF experiments for split2
# ============================================================
# Purpose: Re-execute all 36 BalancedRF configurations after
#          fixing the random_state hardcode bug (commit 10bd2f9).
#          Previously, random_state=42 was hardcoded in model
#          factory/classifiers/optuna_tuning, making seed=42
#          and seed=123 produce identical hard predictions.
#
# Configurations: 3 modes × 3 distances × 2 domains × 2 seeds = 36
#
# Queue strategy: Distributes jobs across DEFAULT, SINGLE,
#                 SMALL, and LONG queues to maximize throughput.
#                 Skips queues that are at their per-user limit.
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# PBS settings
NCPUS=4
MEM="32gb"
SCRIPT_PATH="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

# Queue definitions: name, walltime, max_queued_per_user
QUEUES=("DEFAULT:24:00:00:40" "SINGLE:24:00:00:40" "SMALL:24:00:00:30" "LONG:24:00:00:15")

# Create log directory
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"

# Dry run mode (set DRY_RUN=true to preview without submitting)
DRY_RUN="${DRY_RUN:-false}"

# Experiment configurations
DISTANCES=("dtw" "mmd" "wasserstein")
DOMAINS=("in_domain" "out_domain")
MODES=("source_only" "target_only" "mixed")
SEEDS=(42 123)

CONDITION="balanced_rf"
N_TRIALS=150
RANKING="knn"

echo "============================================================"
echo "RERUN: BalancedRF split2 (random_state fix)"
echo "============================================================"
echo "Distances: ${DISTANCES[*]}"
echo "Domains: ${DOMAINS[*]}"
echo "Modes: ${MODES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "N_TRIALS: $N_TRIALS"
echo "DRY_RUN: $DRY_RUN"
echo "============================================================"

# Build job list
declare -a JOB_CONFIGS=()
for dist in "${DISTANCES[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        for mode in "${MODES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                JOB_CONFIGS+=("${dist}:${domain}:${mode}:${seed}")
            done
        done
    done
done

echo "Total configurations: ${#JOB_CONFIGS[@]}"
echo ""

# Count current jobs per queue
count_queue_jobs() {
    local queue=$1
    qstat -u "$USER" 2>/dev/null | awk -v q="$queue" 'NR>5 && $3==q' | wc -l
}

job_count=0
submitted_jobs=()
failed_jobs=()

for config in "${JOB_CONFIGS[@]}"; do
    IFS=':' read -r dist domain mode seed <<< "$config"

    # Generate job name
    case "$mode" in
        source_only) mode_short="so" ;;
        target_only) mode_short="to" ;;
        mixed)       mode_short="mx" ;;
    esac
    case "$domain" in
        in_domain)  dom_short="in" ;;
        out_domain) dom_short="ou" ;;
    esac
    JOB_NAME="bf_${dist:0:3}_${dom_short}_${mode_short}_s${seed}"

    ((job_count++))

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[$job_count] $JOB_NAME: $dist $domain $mode s$seed"
        continue
    fi

    # Try each queue until one accepts the job
    submitted=false
    for queue_entry in "${QUEUES[@]}"; do
        IFS=':' read -r queue_name wt_h wt_m wt_s max_q <<< "$queue_entry"
        WALLTIME="${wt_h}:${wt_m}:${wt_s}"

        # Check current usage
        current=$(count_queue_jobs "$queue_name")
        if [[ "$current" -ge "$max_q" ]]; then
            continue
        fi

        JOB_ID=$(qsub \
            -N "$JOB_NAME" \
            -q "$queue_name" \
            -l select=1:ncpus=${NCPUS}:mem=${MEM} \
            -l walltime=${WALLTIME} \
            -v CONDITION="$CONDITION",MODE="$mode",DISTANCE="$dist",DOMAIN="$domain",RATIO="0",SEED="$seed",N_TRIALS="$N_TRIALS",RANKING="$RANKING",RUN_EVAL="true" \
            "$SCRIPT_PATH" 2>&1)

        if [[ $? -eq 0 ]]; then
            submitted_jobs+=("$JOB_ID:$queue_name:$JOB_NAME")
            echo "[$job_count] $JOB_NAME -> $queue_name ($JOB_ID)"
            submitted=true
            sleep 0.3
            break
        fi
    done

    if [[ "$submitted" == "false" ]]; then
        failed_jobs+=("$JOB_NAME:$dist:$domain:$mode:$seed")
        echo "[$job_count] $JOB_NAME -> FAILED (all queues full)"
    fi
done

echo ""
echo "============================================================"
echo "SUBMISSION SUMMARY"
echo "============================================================"
echo "Total configurations: ${#JOB_CONFIGS[@]}"
if [[ "$DRY_RUN" == "false" ]]; then
    echo "Submitted: ${#submitted_jobs[@]}"
    echo "Failed: ${#failed_jobs[@]}"

    if [[ ${#failed_jobs[@]} -gt 0 ]]; then
        echo ""
        echo "Failed jobs (rerun this script later when queue slots free up):"
        for job in "${failed_jobs[@]}"; do
            echo "  $job"
        done
    fi

    # Save job list
    if [[ ${#submitted_jobs[@]} -gt 0 ]]; then
        JOB_LIST_FILE="${LOG_DIR}/rerun_balanced_rf_split2_$(date +%Y%m%d_%H%M%S).txt"
        printf '%s\n' "${submitted_jobs[@]}" > "$JOB_LIST_FILE"
        echo ""
        echo "Job list saved to: $JOB_LIST_FILE"
    fi

    if [[ ${#failed_jobs[@]} -gt 0 ]]; then
        FAILED_FILE="${LOG_DIR}/rerun_balanced_rf_split2_failed_$(date +%Y%m%d_%H%M%S).txt"
        printf '%s\n' "${failed_jobs[@]}" > "$FAILED_FILE"
        echo "Failed list saved to: $FAILED_FILE"
        echo ""
        echo "To retry: bash $0"
    fi
fi
echo "============================================================"

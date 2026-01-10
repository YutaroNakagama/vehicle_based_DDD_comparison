#!/bin/bash
# ============================================================
# Retry failed domain analysis jobs with queue distribution
# ============================================================
# Distributes jobs across SINGLE, SMALL, and LONG queues
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

SCRIPT_PATH="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison.sh"
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"

# Available queues (rotate to distribute load)
QUEUES=("SINGLE" "SMALL" "LONG")
QUEUE_IDX=0

WALLTIME="168:00:00"
NCPUS=4
MEM="32gb"

DISTANCES=("dtw" "mmd" "wasserstein")
DOMAINS=("in_domain" "mid_domain" "out_domain")
MODES=("source_only" "target_only")
SEEDS=(42 123)

CONDITIONS=(
    "baseline:0"
    "smote_plain:0.1"
    "smote_plain:0.5"
    "smote:0.1"
    "smote:0.5"
    "balanced_rf:0"
    "undersample:0.1"
    "undersample:0.5"
)

N_TRIALS=150
RANKING="knn"

# Get already submitted job IDs
SUBMITTED_FILE="/tmp/submitted_jobs.txt"
qstat -u s2240011 2>/dev/null | grep -E "^[0-9]" | awk '{print $1}' | cut -d. -f1 | sort -n > "$SUBMITTED_FILE"
ALREADY_SUBMITTED=$(cat "$SUBMITTED_FILE" | wc -l)

echo "============================================================"
echo "DOMAIN ANALYSIS - QUEUE DISTRIBUTED RETRY"
echo "============================================================"
echo "Already submitted: $ALREADY_SUBMITTED jobs"
echo "Queues: ${QUEUES[*]}"
echo "============================================================"

job_count=0
submitted_count=0
skipped_count=0
submitted_jobs=()

# Track which combinations have been submitted by checking job names
get_expected_job_id() {
    local idx=$1
    # Job IDs start from 14673165 for domain jobs
    echo $((14673165 + idx - 1))
}

for dist in "${DISTANCES[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        for mode in "${MODES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for cond_entry in "${CONDITIONS[@]}"; do
                    CONDITION="${cond_entry%%:*}"
                    RATIO="${cond_entry##*:}"
                    
                    ((job_count++))
                    
                    # Generate job name (same as original)
                    if [[ "$CONDITION" == "baseline" || "$CONDITION" == "balanced_rf" ]]; then
                        JOB_NAME="dom_${CONDITION:0:4}_${dist:0:3}_${domain:0:3}_${mode:0:3}_s${seed}"
                    else
                        JOB_NAME="dom_${CONDITION:0:4}_r${RATIO}_${dist:0:3}_${domain:0:3}_${mode:0:3}_s${seed}"
                    fi
                    JOB_NAME="${JOB_NAME:0:15}"
                    
                    # Check if already submitted (by job index)
                    expected_id=$(get_expected_job_id $job_count)
                    if grep -q "^${expected_id}$" "$SUBMITTED_FILE" 2>/dev/null; then
                        ((skipped_count++))
                        continue
                    fi
                    
                    # Select queue (rotate)
                    QUEUE="${QUEUES[$QUEUE_IDX]}"
                    QUEUE_IDX=$(( (QUEUE_IDX + 1) % ${#QUEUES[@]} ))
                    
                    echo "[$job_count] $JOB_NAME -> $QUEUE: $CONDITION $dist $domain $mode s$seed r$RATIO"
                    
                    JOB_ID=$(qsub \
                        -N "$JOB_NAME" \
                        -q "$QUEUE" \
                        -l select=1:ncpus=${NCPUS}:mem=${MEM} \
                        -l walltime=${WALLTIME} \
                        -v CONDITION="$CONDITION",MODE="$mode",DISTANCE="$dist",DOMAIN="$domain",RATIO="$RATIO",SEED="$seed",N_TRIALS="$N_TRIALS",RANKING="$RANKING",RUN_EVAL="true" \
                        "$SCRIPT_PATH" 2>&1)
                    
                    if [[ "$JOB_ID" == *"spcc-adm1"* ]]; then
                        ((submitted_count++))
                        submitted_jobs+=("$JOB_ID:$JOB_NAME:$QUEUE")
                        echo "  -> Submitted: $JOB_ID"
                    else
                        echo "  -> Failed: $JOB_ID"
                    fi
                    
                    sleep 0.3
                done
            done
        done
    done
done

echo ""
echo "============================================================"
echo "SUBMISSION COMPLETE"
echo "============================================================"
echo "Total combinations: $job_count"
echo "Already submitted (skipped): $skipped_count"
echo "Newly submitted: $submitted_count"
echo "============================================================"

if [[ ${#submitted_jobs[@]} -gt 0 ]]; then
    JOB_LIST_FILE="${LOG_DIR}/domain_imbalv2_retry_$(date +%Y%m%d_%H%M%S).txt"
    printf '%s\n' "${submitted_jobs[@]}" > "$JOB_LIST_FILE"
    echo "Job list saved to: $JOB_LIST_FILE"
fi

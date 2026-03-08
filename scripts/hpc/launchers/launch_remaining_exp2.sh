#!/bin/bash
# ============================================================
# Experiment 2 remaining job submit launcher (2026-02-07)
# ============================================================
# Completed:
#   - baseline   domain-split 24/24, pooled 0/2
#   - balanced_rf domain-split 24/24, pooled 2/2  ✅
#
# Not started:
#   - smote_plain  domain-split 0/48, pooled 0/2
#   - smote        domain-split 0/48, pooled 0/2
#   - undersample  domain-split 0/48, pooled 0/2
#
# Submit job count:
#   baseline  pooled ×2 seeds                   =   2
#   smote_plain  24 combos × 2 ratios + 2 pooled = 50
#   smote        24 combos × 2 ratios + 2 pooled = 50
#   undersample  24 combos × 2 ratios + 2 pooled = 50
#                                          Total = 152
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

# Parse arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_remaining_exp2_${TIMESTAMP}.log"

JOB_COUNT=0
FAIL_COUNT=0

# ------ Helper: submit domain-split job ------
submit_domain() {
    local condition="$1" mode="$2" distance="$3" domain="$4" seed="$5"
    local ratio="${6:-}"

    # Resources
    local ncpus_mem walltime queue
    case "$condition" in
        smote|smote_plain) ncpus_mem="ncpus=4:mem=10gb"; walltime="08:00:00"; queue="SINGLE" ;;
        undersample)       ncpus_mem="ncpus=4:mem=8gb";  walltime="06:00:00"; queue="SINGLE" ;;
        baseline)          ncpus_mem="ncpus=4:mem=8gb";  walltime="06:00:00"; queue="SINGLE" ;;
        balanced_rf)       ncpus_mem="ncpus=8:mem=12gb"; walltime="08:00:00"; queue="LONG"   ;;
        *)                 ncpus_mem="ncpus=4:mem=8gb";  walltime="08:00:00"; queue="SINGLE" ;;
    esac

    local cond_short="${condition:0:2}"
    local job_name="${cond_short}_${distance:0:1}${domain:0:1}_${mode:0:1}"
    if [[ -n "$ratio" ]]; then
        job_name="${job_name}_r${ratio}_s${seed}"
    else
        job_name="${job_name}_s${seed}"
    fi

    local vars="CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    [[ -n "$ratio" ]] && vars="$vars,RATIO=$ratio"

    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $vars $DOMAIN_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] $condition | $distance | $domain | $mode | r=${ratio:-N/A} | s$seed"
        ((JOB_COUNT++))
    else
        local job_id
        if job_id=$(eval "$cmd" 2>&1); then
            echo "[SUBMIT] $condition | $distance | $domain | $mode | r=${ratio:-N/A} | s$seed → $job_id"
            echo "$condition:$distance:$domain:$mode:${ratio:-N/A}:$seed:$job_id" >> "$LOG_FILE"
            ((JOB_COUNT++))
        else
            echo "[FAIL]   $condition | $distance | $domain | $mode | r=${ratio:-N/A} | s$seed"
            ((FAIL_COUNT++))
        fi
        sleep 0.3
    fi
}

# ------ Helper: submit pooled job ------
submit_pooled() {
    local method="$1" seed="$2"
    local ratio="${3:-0.5}"

    local ncpus_mem walltime queue
    case "$method" in
        balanced_rf)       ncpus_mem="ncpus=8:mem=12gb"; walltime="08:00:00"; queue="LONG"   ;;
        smote|smote_plain) ncpus_mem="ncpus=4:mem=10gb"; walltime="08:00:00"; queue="SINGLE" ;;
        undersample*|baseline) ncpus_mem="ncpus=4:mem=8gb";  walltime="06:00:00"; queue="SINGLE" ;;
        *)                 ncpus_mem="ncpus=4:mem=8gb";  walltime="08:00:00"; queue="SINGLE" ;;
    esac

    # Map condition names to METHOD names used by pbs_imbalance_comparison.sh
    local pbs_method="$method"
    case "$method" in
        smote)       pbs_method="smote_subjectwise" ;;
        undersample) pbs_method="undersample_rus"   ;;
    esac

    local job_name="pl_${method:0:2}_s${seed}"
    local vars="METHOD=$pbs_method,SEED=$seed,RATIO=$ratio,N_TRIALS=$N_TRIALS,RUN_EVAL=true"
    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $vars $POOLED_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] pooled | $method | s$seed"
        ((JOB_COUNT++))
    else
        local job_id
        if job_id=$(eval "$cmd" 2>&1); then
            echo "[SUBMIT] pooled | $method | s$seed → $job_id"
            echo "pooled:$method:$seed:$job_id" >> "$LOG_FILE"
            ((JOB_COUNT++))
        else
            echo "[FAIL]   pooled | $method | s$seed"
            ((FAIL_COUNT++))
        fi
        sleep 0.3
    fi
}

# ============================================================
echo "============================================================"
echo "  experiment2 remainingjob(s)submit ($(date))"
echo "============================================================"
echo "  Dry run : $DRY_RUN"
echo "  Expected: 152 jobs"
echo ""

{
    echo "# Launch started at $(date)"
    echo "# Script: $0"
    echo ""
} > "$LOG_FILE"

# --- 1) baseline pooled (2 jobs) ---
echo "--- [1/4] baseline pooled (2 jobs) ---"
for SEED in "${SEEDS[@]}"; do
    submit_pooled "baseline" "$SEED"
done

# --- 2) smote_plain: 48 domain-split + 2 pooled ---
echo ""
echo "--- [2/4] smote_plain (48 domain + 2 pooled = 50 jobs) ---"
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                for RATIO in "${RATIOS[@]}"; do
                    submit_domain "smote_plain" "$MODE" "$DISTANCE" "$DOMAIN" "$SEED" "$RATIO"
                done
            done
        done
    done
done
for SEED in "${SEEDS[@]}"; do
    submit_pooled "smote_plain" "$SEED"
done

# --- 3) smote (subject-wise): 48 domain-split + 2 pooled ---
echo ""
echo "--- [3/4] smote / subject-wise SMOTE (48 domain + 2 pooled = 50 jobs) ---"
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

# --- 4) undersample: 48 domain-split + 2 pooled ---
echo ""
echo "--- [4/4] undersample / RUS (48 domain + 2 pooled = 50 jobs) ---"
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
    echo "# Submitted: $JOB_COUNT"
    echo "# Failed: $FAIL_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "  Dry run complete. Would submit: $JOB_COUNT jobs"
else
    echo "  Submitted: $JOB_COUNT jobs"
    echo "  Failed:    $FAIL_COUNT jobs"
    echo "  Log:       $LOG_FILE"
fi
echo "============================================================"

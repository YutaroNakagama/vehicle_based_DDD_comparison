#!/bin/bash
# ============================================================
# source_only resubmit launcher (2026-02-07)
# ============================================================
# train_stages.py after source_group_name → source_domain fix,
# Resubmit 51 source_only jobs that failed with NameError.
#
#   smote (imbalv3):  3dist × 2dom × 2seed × 2ratio = 24
#   smote_plain:      dtw×in(4) + mmd×in(4) + mmd×out(4) + wass×out(2) = 14
#   undersample:      dtw×in(4) + dtw×out(1) + wass×in(4) + wass×out(4) = 13
#                                                               Total = 51
#
# Queue distribution: DEFAULT (40) → SMALL (11)
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
DOMAIN_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

N_TRIALS=100
RANKING="knn"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_source_only_retry_${TIMESTAMP}.log"

JOB_COUNT=0
FAIL_COUNT=0

# Queue list with fallback order
QUEUES=(DEFAULT SMALL LONG SEMINAR)

submit_domain() {
    local condition="$1" distance="$2" domain="$3" seed="$4" ratio="$5"
    local mode="source_only"

    local ncpus_mem walltime
    case "$condition" in
        smote|smote_plain) ncpus_mem="ncpus=4:mem=10gb"; walltime="08:00:00" ;;
        undersample)       ncpus_mem="ncpus=4:mem=8gb";  walltime="06:00:00" ;;
        *)                 ncpus_mem="ncpus=4:mem=8gb";  walltime="08:00:00" ;;
    esac

    local cond_short="${condition:0:2}"
    local job_name="rs_${cond_short}_${distance:0:1}${domain:0:1}_r${ratio}_s${seed}"
    local vars="CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true,RATIO=$ratio"

    if $DRY_RUN; then
        echo "[DRY-RUN] $condition | $distance | $domain | source_only | r=$ratio | s$seed"
        ((JOB_COUNT++))
        return 0
    fi

    # Try each queue in order until one accepts
    for queue in "${QUEUES[@]}"; do
        local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $vars $DOMAIN_SCRIPT"
        local job_id
        if job_id=$(eval "$cmd" 2>&1); then
            echo "[SUBMIT] [$queue] $condition | $distance | $domain | source_only | r=$ratio | s$seed → $job_id"
            echo "$queue:$condition:$distance:$domain:source_only:$ratio:$seed:$job_id" >> "$LOG_FILE"
            ((JOB_COUNT++))
            sleep 0.2
            return 0
        fi
    done

    # All queues rejected
    echo "[FAIL]   $condition | $distance | $domain | r=$ratio | s$seed — all queues full"
    ((FAIL_COUNT++))
}

echo "============================================================"
echo "  source_only resubmit (NameError fix)  $(date)"
echo "============================================================"

# ──────────────────────────────────────────────────
# 1. smote (imbalv3): 24 jobs
# ──────────────────────────────────────────────────
echo ""
echo "── smote (24 jobs) ──"
for dist in mmd dtw wasserstein; do
    for dom in in_domain out_domain; do
        for seed in 42 123; do
            for ratio in 0.1 0.5; do
                submit_domain smote "$dist" "$dom" "$seed" "$ratio"
            done
        done
    done
done

# ──────────────────────────────────────────────────
# 2. smote_plain: 14 jobs
# ──────────────────────────────────────────────────
echo ""
echo "── smote_plain (14 jobs) ──"

# dtw × in_domain (4)
for seed in 42 123; do
    for ratio in 0.1 0.5; do
        submit_domain smote_plain dtw in_domain "$seed" "$ratio"
    done
done

# mmd × in_domain (4)
for seed in 42 123; do
    for ratio in 0.1 0.5; do
        submit_domain smote_plain mmd in_domain "$seed" "$ratio"
    done
done

# mmd × out_domain (4)
for seed in 42 123; do
    for ratio in 0.1 0.5; do
        submit_domain smote_plain mmd out_domain "$seed" "$ratio"
    done
done

# wasserstein × out_domain (2) — seed=123 only
for ratio in 0.1 0.5; do
    submit_domain smote_plain wasserstein out_domain 123 "$ratio"
done

# ──────────────────────────────────────────────────
# 3. undersample: 13 jobs
# ──────────────────────────────────────────────────
echo ""
echo "── undersample (13 jobs) ──"

# dtw × in_domain (4)
for seed in 42 123; do
    for ratio in 0.1 0.5; do
        submit_domain undersample dtw in_domain "$seed" "$ratio"
    done
done

# dtw × out_domain × 0.5 × 123 (1)
submit_domain undersample dtw out_domain 123 0.5

# wasserstein × in_domain (4)
for seed in 42 123; do
    for ratio in 0.1 0.5; do
        submit_domain undersample wasserstein in_domain "$seed" "$ratio"
    done
done

# wasserstein × out_domain (4)
for seed in 42 123; do
    for ratio in 0.1 0.5; do
        submit_domain undersample wasserstein out_domain "$seed" "$ratio"
    done
done

echo ""
echo "============================================================"
echo "  Submitted: $JOB_COUNT   Failed: $FAIL_COUNT"
echo "  Log: $LOG_FILE"
echo "============================================================"

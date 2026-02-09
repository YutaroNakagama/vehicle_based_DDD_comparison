#!/bin/bash
# ============================================================
# SvmW 不足ジョブ再投入スクリプト (70 jobs)
# ============================================================
# imbalv3: 39 missing, smote_plain: 31 missing
# 原因: 前回の投入で walltime=6h が不十分 (SvmW Optuna 100 trials は ~7h 必要)
# 今回: walltime=12:00:00 で SEMINAR キューに投入
# ============================================================
set -uo pipefail

# Workaround: /var/tmp is full, PBS qsub needs a writable temp dir
export TMPDIR="${HOME}/tmp"
mkdir -p "$TMPDIR"

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

QUEUE="SEMINAR"
WALLTIME="12:00:00"
NCPUS_MEM="ncpus=4:mem=8gb"
N_TRIALS=100
RANKING="knn"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
LOG_FILE="$LOG_DIR/resubmit_svmw_missing_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY RUN] No jobs will be submitted."
fi

JOB_COUNT=0
FAIL_COUNT=0

submit_job() {
    local condition="$1"
    local distance="$2"
    local domain="$3"
    local mode="$4"
    local ratio="$5"
    local seed="$6"

    local c_short="${condition:0:2}"
    local d_short="${distance:0:1}"
    local dom_short="${domain:0:1}"
    local m_short="${mode:0:1}"
    local r_short=$(echo "$ratio" | tr -d '.')
    local job_name="Sw_${c_short}_${d_short}${dom_short}_${m_short}_r${r_short}_s${seed}"

    local env_vars="MODEL=SvmW,CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,RATIO=$ratio,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"

    local cmd="qsub -N $job_name -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v $env_vars $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY] $condition | $distance | $domain | $mode | r=$ratio | s$seed"
        echo "  CMD: $cmd"
        ((JOB_COUNT++))
        return 0
    fi

    if job_id=$(eval "$cmd" 2>&1); then
        echo "[OK] $condition | $distance | $domain | $mode | r=$ratio | s$seed → $job_id"
        echo "SvmW:$condition:$distance:$domain:$mode:$ratio:$seed:$job_id" >> "$LOG_FILE"
        ((JOB_COUNT++))
    else
        echo "[FAIL] $condition | $distance | $domain | $mode | r=$ratio | s$seed — $job_id"
        ((FAIL_COUNT++))
    fi
    sleep 0.1
}

echo "============================================================"
echo "  SvmW Missing Jobs Resubmission (70 jobs)"
echo "  Queue: $QUEUE | Walltime: $WALLTIME"
echo "  $(date)"
echo "============================================================"

{
    echo "# SvmW missing resubmission: $(date)"
    echo "# Queue=$QUEUE Walltime=$WALLTIME"
    echo ""
} > "$LOG_FILE"

# ========================================
# MISSING imbalv3 (subject-wise SMOTE): 39 jobs
# ========================================
echo ""
echo "--- imbalv3 (subject-wise SMOTE): 39 jobs ---"

# All ratio=0.5 (24 jobs: 3 dist × 2 dom × 2 mode × 2 seed)
for dist in dtw mmd wasserstein; do
    for dom in in_domain out_domain; do
        for mode in source_only target_only; do
            for seed in 42 123; do
                submit_job smote "$dist" "$dom" "$mode" "0.5" "$seed"
            done
        done
    done
done

# Missing ratio=0.1 (15 jobs)
# dtw: all 8 missing (in_domain×{s,t}×{42,123} + out_domain×{s×123,t×{42,123}})
submit_job smote dtw in_domain source_only 0.1 123
submit_job smote dtw in_domain source_only 0.1 42
submit_job smote dtw in_domain target_only 0.1 123
submit_job smote dtw in_domain target_only 0.1 42
submit_job smote dtw out_domain source_only 0.1 123
submit_job smote dtw out_domain target_only 0.1 123
submit_job smote dtw out_domain target_only 0.1 42
# mmd: 3 missing
submit_job smote mmd out_domain source_only 0.1 123
submit_job smote mmd out_domain target_only 0.1 123
# wasserstein: 6 missing
submit_job smote wasserstein in_domain source_only 0.1 123
submit_job smote wasserstein in_domain source_only 0.1 42
submit_job smote wasserstein in_domain target_only 0.1 123
submit_job smote wasserstein in_domain target_only 0.1 42
submit_job smote wasserstein out_domain target_only 0.1 123
submit_job smote wasserstein out_domain target_only 0.1 42

# ========================================
# MISSING smote_plain: 31 jobs
# ========================================
echo ""
echo "--- smote_plain: 31 jobs ---"

# All ratio=0.5 (24 jobs: 3 dist × 2 dom × 2 mode × 2 seed)
for dist in dtw mmd wasserstein; do
    for dom in in_domain out_domain; do
        for mode in source_only target_only; do
            for seed in 42 123; do
                submit_job smote_plain "$dist" "$dom" "$mode" "0.5" "$seed"
            done
        done
    done
done

# Missing ratio=0.1 (7 jobs)
submit_job smote_plain dtw in_domain source_only 0.1 42
submit_job smote_plain dtw in_domain target_only 0.1 123
submit_job smote_plain dtw out_domain source_only 0.1 123
submit_job smote_plain mmd in_domain source_only 0.1 42
submit_job smote_plain mmd out_domain source_only 0.1 123
submit_job smote_plain mmd out_domain target_only 0.1 42
submit_job smote_plain wasserstein in_domain target_only 0.1 123

echo ""
echo "============================================================"

{
    echo ""
    echo "# Completed: $(date)"
    echo "# Submitted: $JOB_COUNT"
    echo "# Failed: $FAIL_COUNT"
} >> "$LOG_FILE"

echo "  Submitted: $JOB_COUNT"
echo "  Failed:    $FAIL_COUNT"
echo "  Log: $LOG_FILE"
echo "============================================================"

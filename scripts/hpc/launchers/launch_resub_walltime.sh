#!/bin/bash
# =============================================================================
# Launcher: Resubmit walltime-killed SvmA/SvmW + missing SvmW pooled
# =============================================================================
# Resubmits:
#   - SvmA: 17 split2 jobs (walltime 6h→24h)
#   - SvmW: 70 split2 jobs (walltime 6h→24h)
#   - SvmW: 2 pooled jobs (s42, s123)
#
# All jobs go to SEMINAR queue (no per-user max_queued limit).
#
# Usage:
#   ./scripts/hpc/launchers/launch_resub_walltime.sh --dry-run
#   ./scripts/hpc/launchers/launch_resub_walltime.sh
# =============================================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SPLIT2_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
POOLED_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research.sh"

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

QUEUE="SEMINAR"
N_TRIALS=100
RANKING="knn"

SUBMIT_COUNT=0
SKIP_COUNT=0

echo "============================================================"
echo "  Walltime Resubmission Launcher"
echo "  $(date)"
echo "  Mode: $( $DRY_RUN && echo 'DRY-RUN' || echo 'LIVE')"
echo "============================================================"
echo ""

submit_split2() {
    local model="$1" cond="$2" mode="$3" dist="$4" dom="$5" seed="$6" ratio="$7"
    local ncpus mem walltime

    case "$model" in
        SvmA) ncpus=8; mem="32gb"; walltime="24:00:00" ;;
        SvmW) ncpus=4; mem="8gb";  walltime="24:00:00" ;;
    esac

    local c2="${cond:0:2}"
    local d2="${dist:0:1}${dom:0:1}"
    local m1="${mode:0:1}"
    local JOB_NAME="r${model:0:2}_${c2}_${d2}_${m1}_r${ratio}_s${seed}"
    JOB_NAME="${JOB_NAME:0:15}"

    local env_vars="MODEL=$model,CONDITION=$cond,MODE=$mode,DISTANCE=$dist,DOMAIN=$dom,SEED=$seed,RATIO=$ratio,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"

    if $DRY_RUN; then
        echo "[DRY] $model | $cond | $mode | $dist | $dom | r=$ratio | s$seed"
        ((SUBMIT_COUNT++))
    else
        local JOB_ID
        JOB_ID=$(qsub -N "$JOB_NAME" \
            -l "select=1:ncpus=${ncpus}:mem=${mem}" \
            -l "walltime=$walltime" \
            -q "$QUEUE" \
            -v "$env_vars" \
            "$SPLIT2_SCRIPT" 2>&1) || { echo "[FAIL] $JOB_NAME: $JOB_ID"; ((SKIP_COUNT++)); return; }
        echo "[OK] $model | $cond | $mode | $dist | $dom | r=$ratio | s$seed → $JOB_ID"
        ((SUBMIT_COUNT++))
        sleep 0.2
    fi
}

submit_pooled() {
    local model="$1" seed="$2"
    local ncpus mem walltime

    case "$model" in
        SvmW) ncpus=4; mem="8gb";  walltime="24:00:00" ;;
        *)    ncpus=8; mem="32gb"; walltime="24:00:00" ;;
    esac

    local JOB_NAME="pool_${model}_s${seed}"

    if $DRY_RUN; then
        echo "[DRY] POOLED: $model s$seed"
        ((SUBMIT_COUNT++))
    else
        local JOB_ID
        JOB_ID=$(qsub -N "$JOB_NAME" \
            -l "select=1:ncpus=${ncpus}:mem=${mem}" \
            -l "walltime=$walltime" \
            -q "$QUEUE" \
            -v "MODEL=$model,SEED=$seed" \
            "$POOLED_SCRIPT" 2>&1) || { echo "[FAIL] $JOB_NAME: $JOB_ID"; ((SKIP_COUNT++)); return; }
        echo "[OK] POOLED: $model s$seed → $JOB_ID"
        ((SUBMIT_COUNT++))
        sleep 0.2
    fi
}

# ===== 1. SvmA split2: 17 walltime-killed jobs =====
echo "--- SvmA split2 resubmit (17 jobs, walltime 24h) ---"
submit_split2 SvmA undersample source_only mmd in_domain 42 0.5
submit_split2 SvmA baseline source_only mmd in_domain 123 0.5
submit_split2 SvmA smote_plain source_only mmd in_domain 123 0.1
submit_split2 SvmA undersample source_only mmd in_domain 123 0.1
submit_split2 SvmA smote_plain source_only mmd in_domain 123 0.5
submit_split2 SvmA undersample source_only mmd in_domain 123 0.5
submit_split2 SvmA smote source_only dtw out_domain 42 0.1
submit_split2 SvmA smote_plain source_only dtw out_domain 42 0.5
submit_split2 SvmA undersample source_only dtw out_domain 123 0.1
submit_split2 SvmA smote target_only dtw out_domain 42 0.1
submit_split2 SvmA smote target_only dtw out_domain 123 0.1
submit_split2 SvmA smote target_only dtw out_domain 123 0.5
submit_split2 SvmA undersample target_only dtw in_domain 42 0.1
submit_split2 SvmA smote_plain target_only dtw in_domain 123 0.1
submit_split2 SvmA baseline source_only wasserstein in_domain 42 0.5
submit_split2 SvmA smote_plain source_only wasserstein in_domain 42 0.5
submit_split2 SvmA smote target_only wasserstein in_domain 42 0.5
echo ""

# ===== 2. SvmW split2: 70 walltime-killed jobs =====
echo "--- SvmW split2 resubmit (70 jobs, walltime 24h) ---"
# Read from extracted params file
while read cond mode dist dom seed ratio; do
    submit_split2 SvmW "$cond" "$mode" "$dist" "$dom" "$seed" "$ratio"
done < /tmp/svmw_resub_unique.txt
echo ""

# ===== 3. SvmW pooled: 2 new jobs =====
echo "--- SvmW pooled (2 jobs, walltime 24h) ---"
submit_pooled SvmW 42
submit_pooled SvmW 123
echo ""

# ===== Summary =====
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Submitted: $SUBMIT_COUNT"
echo "  Failed:    $SKIP_COUNT"
echo "  Total expected: 89 (17 SvmA + 70 SvmW + 2 SvmW pooled)"
echo "============================================================"

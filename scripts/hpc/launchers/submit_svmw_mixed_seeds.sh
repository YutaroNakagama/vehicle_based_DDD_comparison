#!/bin/bash
# ============================================================
# Submit SvmW MODE=mixed jobs for the 13 seeds missing from the
# canonical 2-seed scope, expanding mixed-domain coverage to
# match the 15-seed exp3 scope.
#
# Per missing seed: 7 cond-variants × 3 dist × 2 dom = 42 jobs
# Total: 13 seeds × 42 = 546 jobs
#
# Usage:
#   bash scripts/hpc/launchers/submit_svmw_mixed_seeds.sh --dry-run
#   bash scripts/hpc/launchers/submit_svmw_mixed_seeds.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# CPU queues only — SvmW doesn't use GPU
CPU_QUEUES=("SINGLE" "SMALL" "LONG" "LONG-L" "LARGE" "DEF" "XLARGE" "X2LARGE" "VM-CPU" "VM-LM")
CPU_IDX=0

MISSING_SEEDS=(0 1 3 7 13 99 256 512 777 999 1234 1337 2024)
DISTANCES=(mmd dtw wasserstein)
DOMAINS=(in_domain out_domain)

# (CONDITION, RATIO) pairs — 1 baseline + 6 ratio variants = 7 per seed
declare -a COND_VARIANTS=(
    "baseline:"
    "smote:0.1"
    "smote:0.5"
    "smote_plain:0.1"
    "smote_plain:0.5"
    "undersample:0.1"
    "undersample:0.5"
)

# Short labels for job names
short_dist() { case "$1" in dtw) echo d;; mmd) echo m;; wasserstein) echo w;; esac; }
short_dom()  { case "$1" in in_domain) echo i;; out_domain) echo o;; esac; }
short_cond() {
    case "$1" in
        baseline)        echo ba;;
        smote)           echo iv;;
        smote_plain)     echo sm;;
        undersample)     echo un;;
    esac
}

# Build the dedup set of already-queued names
echo "[INFO] Building dedup set from current queue..."
ACTIVE_NAMES=$(qstat -u s2240011 2>/dev/null | awk 'NR>5 && $10 != "C" {print $1}' | \
    while read jid; do qstat -f "$jid" 2>/dev/null | awk '/Job_Name/ {print $3}'; done | sort -u)

in_queue() { echo "$ACTIVE_NAMES" | grep -qx "$1"; }

# Build the list of completed evals to skip
existing_completed() {
    local TAG="$1"
    find results/outputs/evaluation/SvmW -name "eval_results_SvmW_mixed_${TAG}.json" 2>/dev/null \
        | grep -v "_invalidated" | grep -q .
}

submit_one() {
    local COND="$1" RATIO="$2" DIST="$3" DOM="$4" SEED="$5"

    local DL=$(short_dist "$DIST"); local DM=$(short_dom "$DOM"); local CS=$(short_cond "$COND")
    local RTAG=""
    [[ -n "$RATIO" ]] && RTAG="_r${RATIO}"
    local JOB_NAME="Sw_${CS}_${DL}${DM}_mx${RTAG}_s${SEED}"

    # Build the full tag (matches the format used by pbs_prior_research_split2.sh)
    local TAG
    case "$COND" in
        baseline)    TAG="prior_SvmW_baseline_knn_${DIST}_${DOM}_mixed_split2_s${SEED}" ;;
        smote)       TAG="prior_SvmW_imbalv3_knn_${DIST}_${DOM}_mixed_split2_subjectwise_ratio${RATIO}_s${SEED}" ;;
        smote_plain) TAG="prior_SvmW_smote_plain_knn_${DIST}_${DOM}_mixed_split2_ratio${RATIO}_s${SEED}" ;;
        undersample) TAG="prior_SvmW_undersample_rus_knn_${DIST}_${DOM}_mixed_split2_ratio${RATIO}_s${SEED}" ;;
    esac

    if existing_completed "$TAG"; then
        echo "[SKIP-DONE] $JOB_NAME"
        return 1
    fi
    if in_queue "$JOB_NAME"; then
        echo "[SKIP-QUEUED] $JOB_NAME"
        return 1
    fi

    local QUEUE="${CPU_QUEUES[$((CPU_IDX % ${#CPU_QUEUES[@]}))]}"
    ((CPU_IDX++))

    local CMD="qsub -N $JOB_NAME -l select=1:ncpus=8:mem=32gb -l walltime=12:00:00 -q $QUEUE"
    CMD="$CMD -v MODEL=SvmW,CONDITION=$COND,MODE=mixed,DISTANCE=$DIST,DOMAIN=$DOM"
    CMD="$CMD,RATIO=${RATIO:-0.5},SEED=$SEED,N_TRIALS=100,RANKING=knn,RUN_EVAL=true"
    CMD="$CMD $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY] $JOB_NAME [$QUEUE]"
        return 0
    fi
    local OUT
    OUT=$(eval "$CMD" 2>&1)
    if [[ $? -eq 0 ]]; then
        echo "[SUB] $JOB_NAME [$QUEUE] -> $OUT"
        return 0
    else
        echo "[ERR] $JOB_NAME -> $(echo "$OUT" | grep -oE 'QOS[A-Za-z]*|error[^:]*' | head -1)"
        return 1
    fi
}

N_SUB=0; N_SKIP=0; N_ERR=0
for SEED in "${MISSING_SEEDS[@]}"; do
    for COND_RATIO in "${COND_VARIANTS[@]}"; do
        COND="${COND_RATIO%:*}"
        RATIO="${COND_RATIO#*:}"
        for DIST in "${DISTANCES[@]}"; do
            for DOM in "${DOMAINS[@]}"; do
                if submit_one "$COND" "$RATIO" "$DIST" "$DOM" "$SEED"; then
                    if $DRY_RUN; then ((N_SUB++)); else ((N_SUB++)); fi
                else
                    ((N_SKIP++))
                fi
                $DRY_RUN || sleep 0.2
            done
        done
    done
done

echo ""
echo "[DONE] Submitted=$N_SUB  Skipped=$N_SKIP"

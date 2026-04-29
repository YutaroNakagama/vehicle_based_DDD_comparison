#!/bin/bash
# ============================================================
# Hedge submitter: route a subset of Lstm domain_train jobs to
# CPU queues so they make progress while GPU queues are stuck.
#
# Strategy: pick missing Lstm tags NOT currently queued (the
# 362 that QOSMaxSubmitJobPerUserLimit rejected), submit up to
# LIMIT to CPU queues with a distinct job-name prefix (Lc_) so
# they don't conflict with the GPU-routed Ls_* jobs.
#
# Usage:
#   bash scripts/hpc/launchers/submit_lstm_cpu_hedge.sh --dry-run
#   bash scripts/hpc/launchers/submit_lstm_cpu_hedge.sh 30
#   bash scripts/hpc/launchers/submit_lstm_cpu_hedge.sh 100
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
MISSING_LIST="/tmp/exp3_all_missing.txt"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=true; LIMIT="${2:-30}"; } || LIMIT="${1:-30}"

CPU_QUEUES=("SINGLE" "SMALL" "LONG" "LONG-L" "LARGE" "DEF" "XLARGE" "X2LARGE" "VM-CPU" "VM-LM")
CPU_IDX=0

short_dist() { case "$1" in dtw) echo d;; mmd) echo m;; wasserstein) echo w;; esac; }
short_dom()  { case "$1" in in_domain) echo i;; out_domain) echo o;; esac; }
short_cond() {
    case "$1" in
        baseline) echo ba;; imbalv3) echo iv;;
        smote_plain) echo sm;; undersample_rus) echo un;;
    esac
}
cond_env() {
    case "$1" in
        baseline) echo baseline;; imbalv3) echo smote;;
        smote_plain) echo smote_plain;; undersample_rus) echo undersample;;
    esac
}

echo "[INFO] Building dedup set from current queue..."
ACTIVE_NAMES=$(qstat -u s2240011 2>/dev/null | awk 'NR>5 && $10 != "C" {print $1}' | \
    while read jid; do qstat -f "$jid" 2>/dev/null | awk '/Job_Name/ {print $3}'; done | sort -u)
in_queue() { echo "$ACTIVE_NAMES" | grep -qx "$1"; }

# Build matching GPU job-name set (we'll skip Lstm tags whose Ls_* name is already queued)
gpu_name_for_tag() {
    local MODEL="$1" COND="$2" DIST="$3" DOM="$4" RATIO="$5" SEED="$6"
    local DL=$(short_dist "$DIST"); local DM=$(short_dom "$DOM"); local CS=$(short_cond "$COND")
    echo "${MODEL:0:2}_${CS}_${DL}${DM}_dt_r${RATIO}_s${SEED}"
}

submit_one() {
    local COND="$1" DIST="$2" DOM="$3" RATIO="$4" SEED="$5"

    local DL=$(short_dist "$DIST"); local DM=$(short_dom "$DOM"); local CS=$(short_cond "$COND")
    local COND_VAL=$(cond_env "$COND")
    local CPU_NAME="Lc_${CS}_${DL}${DM}_dt_r${RATIO}_s${SEED}"      # CPU hedge name
    local GPU_NAME="Ls_${CS}_${DL}${DM}_dt_r${RATIO}_s${SEED}"      # existing GPU name

    # Skip if either CPU-hedge or GPU version is already queued
    if in_queue "$CPU_NAME" || in_queue "$GPU_NAME"; then
        return 1
    fi

    local QUEUE="${CPU_QUEUES[$((CPU_IDX % ${#CPU_QUEUES[@]}))]}"
    ((CPU_IDX++))

    # Lstm CPU walltime: pessimistic 18h (≈30-40min/fold × 10 folds × safety margin)
    local CMD="qsub -N $CPU_NAME -l select=1:ncpus=8:mem=32gb -l walltime=18:00:00 -q $QUEUE"
    CMD="$CMD -v MODEL=Lstm,CONDITION=$COND_VAL,DISTANCE=$DIST,DOMAIN=$DOM"
    CMD="$CMD,RATIO=$RATIO,SEED=$SEED,RANKING=knn,RUN_EVAL=true,CUDA_VISIBLE_DEVICES="
    CMD="$CMD $JOB_SCRIPT"

    if $DRY_RUN; then echo "[DRY] $CPU_NAME [$QUEUE]"; return 0; fi
    local OUT
    OUT=$(eval "$CMD" 2>&1)
    if [[ $? -eq 0 ]]; then
        echo "[SUB] $CPU_NAME [$QUEUE] -> $OUT"
        return 0
    else
        echo "[ERR] $CPU_NAME -> $(echo "$OUT" | grep -oE 'QOS[A-Za-z]*' | head -1)"
        return 1
    fi
}

N_SUB=0; N_SKIP=0
while IFS=' ' read -r MODEL COND DIST DOM RATIO SEED; do
    [[ -z "$MODEL" || "$MODEL" == "#"* || "$MODEL" != "Lstm" ]] && continue
    if submit_one "$COND" "$DIST" "$DOM" "$RATIO" "$SEED"; then
        ((N_SUB++))
    else
        ((N_SKIP++))
    fi
    if [[ "$N_SUB" -ge "$LIMIT" ]]; then break; fi
    $DRY_RUN || sleep 0.2
done < "$MISSING_LIST"

echo ""
echo "[DONE] Submitted=$N_SUB  Skipped=$N_SKIP  Limit=$LIMIT"

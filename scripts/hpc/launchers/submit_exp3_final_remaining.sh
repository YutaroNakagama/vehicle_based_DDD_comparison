#!/bin/bash
# ============================================================
# Submit ALL remaining exp3 (domain_train) missing jobs
#
#   - Lstm baseline    :  37 jobs
#   - Lstm imbalv3     : 115 jobs
#   - Lstm smote_plain : 142 jobs  (134 already queued → ~8 net new)
#   - Lstm undersample : 146 jobs  (135 already queued → ~11 net new)
#   - SvmA smote_plain :  13 jobs  (1 running → ~12 net new)
#   - SvmA undersample :  25 jobs  (1 queued  → ~24 net new)
#
# Usage:
#   bash scripts/hpc/launchers/submit_exp3_final_remaining.sh --dry-run
#   bash scripts/hpc/launchers/submit_exp3_final_remaining.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_CPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
JOB_GPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

GPU_QUEUES=("GPU-1" "GPU-1A" "GPU-S" "GPU-L" "GPU-LA" "VM-GPU-L")
CPU_QUEUES=("SINGLE" "SMALL" "LONG" "LONG-L" "LARGE" "DEF" "XLARGE" "X2LARGE" "VM-CPU" "VM-LM")
GPU_IDX=0
CPU_IDX=0

# ---- Build active-queue name set for dedup ----
echo "[INFO] Fetching current queue names for dedup..."
ACTIVE_NAMES=$(qstat -u s2240011 2>/dev/null | awk 'NR>5 && $10 != "C" {print $1}' | \
    while read jid; do
        qstat -f "$jid" 2>/dev/null | awk '/Job_Name/ {print $3}'
    done | sort -u)
ACTIVE_COUNT=$(echo "$ACTIVE_NAMES" | grep -c . || true)
echo "[INFO] Active job names loaded: $ACTIVE_COUNT"

in_queue() {
    local name="$1"
    echo "$ACTIVE_NAMES" | grep -qx "$name"
}

# ---- Helper functions ----
short_dist() { case "$1" in dtw) echo d;; mmd) echo m;; wasserstein) echo w;; esac; }
short_dom()  { case "$1" in in_domain) echo i;; out_domain) echo o;; esac; }

short_cond() {
    case "$1" in
        baseline)        echo ba;;
        imbalv3)         echo iv;;   # 'iv' to distinguish from smote_plain 'sm'
        smote_plain)     echo sm;;
        undersample_rus) echo un;;
    esac
}

cond_env() {
    case "$1" in
        baseline)        echo baseline;;
        imbalv3)         echo smote;;
        smote_plain)     echo smote_plain;;
        undersample_rus) echo undersample;;
    esac
}

submit_one() {
    local MODEL="$1" COND="$2" DIST="$3" DOM="$4" RATIO="$5" SEED="$6"
    local DL=$(short_dist "$DIST")
    local DM=$(short_dom  "$DOM")
    local CS=$(short_cond "$COND")
    local COND_VAL=$(cond_env "$COND")
    local JOB_NAME="${MODEL:0:2}_${CS}_${DL}${DM}_dt_r${RATIO}_s${SEED}"

    if in_queue "$JOB_NAME"; then
        echo "[SKIP] $JOB_NAME (already in queue)"
        return
    fi

    local RES WALLTIME QUEUE JOB_SCRIPT
    if [[ "$MODEL" == "Lstm" ]]; then
        JOB_SCRIPT="$JOB_GPU"
        RES="ncpus=4:ngpus=1:mem=32gb"
        WALLTIME="24:00:00"
        QUEUE="${GPU_QUEUES[$((GPU_IDX % ${#GPU_QUEUES[@]}))]}"
        ((GPU_IDX++))
    else
        JOB_SCRIPT="$JOB_CPU"
        RES="ncpus=8:mem=32gb"
        WALLTIME="48:00:00"
        QUEUE="${CPU_QUEUES[$((CPU_IDX % ${#CPU_QUEUES[@]}))]}"
        ((CPU_IDX++))
    fi

    local CMD="qsub -N $JOB_NAME -l select=1:$RES -l walltime=$WALLTIME -q $QUEUE"
    CMD="$CMD -v MODEL=$MODEL,CONDITION=$COND_VAL,DISTANCE=$DIST,DOMAIN=$DOM"
    CMD="$CMD,RATIO=$RATIO,SEED=$SEED,N_TRIALS=100,RANKING=knn,RUN_EVAL=true"
    CMD="$CMD $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY] $JOB_NAME [$QUEUE wall=$WALLTIME] MODEL=$MODEL COND=$COND_VAL DIST=$DIST DOM=$DOM R=$RATIO S=$SEED"
    else
        local OUT
        OUT=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[SUB] $JOB_NAME [$QUEUE] -> $OUT"
        else
            echo "[ERR] $JOB_NAME -> $OUT"
        fi
        sleep 0.3
    fi
}

# ============================================================
# Missing job list (MODEL COND DIST DOM RATIO SEED)
# Generated from: comm -23 expected done
# ============================================================

echo "[INFO] DRY_RUN=$DRY_RUN"
echo "[INFO] Starting submission..."
echo ""

N_SUBMITTED=0
N_SKIPPED=0

while IFS=' ' read -r MODEL COND DIST DOM RATIO SEED; do
    [[ -z "$MODEL" || "$MODEL" == "#"* ]] && continue
    before_gpu=$GPU_IDX
    before_cpu=$CPU_IDX
    submit_one "$MODEL" "$COND" "$DIST" "$DOM" "$RATIO" "$SEED"
    if [[ $GPU_IDX -gt $before_gpu || $CPU_IDX -gt $before_cpu ]]; then
        ((N_SUBMITTED++))
    else
        ((N_SKIPPED++))
    fi
done < /tmp/exp3_all_missing.txt

echo ""
echo "[DONE] Submitted=$N_SUBMITTED  Skipped=$N_SKIPPED  Total=$(( N_SUBMITTED + N_SKIPPED ))"

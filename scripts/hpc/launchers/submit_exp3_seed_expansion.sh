#!/bin/bash
# ============================================================
# One-shot submitter for exp3 (domain_train) seed-expansion
#   Goal: bring SvmA and Lstm to FULL 15-seed coverage
#   matching SvmW [0,1,3,7,13,42,99,123,256,512,777,999,1234,1337,2024]
#
# Reads missing tuples from /tmp/exp3_missing_jobs.txt (one per line:
#   MODEL\tCOND_ANALYSIS\tDISTANCE\tDOMAIN\tRATIO\tSEED)
# and skips combinations matching Job_Name patterns of currently
# queued/running PBS jobs (15101-15124).
#
# Usage:
#   bash scripts/hpc/launchers/submit_exp3_seed_expansion.sh --dry-run
#   bash scripts/hpc/launchers/submit_exp3_seed_expansion.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_CPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
JOB_GPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh"
MISSING_FILE="${MISSING_FILE:-/tmp/exp3_missing_jobs.txt}"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Map analysis condition -> CONDITION env value expected by pbs script
declare -A COND_ENV=(
    [baseline]=baseline
    [imbalv3]=smote
    [smote_plain]=smote_plain
    [undersample_rus]=undersample
)

# Build set of active job-name signatures for filtering
# Format: SHORT_MODEL_SHORT_COND_DISTLETTERDOMLETTER_dt_r{R}_s{S}
ACTIVE_NAMES=$(qstat -u s2240011 2>/dev/null | awk 'NR>5 {print $4}' | sort -u)

# Build comprehensive active set by querying qstat -f for full names
ACTIVE_SET=""
for j in $(qstat -u s2240011 2>/dev/null | awk 'NR>5 {print $1}'); do
    nm=$(qstat -f "$j" 2>/dev/null | awk '/Job_Name/ {print $3; exit}')
    [[ -n "$nm" ]] && ACTIVE_SET="$ACTIVE_SET $nm"
done
echo "[INFO] Active jobs for filter: $(echo $ACTIVE_SET | wc -w)"

GPU_QUEUES=("GPU-1" "GPU-1A" "GPU-S" "GPU-L" "GPU-LA")
CPU_QUEUES=("SINGLE" "SMALL" "LONG" "LARGE")
GPU_IDX=0
CPU_IDX=0

short_dist() { case "$1" in dtw) echo d;; mmd) echo m;; wasserstein) echo w;; esac; }
short_dom()  { case "$1" in in_domain) echo i;; out_domain) echo o;; esac; }
short_cond() {
    # produce 2-char code matching original naming
    case "$1" in
        baseline)        echo ba;;
        imbalv3)         echo sm;;   # smote env  -> 'sm'
        smote_plain)     echo sm;;
        undersample_rus) echo un;;
    esac
}

submit_one() {
    local MODEL="$1" COND_A="$2" DIST="$3" DOM="$4" RATIO="$5" SEED="$6"
    local COND_ENV_VAL="${COND_ENV[$COND_A]}"
    local DL=$(short_dist "$DIST") DM=$(short_dom "$DOM") CS=$(short_cond "$COND_A")
    local R="$RATIO"
    local JOB_NAME="${MODEL:0:2}_${CS}_${DL}${DM}_dt_r${R}_s${SEED}"

    # Skip if a job with same signature is already queued
    for active in $ACTIVE_SET; do
        # Strip "_re" or other suffixes
        local stripped="${active%_re}"
        if [[ "$stripped" == "$JOB_NAME" ]]; then
            echo "[SKIP] $JOB_NAME (already queued as $active)"
            return
        fi
    done

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
    CMD="$CMD -v MODEL=$MODEL,CONDITION=$COND_ENV_VAL,DISTANCE=$DIST,DOMAIN=$DOM,RATIO=$RATIO,SEED=$SEED,N_TRIALS=100,RANKING=knn,RUN_EVAL=true"
    CMD="$CMD $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY] $JOB_NAME [$QUEUE walltime=$WALLTIME] cond_env=$COND_ENV_VAL"
    else
        local OUT
        OUT=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[SUB] $JOB_NAME [$QUEUE] -> $OUT"
        else
            echo "[ERR] $JOB_NAME -> $OUT"
        fi
        sleep 0.4
    fi
}

[[ ! -f "$MISSING_FILE" ]] && { echo "[FATAL] $MISSING_FILE not found"; exit 1; }

TOTAL=$(wc -l < "$MISSING_FILE")
echo "[INFO] Reading $TOTAL missing tuples from $MISSING_FILE"
echo "[INFO] DRY_RUN=$DRY_RUN"

N=0
while IFS=$'\t' read -r MODEL COND DIST DOM RATIO SEED; do
    [[ -z "$MODEL" ]] && continue
    submit_one "$MODEL" "$COND" "$DIST" "$DOM" "$RATIO" "$SEED"
    ((N++))
done < "$MISSING_FILE"

echo
echo "[INFO] Processed $N tuples (CPU submitted=$CPU_IDX, GPU submitted=$GPU_IDX)"
echo "Done."

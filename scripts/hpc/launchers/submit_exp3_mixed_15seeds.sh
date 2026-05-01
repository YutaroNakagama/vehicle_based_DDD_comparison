#!/bin/bash
# ============================================================
# One-shot submitter for exp3 mixed-domain jobs (15-seed scope)
#
# Reads missing tuples from /tmp/exp3_mixed_missing.txt:
#   MODEL\tCONDITION_ENV\tDISTANCE\tDOMAIN\tRATIO\tSEED
# (RATIO is empty for baseline.)
#
# Skips combinations whose Job_Name is already queued/running.
# Routes Lstm to the split2 GPU script + GPU queues, SvmW/SvmA to
# the split2 CPU script + CPU queues, in round-robin order.
#
# Resources mirror launch_prior_research_mixed.sh / submit_missing_mixed_exp3.sh:
#   SvmW: 8c/24g, 16h baseline / 24h smote / smote_plain
#   SvmA: 8c/48g, 30h baseline / 48h smote / smote_plain
#   Lstm: 4c/16g+1gpu, 20h baseline / 24h smote / smote_plain
#
# Usage:
#   bash scripts/hpc/launchers/submit_exp3_mixed_15seeds.sh --dry-run
#   bash scripts/hpc/launchers/submit_exp3_mixed_15seeds.sh
#   MISSING_FILE=/tmp/foo.txt bash scripts/hpc/launchers/submit_exp3_mixed_15seeds.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_CPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
JOB_GPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2_gpu.sh"
MISSING_FILE="${MISSING_FILE:-/tmp/exp3_mixed_missing.txt}"
MODE="mixed"
N_TRIALS=100
RANKING="knn"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Build active job-name set for filtering (avoid resubmitting queued/running)
ACTIVE_SET=""
for j in $(qstat -u s2240011 2>/dev/null | awk 'NR>5 {print $1}'); do
    nm=$(qstat -f "$j" 2>/dev/null | awk '/Job_Name/ {print $3; exit}')
    [[ -n "$nm" ]] && ACTIVE_SET="$ACTIVE_SET $nm"
done
echo "[INFO] Active jobs for filter: $(echo $ACTIVE_SET | wc -w)"

GPU_QUEUES=("GPU-1" "GPU-1A" "GPU-S" "GPU-L" "GPU-LA")
CPU_QUEUES=("SINGLE" "DEF" "SMALL" "LONG" "LARGE")
GPU_IDX=0; CPU_IDX=0

short_dist() { case "$1" in dtw) echo d;; mmd) echo m;; wasserstein) echo w;; esac; }
short_dom()  { case "$1" in in_domain) echo i;; out_domain) echo o;; esac; }
short_cond() {
    case "$1" in
        baseline)   echo bs;;
        smote)      echo sm;;
        smote_plain) echo sp;;
        undersample) echo un;;
    esac
}

get_resources() {
    local model="$1" cond="$2"
    local is_smote=false
    [[ "$cond" == "smote" || "$cond" == "smote_plain" ]] && is_smote=true
    case "$model" in
        SvmW) $is_smote && echo "ncpus=8:mem=24gb 24:00:00" || echo "ncpus=8:mem=24gb 16:00:00" ;;
        SvmA) $is_smote && echo "ncpus=8:mem=48gb 48:00:00" || echo "ncpus=8:mem=48gb 30:00:00" ;;
        Lstm) $is_smote && echo "ncpus=4:ngpus=1:mem=16gb 24:00:00" || echo "ncpus=4:ngpus=1:mem=16gb 20:00:00" ;;
    esac
}

submit_one() {
    local MODEL="$1" COND="$2" DIST="$3" DOM="$4" RATIO="$5" SEED="$6"
    [[ "$RATIO" == "-" ]] && RATIO=""
    local DL=$(short_dist "$DIST") DM=$(short_dom "$DOM") CS=$(short_cond "$COND")
    local JOB_NAME
    if [[ -z "$RATIO" ]]; then
        JOB_NAME="${MODEL:0:2}_${CS}_${DL}${DM}_m_s${SEED}"
    else
        JOB_NAME="${MODEL:0:2}_${CS}_${DL}${DM}_m_r${RATIO}_s${SEED}"
    fi

    for active in $ACTIVE_SET; do
        local stripped="${active%_re}"
        if [[ "$stripped" == "$JOB_NAME" ]]; then
            echo "[SKIP] $JOB_NAME (already queued as $active)"
            return
        fi
    done

    local RES_LINE NCPUS_MEM WALLTIME QUEUE JOB_SCRIPT
    RES_LINE=$(get_resources "$MODEL" "$COND")
    NCPUS_MEM=$(echo "$RES_LINE" | cut -d' ' -f1)
    WALLTIME=$(echo "$RES_LINE" | cut -d' ' -f2)

    if [[ "$MODEL" == "Lstm" ]]; then
        JOB_SCRIPT="$JOB_GPU"
        QUEUE="${GPU_QUEUES[$((GPU_IDX % ${#GPU_QUEUES[@]}))]}"
        ((GPU_IDX++))
    else
        JOB_SCRIPT="$JOB_CPU"
        QUEUE="${CPU_QUEUES[$((CPU_IDX % ${#CPU_QUEUES[@]}))]}"
        ((CPU_IDX++))
    fi

    local VARS="MODEL=$MODEL,CONDITION=$COND,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    [[ -n "$RATIO" ]] && VARS="$VARS,RATIO=$RATIO"

    local CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v $VARS $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY] $JOB_NAME [$QUEUE walltime=$WALLTIME] cond=$COND ratio=${RATIO:-—}"
    else
        local OUT
        OUT=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[SUB] $JOB_NAME [$QUEUE] -> $OUT"
        else
            echo "[ERR] $JOB_NAME [$QUEUE] -> $OUT"
        fi
        sleep 0.3
    fi
}

[[ ! -f "$MISSING_FILE" ]] && { echo "[FATAL] $MISSING_FILE not found"; exit 1; }
TOTAL=$(wc -l < "$MISSING_FILE")
echo "[INFO] Reading $TOTAL missing mixed tuples from $MISSING_FILE"
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

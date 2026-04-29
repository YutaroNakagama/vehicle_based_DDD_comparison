#!/bin/bash
# ============================================================
# One-shot submitter for exp3 (domain_train) missing jobs
#   - SvmA: 1 job  (smote_plain wasserstein out_domain ratio=0.5 s42)
#   - Lstm: 21 jobs (smote_plain / undersample_rus at ratio=0.1)
#
# Usage:
#   bash scripts/hpc/launchers/submit_exp3_missing_22.sh --dry-run
#   bash scripts/hpc/launchers/submit_exp3_missing_22.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_CPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
JOB_GPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# CPU queue rotation (SvmA only - 1 job, just use SINGLE)
# GPU queue rotation (Lstm - cycle through GPU queues)
GPU_QUEUES=("GPU-1" "GPU-1A" "GPU-S" "GPU-L" "GPU-LA")
GPU_IDX=0

submit_one() {
    local MODEL="$1" COND="$2" DIST="$3" DOM="$4" RATIO="$5" SEED="$6"
    local JOB_NAME RES WALLTIME QUEUE JOB_SCRIPT

    if [[ "$MODEL" == "Lstm" ]]; then
        JOB_SCRIPT="$JOB_GPU"
        RES="ncpus=4:ngpus=1:mem=32gb"
        WALLTIME="24:00:00"   # smote / undersample need extra time
        QUEUE="${GPU_QUEUES[$((GPU_IDX % ${#GPU_QUEUES[@]}))]}"
        ((GPU_IDX++))
    else
        JOB_SCRIPT="$JOB_CPU"
        RES="ncpus=8:mem=32gb"
        WALLTIME="48:00:00"
        QUEUE="SINGLE"
    fi

    local COND_SHORT="${COND:0:2}"
    JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DIST:0:1}${DOM:0:1}_dt_r${RATIO}_s${SEED}"

    local CMD="qsub -N $JOB_NAME -l select=1:$RES -l walltime=$WALLTIME -q $QUEUE"
    CMD="$CMD -v MODEL=$MODEL,CONDITION=$COND,DISTANCE=$DIST,DOMAIN=$DOM,RATIO=$RATIO,SEED=$SEED,N_TRIALS=100,RANKING=knn,RUN_EVAL=true"
    CMD="$CMD $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY] $JOB_NAME [$QUEUE walltime=$WALLTIME]"
    else
        local JID
        JID=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[SUB] $JOB_NAME [$QUEUE] -> $JID"
        else
            echo "[ERR] $JOB_NAME -> $JID"
        fi
        sleep 0.3
    fi
}

# ----- SvmA: 1 missing -----
submit_one SvmA smote_plain wasserstein out_domain 0.5 42

# ----- Lstm: 21 missing (all ratio=0.1) -----
# smote_plain (10 entries)
submit_one Lstm smote_plain dtw         in_domain  0.1 42
submit_one Lstm smote_plain dtw         in_domain  0.1 123
submit_one Lstm smote_plain dtw         out_domain 0.1 42
submit_one Lstm smote_plain dtw         out_domain 0.1 123
submit_one Lstm smote_plain mmd         in_domain  0.1 42
submit_one Lstm smote_plain mmd         out_domain 0.1 123
submit_one Lstm smote_plain wasserstein in_domain  0.1 42
submit_one Lstm smote_plain wasserstein in_domain  0.1 123
submit_one Lstm smote_plain wasserstein out_domain 0.1 42
submit_one Lstm smote_plain wasserstein out_domain 0.1 123

# undersample_rus -> CONDITION=undersample (11 entries)
submit_one Lstm undersample dtw         in_domain  0.1 42
submit_one Lstm undersample dtw         in_domain  0.1 123
submit_one Lstm undersample dtw         out_domain 0.1 42
submit_one Lstm undersample dtw         out_domain 0.1 123
submit_one Lstm undersample mmd         in_domain  0.1 42
submit_one Lstm undersample mmd         in_domain  0.1 123
submit_one Lstm undersample mmd         out_domain 0.1 123
submit_one Lstm undersample wasserstein in_domain  0.1 42
submit_one Lstm undersample wasserstein in_domain  0.1 123
submit_one Lstm undersample wasserstein out_domain 0.1 42
submit_one Lstm undersample wasserstein out_domain 0.1 123

echo "Done."

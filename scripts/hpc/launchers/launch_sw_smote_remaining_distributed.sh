#!/bin/bash
# ============================================================
# Subject-wise SMOTE (FIXED) - Remaining Jobs (92-162)
# Distribute across available queues: DEFAULT, TINY, SMALL, SINGLE
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_LOG="${PROJECT_ROOT}/logs/sw_smote_fixed_remaining_${TIMESTAMP}.txt"

echo "============================================================"
echo "Subject-wise SMOTE (FIXED) - Remaining Jobs"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Time: $(date)"
echo ""

SEED=42

# Queue rotation - prioritize less loaded queues
# DEFAULT has 0, TINY has 0, SMALL has some, SINGLE has most
QUEUES=("DEFAULT" "TINY" "DEFAULT" "SMALL" "DEFAULT" "TINY" "SINGLE")
QUEUE_IDX=0

# Initialize job log
echo "# SW-SMOTE Fixed Remaining Jobs: $TIMESTAMP" > "$JOB_LOG"
echo "" >> "$JOB_LOG"

JOB_COUNT=91  # Starting from 92

submit_job() {
    local RATIO=$1
    local RANKING=$2
    local DISTANCE=$3
    local DOMAIN=$4
    local MODE=$5
    
    JOB_COUNT=$((JOB_COUNT + 1))
    
    RATIO_STR=$(echo $RATIO | sed 's/\./_/g')
    TAG="swsmote_v2_r${RATIO_STR}_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_s${SEED}"
    SUBJECT_FILE="results/domain_analysis/distance/subject-wise/ranks/ranks29/${RANKING}/${DISTANCE}_${DOMAIN}.txt"
    
    QUEUE="${QUEUES[$QUEUE_IDX]}"
    QUEUE_IDX=$(( (QUEUE_IDX + 1) % ${#QUEUES[@]} ))
    
    # Adjust resources based on queue
    if [[ "$QUEUE" == "TINY" ]]; then
        MEM="4gb"
        WALLTIME="2:00:00"
        NCPUS=1
    elif [[ "$MODE" == "pooled" ]]; then
        MEM="8gb"
        WALLTIME="12:00:00"
        NCPUS=4
    else
        MEM="6gb"
        WALLTIME="10:00:00"
        NCPUS=4
    fi
    
    JOB=$(qsub -N "swf_${RATIO_STR}_${RANKING:0:2}_${DISTANCE:0:1}_${DOMAIN:0:3}_${MODE:0:3}" \
        -q "$QUEUE" \
        -l "select=1:ncpus=${NCPUS}:mem=${MEM}" \
        -l "walltime=${WALLTIME}" \
        -j oe \
        -o "${PROJECT_ROOT}/logs/hpc/" \
        -v "PROJECT_ROOT=${PROJECT_ROOT},TAG=${TAG},RATIO=${RATIO},MODE=${MODE},SUBJECT_FILE=${SUBJECT_FILE},SEED=${SEED}" \
        pbs_train_sw_smote_ranking.sh 2>&1)
    
    if [[ "$JOB" == *"spcc-adm1"* ]]; then
        echo "[$JOB_COUNT/162] $TAG -> $JOB (Queue: $QUEUE)"
        echo "$JOB $TAG $QUEUE" >> "$JOB_LOG"
    else
        echo "[$JOB_COUNT/162] FAILED: $TAG (Queue: $QUEUE) - $JOB"
        echo "FAILED $TAG $QUEUE $JOB" >> "$JOB_LOG"
        # Try next queue
        QUEUE_IDX=$(( (QUEUE_IDX + 1) % ${#QUEUES[@]} ))
    fi
}

echo "Submitting remaining jobs (92-162)..."
echo ""

# Continue from job 92: r0_5_lof_mmd_in_domain_source_only
submit_job 0.5 lof mmd in_domain source_only
submit_job 0.5 lof mmd in_domain target_only
submit_job 0.5 lof mmd mid_domain pooled
submit_job 0.5 lof mmd mid_domain source_only
submit_job 0.5 lof mmd mid_domain target_only
submit_job 0.5 lof mmd out_domain pooled
submit_job 0.5 lof mmd out_domain source_only
submit_job 0.5 lof mmd out_domain target_only
submit_job 0.5 lof wasserstein in_domain pooled
submit_job 0.5 lof wasserstein in_domain source_only
submit_job 0.5 lof wasserstein in_domain target_only
submit_job 0.5 lof wasserstein mid_domain pooled
submit_job 0.5 lof wasserstein mid_domain source_only
submit_job 0.5 lof wasserstein mid_domain target_only
submit_job 0.5 lof wasserstein out_domain pooled
submit_job 0.5 lof wasserstein out_domain source_only
submit_job 0.5 lof wasserstein out_domain target_only

# ratio=1.0, all combinations
for RANKING in knn lof; do
    for DISTANCE in dtw mmd wasserstein; do
        for DOMAIN in in_domain mid_domain out_domain; do
            for MODE in pooled source_only target_only; do
                submit_job 1.0 $RANKING $DISTANCE $DOMAIN $MODE
            done
        done
    done
done

echo ""
echo "============================================================"
echo "Remaining jobs submitted!"
echo "Total submitted this round: $((JOB_COUNT - 91))"
echo "Job log: $JOB_LOG"
echo ""
echo "Monitor: qstat -u \$USER"
echo "============================================================"

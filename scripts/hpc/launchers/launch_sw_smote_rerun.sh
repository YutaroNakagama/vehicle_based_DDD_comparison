#!/bin/bash
# ============================================================
# SW-SMOTE Rerun: Failed + Missing Jobs (19 total)
# ============================================================
# 17 jobs failed due to walltime exceeded (TINY queue: 30min)
# 2 jobs were never submitted
# 
# Fix: Use SINGLE queue with 6 hours walltime
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_LOG="${PROJECT_ROOT}/logs/sw_smote_rerun_${TIMESTAMP}.txt"

echo "============================================================"
echo "SW-SMOTE Rerun: Failed + Missing Jobs"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Time: $(date)"
echo ""

# Fixed parameters
SEED=42
QUEUE="SINGLE"
WALLTIME="06:00:00"
MEM="8gb"
NCPUS=4

# Jobs to rerun (17 failed + 2 missing = 19 total)
# Format: RATIO RANKING DISTANCE DOMAIN MODE
JOBS_TO_RUN=(
    # Failed jobs (walltime exceeded)
    "0.5 lof mmd out_domain pooled"
    "0.5 lof wasserstein in_domain pooled"
    "1.0 knn dtw in_domain target_only"
    "1.0 knn dtw mid_domain target_only"
    "1.0 knn mmd in_domain pooled"
    "1.0 knn mmd mid_domain pooled"
    "1.0 knn mmd out_domain source_only"
    "1.0 knn wasserstein in_domain source_only"
    "1.0 knn wasserstein mid_domain target_only"
    "1.0 knn wasserstein out_domain target_only"
    "1.0 lof dtw mid_domain pooled"
    "1.0 lof dtw out_domain pooled"
    "1.0 lof mmd in_domain source_only"
    "1.0 lof mmd mid_domain source_only"
    "1.0 lof mmd out_domain target_only"
    "1.0 lof wasserstein in_domain target_only"
    "1.0 lof wasserstein out_domain pooled"
    # Missing jobs (never submitted)
    "1.0 lof wasserstein out_domain source_only"
    "1.0 lof wasserstein out_domain target_only"
)

TOTAL_JOBS=${#JOBS_TO_RUN[@]}
echo "Total jobs to submit: $TOTAL_JOBS"
echo "Queue: $QUEUE"
echo "Walltime: $WALLTIME"
echo "Memory: $MEM"
echo "============================================================"
echo ""

# Initialize job log
echo "# SW-SMOTE Rerun: $TIMESTAMP" > "$JOB_LOG"
echo "# Total jobs: $TOTAL_JOBS" >> "$JOB_LOG"
echo "# Queue: $QUEUE, Walltime: $WALLTIME" >> "$JOB_LOG"
echo "" >> "$JOB_LOG"

JOB_COUNT=0

for job_params in "${JOBS_TO_RUN[@]}"; do
    read -r RATIO RANKING DISTANCE DOMAIN MODE <<< "$job_params"
    
    JOB_COUNT=$((JOB_COUNT + 1))
    
    # Generate tag
    RATIO_STR=$(echo $RATIO | sed 's/\./_/g')
    TAG="swsmote_v2_r${RATIO_STR}_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_s${SEED}"
    
    # Get subject list file
    SUBJECT_FILE="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING}/${DISTANCE}_${DOMAIN}.txt"
    
    # Submit job
    JOB=$(qsub -N "swr_${RATIO_STR}_${RANKING:0:2}_${DISTANCE:0:1}_${DOMAIN:0:3}_${MODE:0:3}" \
        -q "$QUEUE" \
        -l "select=1:ncpus=${NCPUS}:mem=${MEM}" \
        -l "walltime=${WALLTIME}" \
        -j oe \
        -o "${PROJECT_ROOT}/logs/hpc/" \
        -v "PROJECT_ROOT=${PROJECT_ROOT},TAG=${TAG},RATIO=${RATIO},MODE=${MODE},SUBJECT_FILE=${SUBJECT_FILE},SEED=${SEED}" \
        pbs_train_sw_smote_ranking.sh)
    
    echo "[$JOB_COUNT/$TOTAL_JOBS] $TAG -> $JOB"
    echo "$JOB $TAG $QUEUE" >> "$JOB_LOG"
done

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "Total: $JOB_COUNT jobs"
echo "Job log: $JOB_LOG"
echo ""
echo "Monitor: qstat -u \$USER"
echo "============================================================"

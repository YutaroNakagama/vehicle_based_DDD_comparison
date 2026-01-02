#!/bin/bash
# ============================================================
# Subject-wise SMOTE (FIXED) + Ranking Methods Full Test
# ============================================================
# Fix: subject_id now preserved in split_data() for proper 
#      subject-wise oversampling
# 
# Target ratio: 0.1, 0.5, 1.0
# Ranking: knn, lof
# Distance: dtw, mmd, wasserstein
# Domain: in_domain, mid_domain, out_domain
# Mode: pooled, source_only, target_only
# Queue distribution: SINGLE, SMALL, DEFAULT for load balancing
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_LOG="${PROJECT_ROOT}/logs/sw_smote_fixed_jobs_${TIMESTAMP}.txt"

echo "============================================================"
echo "Subject-wise SMOTE (FIXED) + Ranking Full Test"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Time: $(date)"
echo ""

# Parameters
RATIOS="0.1 0.5 1.0"
RANKINGS="knn lof"
DISTANCES="dtw mmd wasserstein"
DOMAINS="in_domain mid_domain out_domain"
MODES="pooled source_only target_only"
SEED=42

# Queue rotation for load balancing
QUEUES=("SINGLE" "SMALL" "DEFAULT")
QUEUE_IDX=0

# Count total jobs
TOTAL_JOBS=0
for RATIO in $RATIOS; do
    for RANKING in $RANKINGS; do
        for DISTANCE in $DISTANCES; do
            for DOMAIN in $DOMAINS; do
                for MODE in $MODES; do
                    TOTAL_JOBS=$((TOTAL_JOBS + 1))
                done
            done
        done
    done
done

echo "Total jobs to submit: $TOTAL_JOBS"
echo "Ratios: $RATIOS"
echo "Rankings: $RANKINGS"
echo "Distances: $DISTANCES"
echo "Domains: $DOMAINS"
echo "Modes: $MODES"
echo "============================================================"
echo ""

# Initialize job log
echo "# Subject-wise SMOTE (FIXED) Full Test: $TIMESTAMP" > "$JOB_LOG"
echo "# Total jobs: $TOTAL_JOBS" >> "$JOB_LOG"
echo "# Fix: subject_id preserved in split_data for proper SW oversampling" >> "$JOB_LOG"
echo "" >> "$JOB_LOG"

ALL_JOBS=""
JOB_COUNT=0

for RATIO in $RATIOS; do
    RATIO_STR=$(echo $RATIO | sed 's/\./_/g')  # 0.1 -> 0_1
    
    for RANKING in $RANKINGS; do
        for DISTANCE in $DISTANCES; do
            for DOMAIN in $DOMAINS; do
                for MODE in $MODES; do
                    JOB_COUNT=$((JOB_COUNT + 1))
                    
                    # Generate tag (with v2 suffix to distinguish from buggy runs)
                    TAG="swsmote_v2_r${RATIO_STR}_${RANKING}_${DISTANCE}_${DOMAIN}_${MODE}_s${SEED}"
                    
                    # Get subject list file
                    SUBJECT_FILE="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING}/${DISTANCE}_${DOMAIN}.txt"
                    
                    # Select queue (round-robin)
                    QUEUE="${QUEUES[$QUEUE_IDX]}"
                    QUEUE_IDX=$(( (QUEUE_IDX + 1) % ${#QUEUES[@]} ))
                    
                    # Set resources based on mode
                    if [[ "$MODE" == "pooled" ]]; then
                        MEM="8gb"
                        WALLTIME="12:00:00"
                        NCPUS=4
                    else
                        MEM="6gb"
                        WALLTIME="10:00:00"
                        NCPUS=4
                    fi
                    
                    # Mode flag for train.py
                    MODE_FLAG="--mode $MODE"
                    if [[ "$MODE" != "pooled" ]]; then
                        MODE_FLAG="$MODE_FLAG --target_file $SUBJECT_FILE"
                    fi
                    
                    # Submit job
                    JOB=$(qsub -N "swf_${RATIO_STR}_${RANKING:0:2}_${DISTANCE:0:1}_${DOMAIN:0:3}_${MODE:0:3}" \
                        -q "$QUEUE" \
                        -l "select=1:ncpus=${NCPUS}:mem=${MEM}" \
                        -l "walltime=${WALLTIME}" \
                        -j oe \
                        -o "${PROJECT_ROOT}/logs/hpc/" \
                        -v "PROJECT_ROOT=${PROJECT_ROOT},TAG=${TAG},RATIO=${RATIO},MODE=${MODE},SUBJECT_FILE=${SUBJECT_FILE},SEED=${SEED}" \
                        pbs_train_sw_smote_ranking.sh)
                    
                    echo "[$JOB_COUNT/$TOTAL_JOBS] $TAG -> $JOB (Queue: $QUEUE)"
                    echo "$JOB $TAG $QUEUE" >> "$JOB_LOG"
                    ALL_JOBS="$ALL_JOBS $JOB"
                done
            done
        done
    done
done

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "Total: $JOB_COUNT jobs"
echo "Job log: $JOB_LOG"
echo ""
echo "Monitor: qstat -u \$USER"
echo "============================================================"

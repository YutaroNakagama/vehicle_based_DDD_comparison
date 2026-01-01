#!/bin/bash
# ============================================================
# Subject-wise SMOTE: Multi-Ratio Comparison
# ============================================================
# Target ratios: 0.1, 0.5, 1.0
# Pooled mode with multiple seeds for statistical validation
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_LOG="${PROJECT_ROOT}/scripts/hpc/logs/imbalance/job_ids_sw_smote_ratio_${TIMESTAMP}.txt"

# Parameters
RATIOS="0.1 0.5 1.0"
SEEDS="42 123 456"
QUEUES=("SINGLE" "SMALL" "DEFAULT")
QUEUE_IDX=0

echo "============================================================"
echo "Subject-wise SMOTE Multi-Ratio Comparison"
echo "============================================================"
echo "Ratios: $RATIOS"
echo "Seeds: $SEEDS"
echo "Time: $(date)"
echo "============================================================"
echo ""

echo "# Subject-wise SMOTE Multi-Ratio: $TIMESTAMP" > "$JOB_LOG"

ALL_JOBS=""
JOB_COUNT=0

for RATIO in $RATIOS; do
    RATIO_STR=$(echo $RATIO | sed 's/\./_/g')
    
    for SEED in $SEEDS; do
        JOB_COUNT=$((JOB_COUNT + 1))
        
        TAG="imbal_v2_smote_sw_r${RATIO_STR}_seed${SEED}"
        
        # Queue rotation
        QUEUE="${QUEUES[$QUEUE_IDX]}"
        QUEUE_IDX=$(( (QUEUE_IDX + 1) % ${#QUEUES[@]} ))
        
        JOB=$(qsub -N "sw_r${RATIO_STR}_s${SEED}" \
            -q "$QUEUE" \
            -l "select=1:ncpus=4:mem=6gb" \
            -l "walltime=12:00:00" \
            -j oe \
            -o "${PROJECT_ROOT}/scripts/hpc/logs/" \
            -v "PROJECT_ROOT=${PROJECT_ROOT},SEED=${SEED},RATIO=${RATIO},TAG=${TAG}" \
            pbs_train_sw_smote_ratio.sh)
        
        echo "[$JOB_COUNT] Ratio=$RATIO Seed=$SEED -> $JOB (Queue: $QUEUE)"
        echo "$JOB $TAG" >> "$JOB_LOG"
        ALL_JOBS="$ALL_JOBS $JOB"
        
        # Submit evaluation with dependency
        JOB_ID=$(echo "$JOB" | cut -d'.' -f1)
        EVAL_JOB=$(qsub -W depend=afterok:$JOB \
            -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL=RF,TAG="$TAG",TRAIN_JOBID="$JOB_ID",SEED="$SEED" \
            pbs_evaluate.sh)
        echo "    -> Eval: $EVAL_JOB"
        echo "$EVAL_JOB eval_$TAG" >> "$JOB_LOG"
    done
done

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "Training jobs: $JOB_COUNT"
echo "Total jobs (train+eval): $((JOB_COUNT * 2))"
echo "Job log: $JOB_LOG"
echo ""
echo "Monitor: qstat -u \$USER"
echo "============================================================"

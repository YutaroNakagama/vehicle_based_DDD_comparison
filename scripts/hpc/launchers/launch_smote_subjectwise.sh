#!/bin/bash
# ============================================================
# Subject-wise SMOTE: Multi-Seed Launcher
# ============================================================
# Subject-wise SMOTEを複数シードで並列実行
# メモリ最適化: 6GB/job で同時実行可能な数を最大化
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$SCRIPT_DIR"

# 3シードで実験（国際会議向け統計的頑健性）
SEEDS="${SEEDS:-42 123 456}"

echo "============================================================"
echo "Subject-wise SMOTE - Multi-Seed Parallel Launcher"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Seeds: $SEEDS"
echo "Memory per job: 6GB (optimized)"
echo "Time: $(date)"
echo "============================================================"
echo ""

# Track all job IDs
ALL_JOBS=""
TRAIN_JOBS=""

echo "=== Submitting Training Jobs (Parallel) ==="

for SEED in $SEEDS; do
    JOB=$(qsub -v PBS_O_WORKDIR="$PROJECT_ROOT",SEED="$SEED" pbs_train_smote_subjectwise.sh)
    echo "  [SMOTE-SubjectWise] SEED=$SEED -> $JOB"
    TRAIN_JOBS="$TRAIN_JOBS $JOB"
    ALL_JOBS="$ALL_JOBS $JOB"
done

echo ""
echo "=== Submitting Evaluation Jobs (with dependencies) ==="

for SEED in $SEEDS; do
    # Find the matching training job
    JOB_IDX=$(($(echo $SEEDS | tr ' ' '\n' | grep -n "^${SEED}$" | cut -d: -f1)))
    TRAIN_JOB=$(echo $TRAIN_JOBS | tr ' ' '\n' | sed -n "${JOB_IDX}p")
    JOB_ID=$(echo "$TRAIN_JOB" | cut -d'.' -f1)
    
    TAG="imbal_v2_smote_subjectwise_seed${SEED}"
    
    EVAL_JOB=$(qsub -W depend=afterok:$TRAIN_JOB \
        -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL=RF,TAG="$TAG",TRAIN_JOBID="$JOB_ID",SEED="$SEED" \
        pbs_evaluate.sh)
    echo "  [Eval] SEED=$SEED -> $EVAL_JOB (after $JOB_ID)"
    ALL_JOBS="$ALL_JOBS $EVAL_JOB"
done

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "Training jobs: $(echo $TRAIN_JOBS | wc -w) (parallel)"
echo "Total jobs: $(echo $ALL_JOBS | wc -w)"
echo ""
echo "Monitor with: qstat -u \$USER"
echo "============================================================"

# Save job IDs to file
JOB_LOG="$PROJECT_ROOT/scripts/hpc/logs/imbalance/job_ids_smote_subjectwise_$(date +%Y%m%d_%H%M%S).txt"
echo "# Subject-wise SMOTE multi-seed submission: $(date)" > "$JOB_LOG"
echo "# Seeds: $SEEDS" >> "$JOB_LOG"
echo "# Training jobs (parallel):" >> "$JOB_LOG"
echo "$TRAIN_JOBS" | tr ' ' '\n' | grep -v '^$' >> "$JOB_LOG"
echo "# All jobs:" >> "$JOB_LOG"
echo "$ALL_JOBS" | tr ' ' '\n' | grep -v '^$' >> "$JOB_LOG"
echo ""
echo "Job IDs saved to: $JOB_LOG"

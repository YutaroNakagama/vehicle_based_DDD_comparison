#!/bin/bash
# ============================================================
# Imbalance Comparison V2: Multi-Seed Launcher
# ============================================================
# Run all imbalance experiments with multiple seeds for
# statistical robustness required by international conferences.
#
# Usage:
#   ./launch_multiseed.sh              # Run with default seeds (42, 123, 456)
#   ./launch_multiseed.sh 42 123       # Run with specified seeds
#   SEEDS="42 123 456 789" ./launch_multiseed.sh  # Via environment
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$SCRIPT_DIR"

# Default seeds (3-5 seeds recommended for international conferences)
DEFAULT_SEEDS="42 123 456"
SEEDS="${SEEDS:-${*:-$DEFAULT_SEEDS}}"

# Methods to run
METHODS="baseline smote smote_tomek smote_rus"

echo "============================================================"
echo "Imbalance Comparison V2 - Multi-Seed Launcher"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Seeds: $SEEDS"
echo "Methods: $METHODS"
echo "Time: $(date)"
echo "============================================================"
echo ""

# Track all job IDs
ALL_JOBS=""

for SEED in $SEEDS; do
    echo ""
    echo "=== Submitting jobs for SEED=$SEED ==="
    
    for METHOD in $METHODS; do
        SCRIPT="pbs_train_${METHOD}.sh"
        
        if [[ ! -f "$SCRIPT" ]]; then
            echo "  [SKIP] $SCRIPT not found"
            continue
        fi
        
        # Submit with SEED environment variable
        JOB=$(qsub -v PBS_O_WORKDIR="$PROJECT_ROOT",SEED="$SEED" "$SCRIPT")
        echo "  [$METHOD] SEED=$SEED -> $JOB"
        ALL_JOBS="$ALL_JOBS $JOB"
        
        # Extract job ID for evaluation dependency
        JOB_ID=$(echo "$JOB" | cut -d'.' -f1)
        
        # Tag now includes seed (matches the training script)
        TAG="imbal_v2_${METHOD}_seed${SEED}"
        
        # Submit evaluation job with dependency
        EVAL_JOB=$(qsub -W depend=afterok:$JOB \
            -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL=RF,TAG="$TAG",TRAIN_JOBID="$JOB_ID",SEED="$SEED" \
            pbs_evaluate.sh)
        echo "    -> Eval: $EVAL_JOB (after $JOB_ID)"
        ALL_JOBS="$ALL_JOBS $EVAL_JOB"
    done
done

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "Total jobs: $(echo $ALL_JOBS | wc -w)"
echo ""
echo "Monitor with: qstat -u \$USER"
echo "============================================================"

# Save job IDs to file
JOB_LOG="$PROJECT_ROOT/scripts/hpc/imbalance/job_ids_multiseed_$(date +%Y%m%d_%H%M%S).txt"
echo "# Multi-seed job submission: $(date)" > "$JOB_LOG"
echo "# Seeds: $SEEDS" >> "$JOB_LOG"
echo "$ALL_JOBS" | tr ' ' '\n' | grep -v '^$' >> "$JOB_LOG"
echo "Job IDs saved to: $JOB_LOG"

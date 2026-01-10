#!/bin/bash
# ============================================================
# Resubmit Failed Imbalance Experiments (Optimized)
# ============================================================
# Failed experiments:
#   1. smote ratio=0.5 seed=42 (walltime exceeded at 6h01m)
#   2. smote_subjectwise ratio=0.1 seed=42 (incomplete at 3h34m)
#   3. balanced_rf seed=123 (walltime exceeded at 6h01m)
#
# Optimizations applied:
#   - 4 CPU cores for parallel processing
#   - 16GB memory (increased from default)
#   - 12 hour walltime (safety margin)
#   - N_JOBS_OVERRIDE enabled for sklearn parallelization
# ============================================================

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison

SCRIPT="scripts/hpc/jobs/imbalance/pbs_imbalance_optimized.sh"
LOG_DIR="scripts/hpc/logs/imbalance"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Submitting Failed Imbalance Experiments (Optimized)"
echo "============================================================"
echo "Optimizations: 4 cores, 16GB RAM, 12h walltime"
echo ""

# Job 1: SMOTE ratio=0.5 seed=42
echo "[1/3] Submitting: smote ratio=0.5 seed=42"
JOB1=$(qsub \
    -N imb_smote_0.5_s42_opt \
    -l select=1:ncpus=4:mem=16gb \
    -l walltime=12:00:00 \
    -q DEFAULT \
    -v METHOD=smote,RATIO=0.5,SEED=42,NCPUS=4 \
    "$SCRIPT")
echo "  -> Job ID: $JOB1"

# Job 2: SMOTE Subjectwise ratio=0.1 seed=42
echo "[2/3] Submitting: smote_subjectwise ratio=0.1 seed=42"
JOB2=$(qsub \
    -N imb_sw_0.1_s42_opt \
    -l select=1:ncpus=4:mem=16gb \
    -l walltime=12:00:00 \
    -q DEFAULT \
    -v METHOD=smote_subjectwise,RATIO=0.1,SEED=42,NCPUS=4 \
    "$SCRIPT")
echo "  -> Job ID: $JOB2"

# Job 3: Balanced RF seed=123
echo "[3/3] Submitting: balanced_rf seed=123"
JOB3=$(qsub \
    -N imb_brf_s123_opt \
    -l select=1:ncpus=4:mem=16gb \
    -l walltime=12:00:00 \
    -q DEFAULT \
    -v METHOD=balanced_rf,RATIO=0,SEED=123,NCPUS=4 \
    "$SCRIPT")
echo "  -> Job ID: $JOB3"

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "============================================================"
echo ""
echo "Submitted Jobs:"
echo "  1. smote r=0.5 s=42     : $JOB1"
echo "  2. smote_sw r=0.1 s=42  : $JOB2"
echo "  3. balanced_rf s=123    : $JOB3"
echo ""
echo "Monitor with: qstat -u \$USER"
echo ""

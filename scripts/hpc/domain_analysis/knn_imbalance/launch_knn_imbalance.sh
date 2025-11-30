#!/bin/bash
# ============================================================
# Launch KNN Imbalance Comparison Experiments
# ============================================================
#
# Usage:
#   ./launch_knn_imbalance.sh          # Submit all jobs
#   ./launch_knn_imbalance.sh baseline # Submit baseline only
#   ./launch_knn_imbalance.sh imbal    # Submit imbalance jobs only
#   ./launch_knn_imbalance.sh eval     # Submit evaluation jobs
# ============================================================

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"

# Create log directories
mkdir -p "${PROJECT_ROOT}/logs/knn_imbalance/out"
mkdir -p "${PROJECT_ROOT}/logs/knn_imbalance/err"

cd "${PROJECT_ROOT}"

MODE="${1:-all}"

case "${MODE}" in
    baseline)
        echo "Submitting baseline jobs (6 jobs)..."
        qsub "${SCRIPT_DIR}/pbs_knn_baseline.sh"
        ;;
    imbal)
        echo "Submitting imbalance comparison jobs (24 jobs)..."
        qsub "${SCRIPT_DIR}/pbs_knn_imbalance.sh"
        ;;
    eval)
        echo "Submitting evaluation jobs (30 jobs)..."
        qsub "${SCRIPT_DIR}/pbs_knn_eval.sh"
        ;;
    all)
        echo "Submitting all training jobs..."
        echo ""
        echo "Step 1: Baseline (6 jobs)"
        BASELINE_JOB=$(qsub "${SCRIPT_DIR}/pbs_knn_baseline.sh")
        echo "  Submitted: ${BASELINE_JOB}"
        
        echo ""
        echo "Step 2: Imbalance comparison (24 jobs)"
        IMBAL_JOB=$(qsub "${SCRIPT_DIR}/pbs_knn_imbalance.sh")
        echo "  Submitted: ${IMBAL_JOB}"
        
        echo ""
        echo "============================================================"
        echo "Job Summary"
        echo "============================================================"
        echo "Baseline:   ${BASELINE_JOB}"
        echo "Imbalance:  ${IMBAL_JOB}"
        echo ""
        echo "After training completes, run evaluation:"
        echo "  $0 eval"
        echo "============================================================"
        
        # Save job IDs
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        echo "baseline=${BASELINE_JOB}" > "${SCRIPT_DIR}/.job_ids_${TIMESTAMP}.txt"
        echo "imbalance=${IMBAL_JOB}" >> "${SCRIPT_DIR}/.job_ids_${TIMESTAMP}.txt"
        echo ""
        echo "Job IDs saved to: ${SCRIPT_DIR}/.job_ids_${TIMESTAMP}.txt"
        ;;
    *)
        echo "Usage: $0 [baseline|imbal|eval|all]"
        exit 1
        ;;
esac

echo ""
echo "Check job status: qstat -u \$USER"

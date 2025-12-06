#!/bin/bash
# ============================================================
# Launch KNN Imbalance Comparison Experiments
# ============================================================
#
# Usage:
#   ./launch_knn_imbalance.sh          # Show help
#   ./launch_knn_imbalance.sh full     # Submit full comparison (135 jobs)
#   ./launch_knn_imbalance.sh eval     # Submit evaluation jobs (135 jobs)
#   ./launch_knn_imbalance.sh baseline # Submit baseline only (old, 6 jobs)
#   ./launch_knn_imbalance.sh imbal    # Submit imbalance only (old, 24 jobs)
# ============================================================

set -eu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"

# Create log directories
mkdir -p "${PROJECT_ROOT}/logs/knn_imbalance_full/out"
mkdir -p "${PROJECT_ROOT}/logs/knn_imbalance_full/err"
mkdir -p "${PROJECT_ROOT}/logs/knn_imbalance/out"
mkdir -p "${PROJECT_ROOT}/logs/knn_imbalance/err"

cd "${PROJECT_ROOT}"

MODE="${1:-help}"

case "${MODE}" in
    full)
        echo "============================================================"
        echo "Submitting FULL comparison (135 jobs)"
        echo "============================================================"
        echo ""
        echo "Experiment design:"
        echo "  - Distances: mmd, wasserstein, dtw (3)"
        echo "  - Modes: pooled, source_only, target_only (3)"
        echo "  - Levels: out_domain, mid_domain, in_domain (3)"
        echo "  - Methods: baseline, RUS, Tomek, SMOTE+RUS, SMOTE+Tomek (5)"
        echo "  - Total: 3 × 3 × 3 × 5 = 135 jobs"
        echo ""
        
        FULL_JOB=$(qsub "${SCRIPT_DIR}/pbs_knn_full.sh")
        echo "Submitted: ${FULL_JOB}"
        
        # Save job ID
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        echo "full=${FULL_JOB}" > "${SCRIPT_DIR}/.job_ids_full_${TIMESTAMP}.txt"
        echo ""
        echo "Job ID saved to: ${SCRIPT_DIR}/.job_ids_full_${TIMESTAMP}.txt"
        echo ""
        echo "After training completes, run evaluation:"
        echo "  $0 eval"
        ;;
    eval)
        echo "Submitting evaluation jobs (135 jobs)..."
        EVAL_JOB=$(qsub "${SCRIPT_DIR}/pbs_knn_full_eval.sh")
        echo "Submitted: ${EVAL_JOB}"
        ;;
    baseline)
        echo "Submitting baseline jobs (6 jobs)..."
        qsub "${SCRIPT_DIR}/pbs_knn_baseline.sh"
        ;;
    imbal)
        echo "Submitting imbalance comparison jobs (24 jobs)..."
        qsub "${SCRIPT_DIR}/pbs_knn_imbalance.sh"
        ;;
    all)
        echo "[DEPRECATED] Use 'full' instead for complete comparison"
        echo ""
        echo "Submitting old jobs (baseline + imbalance = 30 jobs)..."
        BASELINE_JOB=$(qsub "${SCRIPT_DIR}/pbs_knn_baseline.sh")
        echo "  Baseline: ${BASELINE_JOB}"
        IMBAL_JOB=$(qsub "${SCRIPT_DIR}/pbs_knn_imbalance.sh")
        echo "  Imbalance: ${IMBAL_JOB}"
        ;;
    help|*)
        echo "Usage: $0 [full|eval|baseline|imbal|all]"
        echo ""
        echo "Commands:"
        echo "  full     - Submit FULL comparison (135 jobs)"
        echo "             3 distances × 3 modes × 3 levels × 5 methods"
        echo "  eval     - Submit evaluation for full comparison (135 jobs)"
        echo "  baseline - Submit baseline only (old script, 6 jobs)"
        echo "  imbal    - Submit imbalance only (old script, 24 jobs)"
        echo "  all      - [DEPRECATED] Submit baseline + imbalance (30 jobs)"
        echo ""
        echo "Recommended workflow:"
        echo "  1. $0 full    # Train all 135 configurations"
        echo "  2. $0 eval    # Evaluate after training completes"
        echo "  3. python scripts/python/domain_analysis/collect_knn_imbalance_results.py --plot"
        exit 1
        ;;
esac

echo ""
echo "Check job status: qstat -u \$USER"

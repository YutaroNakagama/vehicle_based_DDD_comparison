#!/bin/bash
# =============================================================================
# visualize_domain_local.sh
# =============================================================================
# Visualize local domain analysis experiments with imbalance handling.
#
# Usage:
#   ./scripts/local/visualize_domain_local.sh [OPTIONS]
#
# Options:
#   --patterns PATTERNS   Comma-separated patterns (default: smote_plain)
#   --seed SEED           Filter by seed (optional)
#   --all                 Include all patterns: smote_plain,baseline_domain,imbalv3
# =============================================================================

set -e

cd "$(dirname "$0")/../.."
source .venv-linux/bin/activate

PATTERNS="smote_plain"
SEED=""
JOB_ID="local"

while [[ $# -gt 0 ]]; do
    case $1 in
        --patterns) PATTERNS="$2"; shift 2 ;;
        --seed) SEED="--seed $2"; shift 2 ;;
        --all) PATTERNS="smote_plain,baseline_domain,imbalv3"; shift ;;
        --job_id) JOB_ID="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "Domain Local Visualization"
echo "========================================"
echo "Patterns: ${PATTERNS}"
echo "Job ID: ${JOB_ID}"
echo "Seed: ${SEED:-all}"
echo "========================================"

python scripts/python/visualization/visualize_domain_local.py \
    --job_id "${JOB_ID}" \
    --patterns "${PATTERNS}" \
    ${SEED}

echo ""
echo "========================================"
echo "Output files:"
echo "========================================"
ls -la results/analysis/domain/imbalance/metrics/ 2>/dev/null || true
ls -la results/analysis/domain/imbalance/optuna/ 2>/dev/null || true
ls -la results/analysis/domain/imbalance/confusion/ 2>/dev/null || true
echo "========================================"

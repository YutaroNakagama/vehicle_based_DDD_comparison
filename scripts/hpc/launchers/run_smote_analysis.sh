#!/bin/bash
# ==============================================================================
# SMOTE Comparison Results Analysis Launcher
# ==============================================================================
#
# Run this script AFTER all HPC training jobs have completed.
# It executes the full analysis pipeline:
#   1. Aggregate results from all jobs
#   2. Generate comparison tables and statistical tests
#   3. Create publication-ready figures
#
# Usage:
#   ./scripts/hpc/launchers/run_smote_analysis.sh [job_ids...]
#
# Examples:
#   ./scripts/hpc/launchers/run_smote_analysis.sh
#   ./scripts/hpc/launchers/run_smote_analysis.sh 14653722 14653746
#
# ==============================================================================

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Activate conda environment
if command -v conda &> /dev/null; then
    echo -e "${BLUE}[INFO] Activating conda environment...${NC}"
    source ~/conda/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate python310 2>/dev/null || echo -e "${YELLOW}[WARN] conda activate failed, using system Python${NC}"
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

# Parse optional job ID arguments
JOB_ARGS=""
if [ $# -gt 0 ]; then
    JOB_ARGS="--jobs $*"
    echo -e "${BLUE}[INFO] Filtering to job IDs: $*${NC}"
fi

# Output directories (following project structure)
# - Imbalance-only (pooled): results/analysis/exp1_imbalance/smote_comparison/
# - Domain analysis (ranking): results/analysis/exp1_imbalance/smote_comparison/
IMBALANCE_DIR="results/analysis/exp1_imbalance/smote_comparison"
DOMAIN_DIR="results/analysis/exp1_imbalance/smote_comparison"
AGGREGATED_CSV="$IMBALANCE_DIR/aggregated_results.csv"

echo ""
echo "=============================================================="
echo -e "${GREEN}SMOTE Comparison Analysis Pipeline${NC}"
echo "=============================================================="
echo "Project root:   $PROJECT_ROOT"
echo "Imbalance dir:  $IMBALANCE_DIR"
echo "Domain dir:     $DOMAIN_DIR"
echo "=============================================================="
echo ""

# Check if jobs are still running
RUNNING=$(qstat -u "$USER" 2>/dev/null | grep -c " R " || echo "0")
QUEUED=$(qstat -u "$USER" 2>/dev/null | grep -c " Q " || echo "0")

if [ "$RUNNING" -gt 0 ] || [ "$QUEUED" -gt 0 ]; then
    echo -e "${YELLOW}[WARN] Jobs still active: ${RUNNING} running, ${QUEUED} queued${NC}"
    echo -e "${YELLOW}[WARN] Results may be incomplete. Continue? (y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Step 1: Aggregate results
echo ""
echo -e "${BLUE}[Step 1/3] Aggregating training results...${NC}"
echo "--------------------------------------------------------------"

python scripts/python/evaluation/aggregate_smote_results.py \
    --output "$AGGREGATED_CSV" \
    --models RF BalancedRF \
    $JOB_ARGS

if [ ! -f "$AGGREGATED_CSV" ]; then
    echo -e "${RED}[ERROR] Aggregation failed. No results file created.${NC}"
    exit 1
fi

TOTAL_RESULTS=$(wc -l < "$AGGREGATED_CSV")
echo -e "${GREEN}[OK] Aggregated $((TOTAL_RESULTS - 1)) experiment results${NC}"

# Step 2: Generate comparison tables
echo ""
echo -e "${BLUE}[Step 2/3] Generating comparison tables and statistical tests...${NC}"
echo "--------------------------------------------------------------"

python scripts/python/analysis/compare_smote_methods.py \
    --input "$AGGREGATED_CSV" \
    --output-imbalance "$IMBALANCE_DIR/" \
    --output-domain "$DOMAIN_DIR/" \
    --metric test_f1

echo -e "${GREEN}[OK] Comparison tables generated${NC}"

# Step 3: Create visualizations
echo ""
echo -e "${BLUE}[Step 3/3] Creating visualizations...${NC}"
echo "--------------------------------------------------------------"

python scripts/python/visualization/plot_smote_comparison.py \
    --input "$AGGREGATED_CSV" \
    --output-imbalance "$IMBALANCE_DIR/figures/" \
    --output-domain "$DOMAIN_DIR/figures/" \
    --metric test_f1

echo -e "${GREEN}[OK] Figures generated${NC}"

# Summary
echo ""
echo "=============================================================="
echo -e "${GREEN}Analysis Complete!${NC}"
echo "=============================================================="
echo ""
echo "Output files:"
echo ""
echo "📊 Imbalance-only results (pooled mode):"
ls -la "$IMBALANCE_DIR"/*.csv 2>/dev/null | awk '{print "   " $NF}'
ls -la "$IMBALANCE_DIR/figures"/*.png 2>/dev/null | awk '{print "   " $NF}'
echo ""
echo "📊 Domain analysis results (ranking-based):"
ls -la "$DOMAIN_DIR"/*.csv 2>/dev/null | awk '{print "   " $NF}'
ls -la "$DOMAIN_DIR/figures"/*.png 2>/dev/null | awk '{print "   " $NF}'
echo ""
echo "📄 LaTeX tables:"
ls -la "$IMBALANCE_DIR"/*.tex 2>/dev/null | awk '{print "   " $NF}' || echo "   (none)"
echo ""
echo "=============================================================="
echo "View results:"
echo "  cat $IMBALANCE_DIR/pooled_comparison.csv"
echo "  cat $DOMAIN_DIR/ranking_comparison.csv"
echo ""
echo "Copy figures to local machine:"
echo "  scp -r $USER@kagayaki:$PROJECT_ROOT/$IMBALANCE_DIR/figures ."
echo "  scp -r $USER@kagayaki:$PROJECT_ROOT/$DOMAIN_DIR/figures ."
echo "=============================================================="

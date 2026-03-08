#!/bin/bash
# ============================================================
# Exp3 batch evaluation script (2026-02-07)
# ============================================================
# Bulk re-evaluate jobs that trained successfully but failed evaluation
# Auto-extract tag/mode/jobid from each job model dir and run evaluate.py
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Environment setup
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODELS=(SvmW SvmA Lstm)
EVAL_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

echo "============================================================"
echo "  Exp3 Prior Research - Batch Evaluation"
echo "  $(date)"
echo "============================================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    MODEL_ROOT="$PROJECT_ROOT/models/$MODEL"
    echo "--- $MODEL ---"
    
    # Find all prior_* training result JSONs
    while IFS= read -r train_json; do
        [[ -z "$train_json" ]] && continue
        
        # Extract jobid from path: .../training/SvmW/14735927/14735927[1]/train_results_...
        base_dir=$(dirname "$train_json")
        jobid_dir=$(dirname "$base_dir")
        JOBID=$(basename "$jobid_dir")
        
        # Extract tag and mode from filename
        # Format: train_results_SvmW_source_only_prior_SvmW_undersample_..._split2_ratio0.5_s123.json
        fname=$(basename "$train_json" .json)
        # Remove prefix "train_results_MODEL_"
        rest="${fname#train_results_${MODEL}_}"
        # Extract mode (first word before prior_)
        MODE="${rest%%_prior_*}"
        # Extract tag (everything after mode_)
        TAG="${rest#${MODE}_}"
        
        # Check if eval already exists
        EVAL_DIR="$PROJECT_ROOT/results/outputs/evaluation/$MODEL/$JOBID"
        if find "$EVAL_DIR" -name "eval_results_*.json" -print -quit 2>/dev/null | grep -q .; then
            ((SKIP_COUNT++))
            continue
        fi
        
        echo "  [EVAL] $MODEL | $MODE | tag=$TAG | jobid=$JOBID"
        
        if python scripts/python/evaluation/evaluate.py \
            --model "$MODEL" \
            --tag "$TAG" \
            --mode "$MODE" \
            --jobid "$JOBID" 2>&1 | tail -3; then
            ((EVAL_COUNT++))
        else
            echo "  [FAIL] $MODEL | $MODE | tag=$TAG | jobid=$JOBID"
            ((FAIL_COUNT++))
        fi
        
    done < <(find "$PROJECT_ROOT/results/outputs/training/$MODEL" -name "*prior*split2*.json" 2>/dev/null)
done

echo ""
echo "============================================================"
echo "  Results"
echo "============================================================"
echo "  Evaluated: $EVAL_COUNT"
echo "  Skipped (already done): $SKIP_COUNT"
echo "  Failed: $FAIL_COUNT"
echo "============================================================"

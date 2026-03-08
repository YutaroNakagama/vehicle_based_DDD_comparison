#!/bin/bash
#PBS -N reeval_lstm_metrics
#PBS -q SINGLE
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/models/Lstm/reeval_metrics.log

# ============================================================
# Exp3 Lstm re-evaluation: add roc_auc / auc_pr (PBS version)
# ============================================================
# Run as PBS jobs because TensorFlow is required
# Extract mode/tag/seed/jobid from existing eval JSON and re-evaluate
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Environment setup
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL="Lstm"
EVAL_BASE="$PROJECT_ROOT/results/outputs/evaluation/$MODEL"
EVAL_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0
TOTAL=0

echo "============================================================"
echo "  Exp3 Re-Evaluation: $MODEL (add roc_auc / auc_pr)"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "============================================================"
echo ""

# Scan all existing eval JSONs
while IFS= read -r eval_json; do
    [[ -z "$eval_json" ]] && continue
    ((TOTAL++))

    # Extract jobid from directory
    run_dir=$(dirname "$eval_json")
    jobid_dir=$(dirname "$run_dir")
    JOBID=$(basename "$jobid_dir")

    # Extract mode, tag, seed from JSON metadata
    META=$(python3 -c "
import json, re
d = json.load(open('$eval_json'))
mode = d.get('mode', '')
tag  = d.get('tag', d.get('original_tag', ''))
m = re.search(r'_s(\d+)$', tag)
seed = m.group(1) if m else '42'
print(f'{mode}|{tag}|{seed}')
" 2>/dev/null)

    if [[ -z "$META" ]]; then
        echo "  [SKIP] Parse error: $(basename "$eval_json")"
        ((SKIP_COUNT++))
        continue
    fi

    IFS='|' read -r MODE TAG SEED <<< "$META"

    # Skip pooled (separate handling)
    if [[ "$MODE" == "pooled" ]]; then
        ((SKIP_COUNT++))
        continue
    fi

    # Skip if already has roc_auc
    HAS_AUC=$(python3 -c "
import json
d = json.load(open('$eval_json'))
auc = d.get('roc_auc')
print('yes' if auc is not None else 'no')
" 2>/dev/null)

    if [[ "$HAS_AUC" == "yes" ]]; then
        ((SKIP_COUNT++))
        continue
    fi

    echo "  [$EVAL_COUNT/$TOTAL] $MODEL | $MODE | seed=$SEED | jobid=$JOBID | tag=${TAG:0:60}..."

    if python scripts/python/evaluation/evaluate.py \
        --model "$MODEL" \
        --tag "$TAG" \
        --mode "$MODE" \
        --seed "$SEED" \
        --jobid "$JOBID" 2>&1 | tail -3; then
        ((EVAL_COUNT++))
    else
        echo "  [FAIL] $MODEL | $MODE | tag=$TAG | jobid=$JOBID"
        ((FAIL_COUNT++))
    fi

done < <(find "$EVAL_BASE" -name "eval_results_${MODEL}_*.json" 2>/dev/null | sort)

echo ""
echo "============================================================"
echo "  Results: $MODEL"
echo "============================================================"
echo "  Total scanned:   $TOTAL"
echo "  Re-evaluated:    $EVAL_COUNT"
echo "  Skipped:         $SKIP_COUNT"
echo "  Failed:          $FAIL_COUNT"
echo "  Finish:          $(date)"
echo "============================================================"

#!/bin/bash
# ============================================================
# Exp3 re-evaluation script: add roc_auc / auc_pr
# ============================================================
# Extract mode, tag, jobid, seed from existing eval JSON and
# Re-evaluate with updated eval code.
# SvmA can run on login node; Lstm requires TensorFlow (PBS recommended).
#
# Usage:
#   bash scripts/hpc/launchers/reeval_exp3_metrics.sh SvmA
#   bash scripts/hpc/launchers/reeval_exp3_metrics.sh Lstm
#   bash scripts/hpc/launchers/reeval_exp3_metrics.sh SvmA --dry-run
#   bash scripts/hpc/launchers/reeval_exp3_metrics.sh Lstm --force
# ============================================================
set -uo pipefail

MODEL="${1:-}"
shift || true
DRY_RUN=""
FORCE=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN="--dry-run" ;;
        --force)   FORCE="--force" ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <SvmA|Lstm> [--dry-run] [--force]"
    exit 1
fi

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Environment setup
export PATH=~/conda/envs/python310/bin:~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PYTHON="$HOME/conda/envs/python310/bin/python"

EVAL_BASE="$PROJECT_ROOT/results/outputs/evaluation/$MODEL"
EVAL_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0
TOTAL=0

echo "============================================================"
echo "  Exp3 Re-Evaluation: $MODEL (add roc_auc / auc_pr)"
echo "  $(date)"
echo "============================================================"
echo ""

# Scan all existing eval JSONs for this model
while IFS= read -r eval_json; do
    [[ -z "$eval_json" ]] && continue
    ((TOTAL++))

    # Extract jobid from directory: .../evaluation/MODEL/JOBID/JOBID[n]/eval_results_*.json
    run_dir=$(dirname "$eval_json")        # .../JOBID[n]
    jobid_dir=$(dirname "$run_dir")         # .../JOBID
    JOBID=$(basename "$jobid_dir")

    # Extract mode and tag from the JSON metadata
    META=$($PYTHON -c "
import json, sys, re
d = json.load(open('$eval_json'))
mode = d.get('mode', '')
tag  = d.get('tag', d.get('original_tag', ''))
# Extract seed from tag (e.g., _s42 or _s123)
m = re.search(r'_s(\d+)$', tag)
seed = m.group(1) if m else '42'
print(f'{mode}|{tag}|{seed}')
" 2>/dev/null)

    if [[ -z "$META" ]]; then
        echo "  [SKIP] Could not parse: $(basename "$eval_json")"
        ((SKIP_COUNT++))
        continue
    fi

    IFS='|' read -r MODE TAG SEED <<< "$META"

    # Skip pooled evaluations (handled separately)
    if [[ "$MODE" == "pooled" ]]; then
        ((SKIP_COUNT++))
        continue
    fi

    # Check if already has roc_auc (non-null) — skip check with --force
    if [[ "$FORCE" != "--force" ]]; then
        HAS_AUC=$($PYTHON -c "
import json
d = json.load(open('$eval_json'))
auc = d.get('roc_auc')
print('yes' if auc is not None else 'no')
" 2>/dev/null)

        if [[ "$HAS_AUC" == "yes" ]]; then
            ((SKIP_COUNT++))
            continue
        fi
    fi

    echo "  [$EVAL_COUNT] $MODEL | $MODE | seed=$SEED | jobid=$JOBID | tag=${TAG:0:60}..."

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        ((EVAL_COUNT++))
        continue
    fi

    # Re-evaluate
    if $PYTHON scripts/python/evaluation/evaluate.py \
        --model "$MODEL" \
        --tag "$TAG" \
        --mode "$MODE" \
        --seed "$SEED" \
        --jobid "$JOBID" 2>&1 | tail -2; then
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
echo "============================================================"

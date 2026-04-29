#!/bin/bash
# ============================================================
# Eval-only retry for SvmA runs broken by the JOBID-mismatch bug.
#
# The old PBS scripts set PBS_JOBID="manual_yyyymmdd_hhmmss".
# savers.py's regex `\d{5,}\[\d+\]` matched only the trailing 6-digit
# hhmmss component, so models were saved under
# `models/SvmA/<hhmmss>/<hhmmss>[1]/...`, but eval was passed
# `--jobid manual_yyyymmdd_hhmmss` and looked under the full path.
# Result: 'No model file found' and exit 0 with no _within/_cross.json.
#
# This script auto-discovers all today-fresh SvmA model dirs that
# look like `models/SvmA/<6digits>/`, parses the tag from the .pkl
# filename, and retries the eval step with --jobid=<6digits>.
#
# Run after in-flight jobs from the broken submitter complete:
#   bash scripts/hpc/launchers/eval_retry_inflight_svma.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Collect all SvmA model dirs created in the last 48h that have a
# domain_train pkl. Filter to numeric-only dir names (the bug signature).
N_RETRIED=0
N_SKIPPED=0
N_DONE=0

while read pkl; do
    [[ -z "$pkl" ]] && continue

    # Extract dir_id (6 digits, the buggy save id)
    DIR_ID=$(echo "$pkl" | sed -E 's|models/SvmA/([0-9]+)/.*|\1|')
    [[ -z "$DIR_ID" ]] && continue

    # Extract tag: filename is SvmA_domain_train_<TAG>_manual_yyyymmdd_<DIR_ID>_1.pkl
    BASENAME=$(basename "$pkl" .pkl)
    TAG=$(echo "$BASENAME" | sed -E "s/^SvmA_domain_train_(.+)_manual_[0-9]+_${DIR_ID}_[0-9]+\$/\1/")
    if [[ "$TAG" == "$BASENAME" ]]; then
        # Pattern didn't match — maybe newer fixed-script jobs (skip)
        continue
    fi

    # Skip if the canonical _within.json + _cross.json already exist (somewhere)
    if find results/outputs/evaluation/SvmA -name "eval_results_SvmA_domain_train_${TAG}_within.json" 2>/dev/null | grep -q . && \
       find results/outputs/evaluation/SvmA -name "eval_results_SvmA_domain_train_${TAG}_cross.json" 2>/dev/null | grep -q .; then
        ((N_DONE++))
        continue
    fi

    # Determine within/cross domain from tag
    if [[ "$TAG" == *"_in_domain_"* ]]; then
        DOM="in_domain"; CROSS="out_domain"
    else
        DOM="out_domain"; CROSS="in_domain"
    fi
    DIST=$(echo "$TAG" | grep -oE "knn_(mmd|dtw|wasserstein)_" | head -1 | sed -E 's/knn_(.+)_/\1/')
    if [[ -z "$DIST" ]]; then
        echo "[SKIP] $TAG : could not parse distance"
        ((N_SKIPPED++))
        continue
    fi

    TGT="results/analysis/exp2_domain_shift/distance/rankings/split2/knn/${DIST}_${DOM}.txt"
    CTGT="results/analysis/exp2_domain_shift/distance/rankings/split2/knn/${DIST}_${CROSS}.txt"

    echo "----------------------------------------"
    echo "[RETRY] tag=${TAG:0:80}"
    echo "        dir_id=$DIR_ID  dom=$DOM  dist=$DIST"

    if $DRY_RUN; then
        echo "  [DRY] eval within: $TGT"
        echo "  [DRY] eval cross:  $CTGT"
        ((N_RETRIED++))
        continue
    fi

    python scripts/python/evaluation/evaluate.py \
        --model SvmA --tag "$TAG" --mode domain_train \
        --target_file "$TGT" --eval_type within --jobid "$DIR_ID" \
        2>&1 | tail -2 || echo "  [WARN] within eval failed"

    python scripts/python/evaluation/evaluate.py \
        --model SvmA --tag "$TAG" --mode domain_train \
        --target_file "$CTGT" --eval_type cross --jobid "$DIR_ID" \
        2>&1 | tail -2 || echo "  [WARN] cross eval failed"

    ((N_RETRIED++))

done < <(find models/SvmA -mindepth 3 -maxdepth 3 -name "SvmA_domain_train_*.pkl" -mtime -2 2>/dev/null \
            | grep -E "^models/SvmA/[0-9]+/[0-9]+\[" | sort -u)

echo ""
echo "[DONE] retried=$N_RETRIED  already_complete=$N_DONE  skipped=$N_SKIPPED"

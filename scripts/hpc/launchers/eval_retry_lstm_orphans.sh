#!/bin/bash
# ============================================================
# Eval-only retry for Lstm domain_train tags whose `_within.json`
# exists but `_cross.json` is missing — orphans from Issue 5
# (cross-eval silent failure, 2026-04-07 batch).
#
# Auto-discovers any tag in this state by comparing within.json
# against cross.json sets, then re-runs only the cross eval
# pointing at the existing legacy model directory.
#
# Run on the login node (CPU is fine for eval; no GPU needed):
#   bash scripts/hpc/launchers/eval_retry_lstm_orphans.sh --dry-run
#   bash scripts/hpc/launchers/eval_retry_lstm_orphans.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 1. Discover orphan tags: have within.json but no cross.json
python3 - <<'PY' > /tmp/lstm_orphans.txt
import os, glob, re

WITHIN = {}
for f in glob.glob('results/outputs/evaluation/Lstm/**/*_within.json', recursive=True):
    if '_invalidated' in f: continue
    m = re.match(r'eval_results_Lstm_domain_train_(.+)_within\.json', os.path.basename(f))
    if m: WITHIN[m.group(1)] = f

CROSS = set()
for f in glob.glob('results/outputs/evaluation/Lstm/**/*_cross.json', recursive=True):
    if '_invalidated' in f: continue
    m = re.match(r'eval_results_Lstm_domain_train_(.+)_cross\.json', os.path.basename(f))
    if m: CROSS.add(m.group(1))

orphans = {tag: path for tag, path in WITHIN.items() if tag not in CROSS}
for tag in orphans:
    # Find legacy model jobid by looking up the Keras file
    keras = glob.glob(f'models/Lstm/**/Lstm_domain_train_{tag}_*.keras', recursive=True)
    if not keras: continue
    # filename: Lstm_domain_train_<tag>_<jid>_<arr>.keras
    m = re.match(rf'Lstm_domain_train_{re.escape(tag)}_(\d+)_(\d+)\.keras', os.path.basename(keras[0]))
    if not m: continue
    jid = m.group(1)
    print(f"{tag}\t{jid}")
PY

ORPHAN_COUNT=$(wc -l < /tmp/lstm_orphans.txt)
echo "[INFO] Found $ORPHAN_COUNT Lstm orphan tags"

while IFS=$'\t' read -r TAG JID; do
    [[ -z "$TAG" ]] && continue

    # Determine cross-domain target file from tag's _<dom>_ token
    if [[ "$TAG" == *"_in_domain_"* ]]; then
        CROSS_DOM="out_domain"
    else
        CROSS_DOM="in_domain"
    fi
    DIST=$(echo "$TAG" | grep -oE 'knn_(mmd|dtw|wasserstein)_' | sed -E 's/knn_(.+)_/\1/' | head -1)
    [[ -z "$DIST" ]] && { echo "[SKIP] $TAG: no distance parsed"; continue; }

    CTGT="results/analysis/exp2_domain_shift/distance/rankings/split2/knn/${DIST}_${CROSS_DOM}.txt"

    echo "----------------------------------------"
    echo "[RETRY-CROSS] tag=${TAG:0:80}"
    echo "             jid=$JID  cross_target=${CROSS_DOM}"

    if $DRY_RUN; then
        echo "  [DRY] would run cross eval"
        continue
    fi

    python scripts/python/evaluation/evaluate.py \
        --model Lstm --tag "$TAG" --mode domain_train \
        --target_file "$CTGT" --eval_type cross --jobid "$JID" \
        2>&1 | tail -3 || echo "[WARN] cross eval failed"
done < /tmp/lstm_orphans.txt

echo ""
echo "[DONE]"

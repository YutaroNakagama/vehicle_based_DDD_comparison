#!/bin/bash
# ============================================================
# Re-run eval-only for every completed Lstm tag using the fixed
# lstm_eval (validation-set threshold tuning, no test-set leakage).
#
# Models from previous runs are reused (no re-training); only the
# eval step runs, overwriting eval_results_*_within.json /
# eval_results_*_cross.json with the corrected metrics.
#
# This must be run AFTER lstm_eval is patched
# (src/evaluation/models/lstm.py: opt_threshold uses (X_val, y_val)).
#
# Submits PBS array of one job per tag×eval_type, dispatched to
# CPU queues (eval is fast enough on CPU and avoids the H100 GPU
# init issues).
#
# Usage:
#   bash scripts/hpc/launchers/eval_retry_all_lstm.sh --dry-run
#   bash scripts/hpc/launchers/eval_retry_all_lstm.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Build the (tag, jid, kind) tuple list from existing eval JSONs + their
# model directories.
python3 - <<'PY' > /tmp/lstm_eval_retry_targets.txt
import os, glob, re

def yield_targets():
    for f in glob.glob('results/outputs/evaluation/Lstm/**/eval_results_Lstm_domain_train_*.json', recursive=True):
        if '_invalidated' in f:
            continue
        n = os.path.basename(f)
        m = re.match(r'eval_results_Lstm_domain_train_(.+)_(within|cross)\.json', n)
        if not m:
            continue
        tag, kind = m.group(1), m.group(2)
        # Locate the model dir for this tag — find any keras file matching
        keras = glob.glob(f'models/Lstm/**/Lstm_domain_train_{tag}_*.keras', recursive=True)
        if not keras:
            continue
        # Extract jid from model filename: Lstm_..._{jid}_{arr}.keras
        mfn = os.path.basename(keras[0])
        mt = re.match(rf'Lstm_domain_train_{re.escape(tag)}_(\d+)_(\d+)\.keras', mfn)
        if not mt:
            continue
        jid = mt.group(1)
        yield (tag, jid, kind)

# Print unique targets
seen = set()
for t, j, k in yield_targets():
    key = (t, k)
    if key in seen: continue
    seen.add(key)
    print(f"{t}\t{j}\t{k}")
PY

N=$(wc -l < /tmp/lstm_eval_retry_targets.txt)
echo "[INFO] Targets discovered: $N"

if $DRY_RUN; then
    head -10 /tmp/lstm_eval_retry_targets.txt | awk -F'\t' '{printf "[DRY] tag=%.50s... jid=%s kind=%s\n", $1, $2, $3}'
    echo "..."
    tail -3 /tmp/lstm_eval_retry_targets.txt | awk -F'\t' '{printf "[DRY] tag=%.50s... jid=%s kind=%s\n", $1, $2, $3}'
    echo "[INFO] Dry-run complete; would submit $N qsub jobs"
    exit 0
fi

# Submit one PBS job per tag×kind, distributed across CPU queues
CPU_QUEUES=("SINGLE" "SMALL" "LONG" "LARGE" "DEF" "VM-CPU" "VM-LM")
IDX=0
N_SUB=0
N_ERR=0
while IFS=$'\t' read -r TAG JID KIND; do
    [[ -z "$TAG" ]] && continue

    # Determine target file
    DIST=$(echo "$TAG" | grep -oE 'knn_(mmd|dtw|wasserstein)_' | sed -E 's/knn_(.+)_/\1/' | head -1)
    if [[ "$KIND" == "within" ]]; then
        if [[ "$TAG" == *"_in_domain_"* ]]; then DOM="in_domain"; else DOM="out_domain"; fi
    else
        # cross uses opposite domain
        if [[ "$TAG" == *"_in_domain_"* ]]; then DOM="out_domain"; else DOM="in_domain"; fi
    fi
    TGT="results/analysis/exp2_domain_shift/distance/rankings/split2/knn/${DIST}_${DOM}.txt"

    QUEUE="${CPU_QUEUES[$((IDX % ${#CPU_QUEUES[@]}))]}"
    ((IDX++))
    JOBNAME="Lr_${KIND:0:1}_${JID}_$(echo "$TAG" | tr -dc '[:alnum:]' | tail -c 12)"

    OUT=$(qsub -N "$JOBNAME" \
        -l select=1:ncpus=4:mem=16gb -l walltime=01:00:00 -q "$QUEUE" \
        -v PROJECT=$PROJECT_ROOT,MODEL=Lstm,TAG=$TAG,KIND=$KIND,JID=$JID,TGT=$TGT \
        /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_lstm_eval_retry.sh 2>&1)
    if [[ $? -eq 0 ]]; then
        echo "[SUB] $JOBNAME -> $OUT"
        ((N_SUB++))
    else
        echo "[ERR] $JOBNAME -> $(echo "$OUT" | grep -oE 'QOS[A-Za-z]*|error[^,]*' | head -1)"
        ((N_ERR++))
    fi
    sleep 0.2
done < /tmp/lstm_eval_retry_targets.txt

echo ""
echo "[DONE] Submitted=$N_SUB  Errored=$N_ERR  Total=$N"

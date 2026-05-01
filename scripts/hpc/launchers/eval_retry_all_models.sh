#!/bin/bash
# ============================================================
# Re-run eval-only for every completed SvmW / SvmA / Lstm tag using
# the new F1-on-validation threshold tuner.
#
# Pre-fix saved JSONs are detectable by the absence of `threshold_beta`
# (or `threshold_beta != 1.0`); they are scheduled for retry. Models
# already on disk are reused — no re-training.
#
# Usage:
#   bash scripts/hpc/launchers/eval_retry_all_models.sh --dry-run
#   bash scripts/hpc/launchers/eval_retry_all_models.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Build an INDEX of all model artifacts first (one walk, then O(1) lookups
# instead of glob-per-tag which would be 4M+ stats with 5k model dirs).
python3 - <<'PY' > /tmp/eval_retry_all_targets.txt
import os, glob, re, json
from collections import defaultdict

# 1) Index all model artifacts: tag -> jid (latest by mtime if multiple).
# os.walk is faster than recursive glob for our deep-but-narrow tree.
tag_to_jid = {}
for model in ['SvmW', 'SvmA', 'Lstm']:
    pat = '.keras' if model == 'Lstm' else '.pkl'
    skip = ('scaler', 'feature', 'selected', 'pso_history') if model != 'Lstm' else ('fold',)
    base = f'models/{model}'
    if not os.path.isdir(base): continue
    for root, _, files in os.walk(base):
        for n in files:
            if not n.endswith(pat): continue
            if any(s in n for s in skip): continue
            m = re.match(rf'{model}_domain_train_(.+)_(\d+)_(\d+)' + re.escape(pat), n)
            if not m: continue
            tag, jid, _ = m.groups()
            key = (model, tag)
            f = os.path.join(root, n)
            mt = os.path.getmtime(f)
            prev = tag_to_jid.get(key)
            if prev is None or mt > prev[1]:
                tag_to_jid[key] = (jid, mt)

# 2) Scan eval JSONs; emit (model, tag, jid, kind) for those needing retry
seen = set()
for model in ['SvmW', 'SvmA', 'Lstm']:
    base = f'results/outputs/evaluation/{model}'
    if not os.path.isdir(base): continue
    for root, _, files in os.walk(base):
        if '_invalidated' in root: continue
        for n in files:
            if not (n.startswith(f'eval_results_{model}_domain_train_') and n.endswith('.json')):
                continue
            f = os.path.join(root, n)
            try:
                with open(f) as fh: d = json.load(fh)
            except Exception: continue
            if d.get('threshold_beta') == 1.0:
                continue  # already post-fix
            m = re.match(rf'eval_results_{model}_domain_train_(.+)_(within|cross)\.json', n)
            if not m: continue
            tag, kind = m.group(1), m.group(2)
            if (model, tag, kind) in seen: continue
            seen.add((model, tag, kind))
            info = tag_to_jid.get((model, tag))
            if not info: continue
            print(f"{model}\t{tag}\t{info[0]}\t{kind}")
PY

N=$(wc -l < /tmp/eval_retry_all_targets.txt)
echo "[INFO] Pre-fix targets discovered: $N"
echo "[INFO] By model:"
awk -F'\t' '{print $1}' /tmp/eval_retry_all_targets.txt | sort | uniq -c

if $DRY_RUN; then
    echo "[INFO] Dry-run complete; would submit $N qsub jobs"
    exit 0
fi

# SvmW/SvmA eval-only is light (~5-15min). Lstm eval involves model loading
# and knn inference which can exceed 30min, so use 2h walltime and exclude TINY
# (TINY has a hard 30min cap).
CPU_QUEUES=("SINGLE" "SMALL" "LONG" "LARGE" "DEF" "VM-CPU" "VM-LM" "LONG-L" "XLARGE" "X2LARGE")
CPU_QUEUES_LSTM=("SINGLE" "SMALL" "LONG" "LARGE" "DEF" "VM-CPU" "VM-LM" "LONG-L" "XLARGE" "X2LARGE")

# Build a set of full job names already present in the queue (any non-C state)
# so we never resubmit the same Re_* tag twice when this script is invoked
# repeatedly by auto_resubmit_lstm_eval_retry.sh. qstat -u truncates the Name
# column, so we must use `qstat -f` for full names. Parallelise to keep the
# probe fast even with thousands of jobs in queue.
ACTIVE_NAMES_FILE=$(mktemp)
trap "rm -f $ACTIVE_NAMES_FILE" EXIT
qstat -u "$USER" 2>/dev/null | awk 'NR>5 && $10 != "C" {print $1}' \
    | sed 's/\..*//' \
    | xargs -r -n1 -P 16 -I{} bash -c 'qstat -f {} 2>/dev/null | awk -F" = " "/Job_Name/{print \$2; exit}"' \
    > "$ACTIVE_NAMES_FILE" 2>/dev/null
N_ACTIVE=$(wc -l < "$ACTIVE_NAMES_FILE")
echo "[INFO] Active job names indexed: $N_ACTIVE (skipping resubmits with these names)"

IDX=0
N_SUB=0
N_ERR=0
N_SKIP=0
while IFS=$'\t' read -r MODEL TAG JID KIND; do
    [[ -z "$MODEL" ]] && continue

    DIST=$(echo "$TAG" | grep -oE 'knn_(mmd|dtw|wasserstein)_' | sed -E 's/knn_(.+)_/\1/' | head -1)
    if [[ "$KIND" == "within" ]]; then
        if [[ "$TAG" == *"_in_domain_"* ]]; then DOM="in_domain"; else DOM="out_domain"; fi
    else
        if [[ "$TAG" == *"_in_domain_"* ]]; then DOM="out_domain"; else DOM="in_domain"; fi
    fi
    TGT="results/analysis/exp2_domain_shift/distance/rankings/split2/knn/${DIST}_${DOM}.txt"

    if [[ "$MODEL" == "Lstm" ]]; then
        QUEUE="${CPU_QUEUES_LSTM[$((IDX % ${#CPU_QUEUES_LSTM[@]}))]}"
        WALLTIME="02:00:00"
    else
        QUEUE="${CPU_QUEUES[$((IDX % ${#CPU_QUEUES[@]}))]}"
        WALLTIME="00:30:00"
    fi
    ((IDX++))
    SHORT_TAG=$(echo "$TAG" | tr -dc '[:alnum:]' | tail -c 10)
    JOBNAME="Re_${MODEL:0:2}${KIND:0:1}_${JID}_${SHORT_TAG}"

    # Skip if a job with this name is already queued/running (prevents
    # duplicates when auto-resubmit invokes this script repeatedly).
    if grep -qxF "$JOBNAME" "$ACTIVE_NAMES_FILE"; then
        ((N_SKIP++))
        continue
    fi

    OUT=$(qsub -N "$JOBNAME" \
        -l select=1:ncpus=2:mem=4gb -l walltime=$WALLTIME -q "$QUEUE" \
        -v PROJECT=$PROJECT_ROOT,MODEL=$MODEL,TAG=$TAG,KIND=$KIND,JID=$JID,TGT=$TGT \
        scripts/hpc/jobs/train/pbs_lstm_eval_retry.sh 2>&1)
    if [[ $? -eq 0 ]]; then
        echo "[SUB] $JOBNAME [$QUEUE] -> $OUT"
        ((N_SUB++))
    else
        echo "[ERR] $JOBNAME -> $(echo "$OUT" | grep -oE 'QOS[A-Za-z]*|error[^,]*' | head -1)"
        ((N_ERR++))
    fi
    # No artificial sleep — qsub itself rate-limits.
done < /tmp/eval_retry_all_targets.txt

echo ""
echo "[DONE] Submitted=$N_SUB  Skipped(in-queue)=$N_SKIP  Errored=$N_ERR  Total=$N"

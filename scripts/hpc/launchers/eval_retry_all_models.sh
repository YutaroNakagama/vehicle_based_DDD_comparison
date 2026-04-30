#!/bin/bash
# ============================================================
# Re-run eval-only for every completed SvmW / SvmA / Lstm tag using
# the new F1-on-validation threshold tuner.
#
# After fix(eval): F1 threshold tuning on validation (commit ecc94dd):
#   - lstm_eval, SvmA_eval, optimize_threshold_f2 all switched β=2→1
#   - lstm_eval and SvmA_eval also moved from test-set to val-set tuning
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

# Build the (model, tag, jid, kind) list — for each tag whose existing eval
# JSON pre-dates the F1 fix, plus has a model artifact on disk for eval-only.
python3 - <<'PY' > /tmp/eval_retry_all_targets.txt
import os, glob, re, json

def is_pre_fix(path):
    """A JSON is pre-fix iff threshold_beta is missing or != 1.0."""
    try:
        with open(path) as f: d = json.load(f)
    except Exception: return False
    return d.get('threshold_beta') != 1.0

def find_model(model, tag):
    """Return (jid, ext) tuple or None."""
    if model == 'Lstm':
        ks = glob.glob(f'models/Lstm/**/Lstm_domain_train_{tag}_*.keras', recursive=True)
        if ks:
            m = re.match(rf'Lstm_domain_train_{re.escape(tag)}_(\d+)_(\d+)\.keras', os.path.basename(ks[0]))
            if m: return m.group(1), 'keras'
    else:
        # SvmW / SvmA: pickled model
        ps = glob.glob(f'models/{model}/**/{model}_domain_train_{tag}_*.pkl', recursive=True)
        ps = [p for p in ps if 'scaler' not in p and 'feature' not in p and 'selected' not in p and 'pso_history' not in p]
        if ps:
            m = re.match(rf'{model}_domain_train_{re.escape(tag)}_(\d+)_(\d+)\.pkl', os.path.basename(ps[0]))
            if m: return m.group(1), 'pkl'
    return None

seen = set()
for model in ['SvmW', 'SvmA', 'Lstm']:
    for f in glob.glob(f'results/outputs/evaluation/{model}/**/eval_results_{model}_domain_train_*.json', recursive=True):
        if '_invalidated' in f: continue
        if not is_pre_fix(f): continue
        n = os.path.basename(f)
        m = re.match(rf'eval_results_{model}_domain_train_(.+)_(within|cross)\.json', n)
        if not m: continue
        tag, kind = m.group(1), m.group(2)
        if (model, tag, kind) in seen: continue
        seen.add((model, tag, kind))
        info = find_model(model, tag)
        if not info: continue
        jid, ext = info
        print(f"{model}\t{tag}\t{jid}\t{kind}")
PY

N=$(wc -l < /tmp/eval_retry_all_targets.txt)
echo "[INFO] Pre-fix targets discovered: $N"
echo "[INFO] By model:"
awk -F'\t' '{print $1}' /tmp/eval_retry_all_targets.txt | sort | uniq -c

if $DRY_RUN; then
    echo "[INFO] Dry-run complete; would submit $N qsub jobs"
    exit 0
fi

# Submit one PBS job per tuple, distributed across CPU queues
CPU_QUEUES=("SINGLE" "SMALL" "LONG" "LARGE" "DEF" "VM-CPU" "VM-LM" "LONG-L")
IDX=0
N_SUB=0
N_ERR=0
while IFS=$'\t' read -r MODEL TAG JID KIND; do
    [[ -z "$MODEL" ]] && continue

    DIST=$(echo "$TAG" | grep -oE 'knn_(mmd|dtw|wasserstein)_' | sed -E 's/knn_(.+)_/\1/' | head -1)
    if [[ "$KIND" == "within" ]]; then
        if [[ "$TAG" == *"_in_domain_"* ]]; then DOM="in_domain"; else DOM="out_domain"; fi
    else
        if [[ "$TAG" == *"_in_domain_"* ]]; then DOM="out_domain"; else DOM="in_domain"; fi
    fi
    TGT="results/analysis/exp2_domain_shift/distance/rankings/split2/knn/${DIST}_${DOM}.txt"

    QUEUE="${CPU_QUEUES[$((IDX % ${#CPU_QUEUES[@]}))]}"
    ((IDX++))
    SHORT_TAG=$(echo "$TAG" | tr -dc '[:alnum:]' | tail -c 10)
    JOBNAME="Re_${MODEL:0:2}${KIND:0:1}_${JID}_${SHORT_TAG}"

    OUT=$(qsub -N "$JOBNAME" \
        -l select=1:ncpus=4:mem=16gb -l walltime=02:00:00 -q "$QUEUE" \
        -v PROJECT=$PROJECT_ROOT,MODEL=$MODEL,TAG=$TAG,KIND=$KIND,JID=$JID,TGT=$TGT \
        scripts/hpc/jobs/train/pbs_lstm_eval_retry.sh 2>&1)
    if [[ $? -eq 0 ]]; then
        echo "[SUB] $JOBNAME [$QUEUE] -> $OUT"
        ((N_SUB++))
    else
        echo "[ERR] $JOBNAME -> $(echo "$OUT" | grep -oE 'QOS[A-Za-z]*|error[^,]*' | head -1)"
        ((N_ERR++))
    fi
    sleep 0.2
done < /tmp/eval_retry_all_targets.txt

echo ""
echo "[DONE] Submitted=$N_SUB  Errored=$N_ERR  Total=$N"

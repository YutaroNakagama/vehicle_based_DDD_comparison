#!/bin/bash
# Auto-resubmit eval-retry (SvmW / SvmA / Lstm) until all models are fixed.
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SUBMITTER="$PROJECT_ROOT/scripts/hpc/launchers/eval_retry_all_models.sh"
cd "$PROJECT_ROOT"

INTERVAL_SEC=900
MAX_PASSES=200

pass=0
while true; do
    pass=$((pass + 1))
    active=$(qstat -u s2240011 2>/dev/null | awk 'NR>5 && $10 != "C"' | wc -l)
    # Count JSONs across all three models that still lack threshold_beta=1.0
    remaining=$(python3 - <<'PY'
import os, glob, json
n_need = 0
for model in ('SvmW', 'SvmA', 'Lstm'):
    pattern = f'results/outputs/evaluation/{model}/**/eval_results_{model}_domain_train_*.json'
    for f in glob.glob(pattern, recursive=True):
        if '_invalidated' in f: continue
        try:
            with open(f) as fh: d = json.load(fh)
        except Exception: continue
        if d.get('threshold_beta') != 1.0:
            n_need += 1
print(n_need)
PY
)
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pass $pass | active=$active | retry_needed=$remaining"
    if [[ "$remaining" -eq 0 ]]; then
        echo "[INFO] All models (SvmW/SvmA/Lstm) eval-retry complete — exiting."
        break
    fi
    bash "$SUBMITTER" 2>&1 | tail -3

    if [[ "$pass" -ge "$MAX_PASSES" ]]; then break; fi
    sleep "$INTERVAL_SEC"
done

#!/bin/bash
# Auto-resubmit Lstm eval-only retry as CPU slots free up.
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
    # Count Lstm tags that still need retry (i.e., model exists but has no JSON
    # newer than today's lstm_eval fix)
    remaining=$(python3 - <<'PY'
import os, glob, re
# A tag needs retry if its eval JSON's threshold_source != "val" (or missing)
import json
n_need = 0
for f in glob.glob('results/outputs/evaluation/Lstm/**/eval_results_Lstm_domain_train_*.json', recursive=True):
    if '_invalidated' in f: continue
    try:
        with open(f) as fh: d = json.load(fh)
    except Exception: continue
    if d.get('threshold_source') != 'val':
        n_need += 1
print(n_need)
PY
)
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Pass $pass | active=$active | retry_needed=$remaining"
    if [[ "$remaining" -eq 0 ]]; then
        echo "[INFO] No Lstm eval-retry work remaining — exiting."
        break
    fi
    bash "$SUBMITTER" 2>&1 | tail -3

    if [[ "$pass" -ge "$MAX_PASSES" ]]; then break; fi
    sleep "$INTERVAL_SEC"
done

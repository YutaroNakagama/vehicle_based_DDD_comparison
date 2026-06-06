#!/usr/bin/env bash
# Cleanup script — runs ONLY after manual GO decision.
# Deletes all sklearn-era SvmA evaluation JSONs + model dirs so the cuML full
# re-run produces a uniform set of 144 outputs.
#
# Safe to run dry-run first:
#   DRY_RUN=1 bash scripts/bash/svma_cleanup_for_cuml_rerun.sh
#
# Live:
#   bash scripts/bash/svma_cleanup_for_cuml_rerun.sh
set -euo pipefail

REPO="/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison"
EVAL="${REPO}/results/outputs/evaluation/SvmA"
MODELS="${REPO}/models/SvmA"
DRY="${DRY_RUN:-0}"

echo "[CLEANUP] DRY_RUN=${DRY}"
echo "[CLEANUP] Eval root : ${EVAL}"
echo "[CLEANUP] Model root: ${MODELS}"

cnt_within=$(find "${EVAL}" -type f -name "*_within.json" 2>/dev/null | wc -l)
cnt_cross=$(find "${EVAL}"  -type f -name "*_cross.json"  2>/dev/null | wc -l)
cnt_jobid_dirs=$(find "${EVAL}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
cnt_model_dirs=$(find "${MODELS}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

echo "[CLEANUP] Currently on disk:"
echo "  within JSONs : ${cnt_within}"
echo "  cross  JSONs : ${cnt_cross}"
echo "  jobid dirs   : ${cnt_jobid_dirs} (eval)"
echo "  model dirs   : ${cnt_model_dirs} (models)"

if [ "${DRY}" = "1" ]; then
    echo "[CLEANUP] DRY_RUN — no files deleted. Exiting."
    exit 0
fi

read -r -p "Delete the above? Type 'YES' to proceed: " confirm
if [ "${confirm}" != "YES" ]; then
    echo "[CLEANUP] Aborted."
    exit 1
fi

# Wipe evaluation JSONs (within + cross) and the per-jobid folders that hold them
find "${EVAL}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
# Wipe SvmA model dirs (joblib .pkl + scaler + selected_features)
find "${MODELS}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

echo "[CLEANUP] Done. Eval & model dirs cleared for cuML re-run."

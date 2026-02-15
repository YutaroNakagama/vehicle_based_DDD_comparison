#!/bin/bash
# ============================================================
# Wait for SvmW preprocessing to finish, then start auto-resub
# ============================================================
set -uo pipefail

PREPROC_PID="${1:?Usage: $0 <preprocess PID>}"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"

echo "[$(date)] Waiting for SvmW preprocessing PID=$PREPROC_PID to finish..."

while kill -0 "$PREPROC_PID" 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Preprocessing PID=$PREPROC_PID completed."

# Verify preprocessed files
EXPECTED=87
ACTUAL=$(ls "$PROJECT_ROOT/data/processed/SvmW/" | wc -l)
echo "[$(date)] Processed files: $ACTUAL / $EXPECTED"

# Check new columns (should only have SteeringWheel, no LaneOffset)
HEADER=$(head -1 "$PROJECT_ROOT/data/interim/wavelet/SvmW/wavelet_S0113_1.csv")
if echo "$HEADER" | grep -q "LaneOffset"; then
    echo "[$(date)] ERROR: Old columns (LaneOffset) still present! Aborting."
    exit 1
else
    echo "[$(date)] OK: Only SteeringWheel columns found (8 features)."
fi

if [[ $ACTUAL -lt $EXPECTED ]]; then
    echo "[$(date)] WARNING: Only $ACTUAL / $EXPECTED files. Proceeding anyway."
fi

echo "[$(date)] Starting SvmW training auto-resub daemon..."
exec bash "$PROJECT_ROOT/scripts/hpc/launchers/auto_resub_svmw_zhao2009.sh"

#!/bin/bash
# ============================================================
# Wait for LSTM preprocessing to finish, then start auto-resub
# ============================================================
# Monitors PID of preprocess, then verifies result and starts
# the training job auto-resubmission daemon.
#
# Usage:
#   nohup bash scripts/hpc/launchers/wait_and_launch_lstm.sh <PREPROC_PID> &
# ============================================================

set -uo pipefail

PREPROC_PID="${1:?Usage: $0 <preprocess PID>}"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"

echo "[$(date)] Waiting for preprocessing PID=$PREPROC_PID to finish..."

# Wait for preprocess to complete
while kill -0 "$PREPROC_PID" 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Preprocessing PID=$PREPROC_PID completed."

# Verify preprocessed files
EXPECTED=87
ACTUAL=$(ls "$PROJECT_ROOT/data/processed/Lstm/" | wc -l)
echo "[$(date)] Processed files: $ACTUAL / $EXPECTED"

# Check new columns exist
HEADER=$(head -1 "$PROJECT_ROOT/data/interim/smooth_std_pe/Lstm/smooth_std_pe_S0113_1.csv")
if echo "$HEADER" | grep -q "speed_std_dev"; then
    echo "[$(date)] OK: New signal columns (speed) found."
else
    echo "[$(date)] ERROR: New signal columns NOT found! Aborting."
    exit 1
fi

if [[ $ACTUAL -lt $EXPECTED ]]; then
    echo "[$(date)] WARNING: Only $ACTUAL / $EXPECTED files. Proceeding anyway."
fi

echo "[$(date)] Starting LSTM training auto-resub daemon..."
exec bash "$PROJECT_ROOT/scripts/hpc/launchers/auto_resub_lstm_wang2022.sh"

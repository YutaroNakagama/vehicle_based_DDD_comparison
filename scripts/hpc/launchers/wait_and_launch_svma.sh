#!/bin/bash
# ============================================================
# Wait for SvmA preprocessing to finish, then start auto-resub
# ============================================================
set -uo pipefail

PREPROC_PID="${1:?Usage: $0 <preprocess PID>}"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"

echo "[$(date)] Waiting for SvmA preprocessing PID=$PREPROC_PID to finish..."

while kill -0 "$PREPROC_PID" 2>/dev/null; do
    sleep 30
done

echo "[$(date)] Preprocessing PID=$PREPROC_PID completed."

# Verify preprocessed files exist
INTERIM_DIR="$PROJECT_ROOT/data/interim/time_freq_domain/SvmA"
ACTUAL=$(ls "$INTERIM_DIR" 2>/dev/null | wc -l)
echo "[$(date)] Interim files: $ACTUAL"

# Check columns: should have Steering_ and SteeringSpeed_ only, no Lateral_ / LongAcc_ / LaneOffset_
SAMPLE=$(ls "$INTERIM_DIR"/*.csv 2>/dev/null | head -1)
if [[ -n "$SAMPLE" ]]; then
    HEADER=$(head -1 "$SAMPLE")
    if echo "$HEADER" | grep -q "LongAcc_\|Lateral_\|LaneOffset_"; then
        echo "[$(date)] ERROR: Old columns (LongAcc/Lateral/LaneOffset) still present! Aborting."
        exit 1
    else
        echo "[$(date)] OK: Only Steering_ + SteeringSpeed_ columns (36 features)."
    fi
    # Count feature columns (excluding Timestamp)
    N_COLS=$(head -1 "$SAMPLE" | tr ',' '\n' | grep -c -v "Timestamp")
    echo "[$(date)] Feature columns: $N_COLS (expected: 36)"
fi

if [[ $ACTUAL -eq 0 ]]; then
    echo "[$(date)] ERROR: No preprocessed files found! Aborting."
    exit 1
fi

echo "[$(date)] Starting SvmA training auto-resub daemon..."
exec bash "$PROJECT_ROOT/scripts/hpc/launchers/auto_resub_svma_arefnezhad2019.sh"

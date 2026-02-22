#!/bin/bash
# ============================================================
# Migrate queued CPU jobs from crowded queues to LARGE queue
# ============================================================
# LARGE queue: ncpus_min=128, max_run=3/user, max_queued=15/user
# This script:
#   1. Reads queued job parameters via qstat
#   2. Cancels the queued job
#   3. Resubmits to LARGE with ncpus=128
#
# Usage: bash migrate_to_large.sh [MAX_JOBS]
#   MAX_JOBS: max number of jobs to migrate (default: 15)
# ============================================================
set -euo pipefail

MAX_JOBS="${1:-15}"
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"

echo "============================================================"
echo "Migrate queued CPU jobs → LARGE queue (ncpus=128)"
echo "Max jobs to migrate: $MAX_JOBS"
echo "============================================================"

# Get queued CPU jobs (exclude GPU queues, exclude running)
mapfile -t QUEUED_JOBS < <(qstat -u s2240011 | tail -n +6 | grep " Q " | grep -v "GPU" | awk '{print $1}')

echo "Total queued CPU jobs found: ${#QUEUED_JOBS[@]}"

# Check current LARGE queue usage
LARGE_QUEUED=$(qstat -u s2240011 | tail -n +6 | awk '$3=="LARGE"' | wc -l)
LARGE_AVAIL=$((15 - LARGE_QUEUED))
echo "Current LARGE queue jobs (ours): $LARGE_QUEUED"
echo "Available LARGE queue slots: $LARGE_AVAIL"

if [[ $LARGE_AVAIL -le 0 ]]; then
    echo "[WARN] LARGE queue is full (15 queued). Cannot migrate."
    exit 0
fi

# Limit to available slots
if [[ $MAX_JOBS -gt $LARGE_AVAIL ]]; then
    MAX_JOBS=$LARGE_AVAIL
    echo "Adjusted MAX_JOBS to $MAX_JOBS (LARGE queue limit)"
fi

MIGRATED=0
FAILED=0

for jid in "${QUEUED_JOBS[@]}"; do
    if [[ $MIGRATED -ge $MAX_JOBS ]]; then
        echo "[INFO] Reached max migration limit ($MAX_JOBS). Stopping."
        break
    fi

    echo ""
    echo "--- Processing $jid ---"

    # Extract submit arguments
    SUBMIT_ARGS=$(qstat -xf "$jid" 2>/dev/null | sed -n '/Submit_arguments/,/project/p' | grep -v project | tr -d '\n' | sed 's/^[[:space:]]*Submit_arguments = //' | sed 's/[[:space:]]\+/ /g' || true)

    if [[ -z "$SUBMIT_ARGS" ]]; then
        echo "[SKIP] Could not extract submit arguments for $jid"
        ((FAILED++)) || true
        continue
    fi

    # Extract components
    JOB_NAME=$(echo "$SUBMIT_ARGS" | grep -oP '(?<=-N )\S+')
    WALLTIME=$(echo "$SUBMIT_ARGS" | grep -oP '(?<=walltime=)\S+')
    VARS=$(echo "$SUBMIT_ARGS" | grep -oP '(?<=-v )\S+.*?(?= /)' | sed 's/[[:space:]]*//g')
    PBS_SCRIPT=$(echo "$SUBMIT_ARGS" | grep -oP '/\S+\.sh$')
    MEM=$(echo "$SUBMIT_ARGS" | grep -oP '(?<=mem=)\d+gb')
    ORIG_QUEUE=$(echo "$SUBMIT_ARGS" | grep -oP '(?<=-q )\S+')

    if [[ -z "$JOB_NAME" || -z "$PBS_SCRIPT" || -z "$VARS" ]]; then
        echo "[SKIP] Failed to parse: $SUBMIT_ARGS"
        ((FAILED++)) || true
        continue
    fi

    # Set defaults if not found
    WALLTIME="${WALLTIME:-48:00:00}"
    MEM="${MEM:-32gb}"

    echo "  Name: $JOB_NAME"
    echo "  From: $ORIG_QUEUE → LARGE"
    echo "  Vars: $VARS"
    echo "  Script: $(basename $PBS_SCRIPT)"

    # Cancel the original job
    echo "  Canceling $jid..."
    if ! qdel "$jid" 2>/dev/null; then
        echo "[SKIP] Failed to cancel $jid (may have started running)"
        ((FAILED++)) || true
        continue
    fi
    sleep 0.5

    # Resubmit to LARGE with ncpus=128
    # Use walltime up to 168h (LARGE max), keep original or extend
    NEW_CMD="qsub -N ${JOB_NAME} -l select=1:ncpus=128:mem=${MEM} -l walltime=${WALLTIME} -q LARGE -v ${VARS} ${PBS_SCRIPT}"
    echo "  Submitting: $NEW_CMD"

    NEW_JID=$(eval "$NEW_CMD" 2>&1) || {
        echo "[ERROR] Failed to submit: $NEW_JID"
        ((FAILED++)) || true
        continue
    }

    echo "  → Submitted as $NEW_JID"
    ((MIGRATED++)) || true
done

echo ""
echo "============================================================"
echo "Migration complete: $MIGRATED migrated, $FAILED failed"
echo "Remaining queued CPU: $((${#QUEUED_JOBS[@]} - MIGRATED))"
echo "============================================================"

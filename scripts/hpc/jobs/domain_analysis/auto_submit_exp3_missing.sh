#!/bin/bash
# ============================================================
# Auto-submission daemon: submit unsubmitted Experiment 3 jobs as slots become available
# ============================================================
# Usage: nohup bash scripts/hpc/jobs/domain_analysis/auto_submit_exp3_missing.sh &
#
# Submits the following in order as queue slots open up:
#   1. SvmA 18 jobs (CPU: SINGLE/LONG/DEFAULT)
#   2. Lstm 38 jobs (GPU: GPU-1A)
#   3. Lstm eval-only 12 tasks (GPU: GPU-1, PBS array)

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

MAX_JOBS=148
CHECK_INTERVAL=300

LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/auto_submit_exp3_missing_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
get_job_count() { qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l; }

log "============================================================"
log "Experiment 3 auto-submission daemon started"
log "Max jobs: $MAX_JOBS / Check interval: ${CHECK_INTERVAL}s"
log "============================================================"

# Extract qsub commands from the generated scripts
QSUB_CMDS=()

# SvmA commands
while IFS= read -r line; do
    QSUB_CMDS+=("$line")
done < <(grep '^qsub ' scripts/hpc/jobs/domain_analysis/submit_missing_svma_exp3.sh)

SVMA_COUNT=${#QSUB_CMDS[@]}
log "SvmA command count: $SVMA_COUNT"

# Lstm train commands
while IFS= read -r line; do
    QSUB_CMDS+=("$line")
done < <(grep '^qsub ' scripts/hpc/jobs/domain_analysis/submit_missing_lstm_train_exp3.sh)

LSTM_TRAIN_COUNT=$((${#QSUB_CMDS[@]} - SVMA_COUNT))
log "Lstm train command count: $LSTM_TRAIN_COUNT"

TOTAL=${#QSUB_CMDS[@]}
log "Total submissions planned: $TOTAL"

IDX=0
SUBMITTED=0
FAILED=0

while [[ $IDX -lt $TOTAL ]]; do
    CURRENT=$(get_job_count)
    AVAIL=$((MAX_JOBS - CURRENT))

    if [[ $AVAIL -le 0 ]]; then
        log "Queue full: $CURRENT/$MAX_JOBS — rechecking in ${CHECK_INTERVAL}s (remaining: $((TOTAL - IDX)))"
        sleep "$CHECK_INTERVAL"
        continue
    fi

    # Submit as many as we can
    BATCH=0
    while [[ $IDX -lt $TOTAL ]] && [[ $AVAIL -gt 0 ]]; do
        CMD="${QSUB_CMDS[$IDX]}"
        JOB_NAME=$(echo "$CMD" | grep -oP '(?<=-N )\S+')

        JOB_ID=$(eval "$CMD" 2>&1)
        RC=$?
        if [[ $RC -eq 0 ]]; then
            log "  [OK] $JOB_NAME → $JOB_ID"
            ((SUBMITTED++))
            ((AVAIL--))
            ((BATCH++))
        else
            log "  [FAIL] $JOB_NAME: $JOB_ID"
            ((FAILED++))
            # If quota error, stop this batch and wait
            if echo "$JOB_ID" | grep -q "exceed"; then
                log "  Queue limit reached — waiting"
                break
            fi
        fi
        ((IDX++))
        sleep 0.3
    done

    log "Submitted: $SUBMITTED/$TOTAL, Failed: $FAILED, Batch: $BATCH"

    if [[ $IDX -lt $TOTAL ]]; then
        sleep "$CHECK_INTERVAL"
    fi
done

# Submit Lstm eval-only PBS array
log ""
log "=== Lstm eval-only PBS array submission ==="
EVAL_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_eval_missing_lstm_exp3.sh"
if [[ -f "$EVAL_SCRIPT" ]]; then
    while true; do
        CURRENT=$(get_job_count)
        AVAIL=$((MAX_JOBS - CURRENT))
        if [[ $AVAIL -gt 0 ]]; then
            JOB_ID=$(qsub "$EVAL_SCRIPT" 2>&1)
            RC=$?
            if [[ $RC -eq 0 ]]; then
                log "  [OK] Lstm eval array → $JOB_ID"
            else
                log "  [FAIL] $JOB_ID"
            fi
            break
        fi
        log "  Queue full — rechecking in ${CHECK_INTERVAL}s"
        sleep "$CHECK_INTERVAL"
    done
else
    log "  [SKIP] eval script not found: $EVAL_SCRIPT"
fi

log ""
log "============================================================"
log "Done: Submitted $SUBMITTED, Failed $FAILED"
log "============================================================"

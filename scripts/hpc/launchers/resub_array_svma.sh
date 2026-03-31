#!/bin/bash
# ============================================================
# Array Job Resubmitter: SvmA domain_train + pooled
#
# Submits domain_train tasks first, then pooled tasks, using
# all available queue slots across SINGLE/DEFAULT/SMALL/LONG.
#
# Usage:
#   nohup bash scripts/hpc/launchers/resub_array_svma.sh \
#     > scripts/hpc/logs/train/resub_array_svma_output.log 2>&1 &
# ============================================================
set -o pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"

# Domain-train config
DT_TASK_FILE="$PROJECT_ROOT/scripts/hpc/logs/train/task_files/array_svma_domain_train.txt"
DT_PBS_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_array_svma.sh"

# Pooled config
PL_TASK_FILE="$PROJECT_ROOT/scripts/hpc/logs/train/task_files/array_svma_pooled.txt"
PL_PBS_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_array_svma_pooled.sh"

POLL_INTERVAL=180  # seconds (SvmA jobs are long, no need to poll frequently)
BATCH_SIZE=10
QUEUES=(SINGLE DEFAULT SMALL LONG)

LOG_FILE="$PROJECT_ROOT/scripts/hpc/logs/train/resub_svma_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

submit_batch() {
    local task_file="$1"
    local pbs_script="$2"
    local next_ref="$3"  # variable name for next index
    local total="$4"
    local max_idx="$5"
    local label="$6"

    eval "local next_idx=\$$next_ref"
    local submitted_any=false

    for q in "${QUEUES[@]}"; do
        [[ $next_idx -gt $max_idx ]] && break

        local end_idx=$((next_idx + BATCH_SIZE - 1))
        [[ $end_idx -gt $max_idx ]] && end_idx=$max_idx

        if [[ $end_idx -le $next_idx ]]; then
            # Single element
            RESULT=$(qsub -q "$q" \
                -v "TASK_FILE=$task_file,PBS_ARRAY_INDEX=$next_idx" \
                "$pbs_script" 2>&1)
            if [[ $? -eq 0 ]]; then
                log "[$label] index $next_idx → $q → $RESULT"
                next_idx=$((next_idx + 1))
                submitted_any=true
                sleep 0.5
            else
                log "[$label] SKIP $next_idx → $q: $RESULT"
            fi
        else
            RESULT=$(qsub -J "${next_idx}-${end_idx}" -q "$q" \
                -v "TASK_FILE=$task_file" \
                "$pbs_script" 2>&1)
            if [[ $? -eq 0 ]]; then
                local count=$((end_idx - next_idx + 1))
                log "[$label] indices ${next_idx}-${end_idx} ($count tasks) → $q → $RESULT"
                next_idx=$((end_idx + 1))
                submitted_any=true
                sleep 0.5
            else
                log "[$label] SKIP ${next_idx}-${end_idx} → $q"
            fi
        fi
    done

    eval "$next_ref=$next_idx"
    $submitted_any
}

# --- PHASE 1: Domain-train ---
DT_TOTAL=$(wc -l < "$DT_TASK_FILE")
DT_MAX=$((DT_TOTAL - 1))
DT_NEXT=0

log "============================================================"
log "SvmA Resubmitter"
log "  Domain-train: $DT_TOTAL tasks (indices 0-$DT_MAX)"
if [[ -f "$PL_TASK_FILE" ]]; then
    PL_TOTAL=$(wc -l < "$PL_TASK_FILE")
    PL_MAX=$((PL_TOTAL - 1))
    log "  Pooled:       $PL_TOTAL tasks (indices 0-$PL_MAX)"
else
    PL_TOTAL=0
    PL_MAX=-1
fi
log "  Batch size:   $BATCH_SIZE per qsub"
log "  Poll:         ${POLL_INTERVAL}s"
log "============================================================"

# Submit domain-train tasks
while [[ $DT_NEXT -le $DT_MAX ]]; do
    if submit_batch "$DT_TASK_FILE" "$DT_PBS_SCRIPT" DT_NEXT "$DT_TOTAL" "$DT_MAX" "DT"; then
        : # submitted something
    else
        log "[DT] All queues full. Next: $DT_NEXT/$DT_TOTAL. Waiting ${POLL_INTERVAL}s..."
    fi

    [[ $DT_NEXT -gt $DT_MAX ]] && break
    sleep "$POLL_INTERVAL"
done

log "[DT] All $DT_TOTAL domain-train tasks submitted!"

# --- PHASE 2: Pooled ---
PL_NEXT=0

if [[ $PL_TOTAL -gt 0 ]]; then
    log ""
    log "--- Starting pooled submissions ---"

    while [[ $PL_NEXT -le $PL_MAX ]]; do
        if submit_batch "$PL_TASK_FILE" "$PL_PBS_SCRIPT" PL_NEXT "$PL_TOTAL" "$PL_MAX" "PL"; then
            :
        else
            log "[PL] All queues full. Next: $PL_NEXT/$PL_TOTAL. Waiting ${POLL_INTERVAL}s..."
        fi

        [[ $PL_NEXT -gt $PL_MAX ]] && break
        sleep "$POLL_INTERVAL"
    done

    log "[PL] All $PL_TOTAL pooled tasks submitted!"
fi

log "============================================================"
log "SvmA resubmitter finished."
log "  Domain-train: $DT_TOTAL tasks"
log "  Pooled:       $PL_TOTAL tasks"
log "  Total:        $((DT_TOTAL + PL_TOTAL)) tasks"
log "============================================================"

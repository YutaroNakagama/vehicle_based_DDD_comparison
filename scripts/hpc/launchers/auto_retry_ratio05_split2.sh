#!/bin/bash
# ============================================================
# Auto-retry launcher for 13 missing ratio=0.5 RF jobs
# ============================================================
# Waits for queue slots to open, then submits jobs one at a time.
# Re-checks every 60 seconds until all 13 jobs are submitted.
# ============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
PBS_SCRIPT="${PROJECT_ROOT}/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain"
LOGFILE="${LOG_DIR}/auto_retry_ratio05_$(date +%Y%m%d_%H%M%S).log"

RANKING="knn"
N_TRIALS=100
RATIO="0.5"
WALLTIME="06:00:00"
MEM="10gb"
NCPUS=4

# Define all 13 configs: COND|MODE|DIST|DOM|SEED
CONFIGS=(
    "smote_plain|mixed|dtw|in_domain|123"
    "smote_plain|mixed|mmd|out_domain|123"
    "smote_plain|mixed|wasserstein|in_domain|123"
    "smote_plain|mixed|wasserstein|out_domain|123"
    "smote|mixed|dtw|in_domain|123"
    "smote|mixed|dtw|out_domain|123"
    "smote|mixed|mmd|out_domain|123"
    "smote|mixed|wasserstein|in_domain|42"
    "smote|mixed|wasserstein|in_domain|123"
    "smote|mixed|wasserstein|out_domain|42"
    "smote|mixed|wasserstein|out_domain|123"
    "undersample|mixed|mmd|in_domain|123"
    "undersample|mixed|mmd|out_domain|42"
)

TOTAL=${#CONFIGS[@]}
SUBMITTED=0
IDX=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"
}

get_queue_count() {
    qstat -u "$USER" 2>/dev/null | grep -c " [QR] " || echo "0"
}

# Try queues in order: SINGLE → SMALL → DEFAULT → LONG
try_submit() {
    local COND="$1" MODE="$2" DIST="$3" DOM="$4" SEED="$5"
    local JOBNAME="rf_r05_${COND}_${MODE}_${DIST}_${DOM}_s${SEED}"

    for QUEUE in SINGLE SMALL DEFAULT LONG; do
        local CMD="qsub -N ${JOBNAME}"
        CMD="$CMD -l select=1:ncpus=${NCPUS}:mem=${MEM}"
        CMD="$CMD -l walltime=${WALLTIME}"
        CMD="$CMD -q ${QUEUE}"
        CMD="$CMD -v CONDITION=${COND},MODE=${MODE},DISTANCE=${DIST},DOMAIN=${DOM},RATIO=${RATIO},SEED=${SEED},N_TRIALS=${N_TRIALS},RANKING=${RANKING},RUN_EVAL=true"
        CMD="$CMD ${PBS_SCRIPT}"

        local result
        if result=$(eval "$CMD" 2>&1); then
            log "  OK [${QUEUE}]: ${JOBNAME} → ${result}"
            return 0
        fi
    done
    return 1
}

log "============================================================"
log "AUTO-RETRY: 13 missing ratio=0.5 RF configs"
log "============================================================"

while [[ $IDX -lt $TOTAL ]]; do
    IFS='|' read -r COND MODE DIST DOM SEED <<< "${CONFIGS[$IDX]}"

    QCOUNT=$(get_queue_count)
    log "Queue: ${QCOUNT} jobs | Submitting ${IDX}/${TOTAL}: ${COND}/${DIST}/${DOM}/s${SEED}"

    if try_submit "$COND" "$MODE" "$DIST" "$DOM" "$SEED"; then
        SUBMITTED=$((SUBMITTED + 1))
        IDX=$((IDX + 1))
        sleep 2
    else
        log "  All queues full. Waiting 60s..."
        sleep 60
    fi
done

log "============================================================"
log "DONE: ${SUBMITTED}/${TOTAL} submitted"
log "============================================================"

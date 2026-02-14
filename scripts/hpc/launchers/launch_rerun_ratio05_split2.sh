#!/bin/bash
# ============================================================
# Launch script: Rerun 13 missing ratio=0.5 RF jobs (split2)
# ============================================================
# These configs had bogus ratio=0.5 eval results (copied from ratio=0.1
# jobs during a batch re-evaluation). The correct ratio=0.5 training
# was never performed for these 13 configurations.
#
# Conditions:
#   smote_plain   (CONDITION=smote_plain)  : 4 jobs
#   sw_smote      (CONDITION=smote)        : 7 jobs
#   undersample   (CONDITION=undersample)  : 2 jobs
#
# All configs: MODE=mixed, RATIO=0.5
# ============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
PBS_SCRIPT="${PROJECT_ROOT}/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain"
LOGFILE="${LOG_DIR}/launch_rerun_ratio05_$(date +%Y%m%d_%H%M%S).log"

RANKING="knn"
N_TRIALS=100
RATIO="0.5"

# Resource settings (RF mixed jobs: ~1GB actual, 6h walltime)
WALLTIME="06:00:00"
MEM="10gb"
NCPUS=4
QUEUE="SINGLE"

SUBMITTED=0
FAILED=0

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOGFILE"
}

submit_job() {
    local COND="$1" MODE="$2" DIST="$3" DOM="$4" SEED="$5"
    local JOBNAME="rf_r05_${COND}_${MODE}_${DIST}_${DOM}_s${SEED}"

    local CMD="qsub -N ${JOBNAME}"
    CMD="$CMD -l select=1:ncpus=${NCPUS}:mem=${MEM}"
    CMD="$CMD -l walltime=${WALLTIME}"
    CMD="$CMD -q ${QUEUE}"
    CMD="$CMD -v CONDITION=${COND},MODE=${MODE},DISTANCE=${DIST},DOMAIN=${DOM},RATIO=${RATIO},SEED=${SEED},N_TRIALS=${N_TRIALS},RANKING=${RANKING},RUN_EVAL=true"
    CMD="$CMD ${PBS_SCRIPT}"

    log "SUBMIT: $JOBNAME"
    local result
    if result=$(eval "$CMD" 2>&1); then
        log "  OK: $result"
        SUBMITTED=$((SUBMITTED + 1))
    else
        log "  FAILED: $result"
        FAILED=$((FAILED + 1))
    fi
    sleep 1
}

log "============================================================"
log "RERUN: 13 missing ratio=0.5 RF configs"
log "============================================================"

# --- smote_plain (4 jobs) ---
submit_job smote_plain mixed dtw          in_domain  123
submit_job smote_plain mixed mmd          out_domain 123
submit_job smote_plain mixed wasserstein  in_domain  123
submit_job smote_plain mixed wasserstein  out_domain 123

# --- sw_smote / CONDITION=smote (7 jobs) ---
submit_job smote       mixed dtw          in_domain  123
submit_job smote       mixed dtw          out_domain 123
submit_job smote       mixed mmd          out_domain 123
submit_job smote       mixed wasserstein  in_domain  42
submit_job smote       mixed wasserstein  in_domain  123
submit_job smote       mixed wasserstein  out_domain 42
submit_job smote       mixed wasserstein  out_domain 123

# --- undersample (2 jobs) ---
submit_job undersample mixed mmd          in_domain  123
submit_job undersample mixed mmd          out_domain 42

log "============================================================"
log "SUMMARY: ${SUBMITTED} submitted, ${FAILED} failed (total 13)"
log "============================================================"

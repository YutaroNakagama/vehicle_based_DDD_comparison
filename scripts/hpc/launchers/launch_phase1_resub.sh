#!/bin/bash
# ============================================================
# Phase 1 Training Resubmission: 12 failed configs
# Queue: DEFAULT (SEMINAR disabled)
# Reason: Original jobs killed by SIGTERM (system event 2026-02-12 01:45)
# ============================================================
set -euo pipefail

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison

QUEUE="DEFAULT"
NCPUS=4
MEM="8gb"
WALLTIME="04:00:00"
SCRIPT="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

COUNT=0

submit() {
    local COND="$1" DIST="$2" DOM="$3" SEED="$4"
    
    # Short name for PBS (max 15 chars)
    local SHORT_COND="${COND:0:2}"
    local SHORT_DIST="${DIST:0:2}"
    local SHORT_DOM="${DOM:0:1}"
    local NAME="ft_${SHORT_COND}_${SHORT_DIST}${SHORT_DOM}_s${SEED}"
    
    echo "  Submitting: $COND $DIST $DOM s$SEED -> $NAME"
    
    qsub -N "$NAME" \
        -l select=1:ncpus=${NCPUS}:mem=${MEM} \
        -l walltime=${WALLTIME} \
        -q ${QUEUE} \
        -v CONDITION=${COND},MODE=target_only,DISTANCE=${DIST},DOMAIN=${DOM},SEED=${SEED},RUN_EVAL=true \
        "$SCRIPT"
    
    COUNT=$((COUNT + 1))
    sleep 0.5
}

echo "============================================================"
echo "  Phase 1 Training Resubmission (12 configs)"
echo "  Queue: $QUEUE  Resources: ncpus=$NCPUS mem=$MEM walltime=$WALLTIME"
echo "============================================================"

# baseline × mmd × 4
submit baseline mmd in_domain 42
submit baseline mmd in_domain 123
submit baseline mmd out_domain 42
submit baseline mmd out_domain 123

# baseline × dtw × 2
submit baseline dtw in_domain 42
submit baseline dtw out_domain 42

# baseline × wasserstein × 4
submit baseline wasserstein in_domain 42
submit baseline wasserstein in_domain 123
submit baseline wasserstein out_domain 42
submit baseline wasserstein out_domain 123

# balanced_rf × wasserstein × 2
submit balanced_rf wasserstein in_domain 42
submit balanced_rf wasserstein in_domain 123

echo ""
echo "============================================================"
echo "  Submitted $COUNT jobs to $QUEUE queue"
echo "============================================================"

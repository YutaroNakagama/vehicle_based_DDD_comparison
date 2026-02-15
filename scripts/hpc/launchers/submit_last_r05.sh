#!/bin/bash
# Wait for a slot to free up, then submit the last ratio=0.5 job
# Config: sw_smote / mixed / wasserstein / out_domain / s42
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
PBS_SCRIPT="${PROJECT_ROOT}/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

MAX_WAIT=7200  # 2 hours
INTERVAL=60
ELAPSED=0

echo "[$(date)] Waiting for job slot to free up..."
while [ $ELAPSED -lt $MAX_WAIT ]; do
    NJOBS=$(qstat -u s2240011 | awk 'END{print NR-5}')
    if [ "$NJOBS" -lt 125 ]; then
        echo "[$(date)] Slot available ($NJOBS/125). Submitting..."
        JID=$(qsub -N rf_r05_sm_wass_od_s42 \
            -l select=1:ncpus=4:mem=10gb \
            -l walltime=10:00:00 \
            -q DEFAULT \
            -v CONDITION=smote,MODE=mixed,DISTANCE=wasserstein,DOMAIN=out_domain,RATIO=0.5,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true \
            "$PBS_SCRIPT" 2>&1)
        echo "[$(date)] Submitted: $JID"
        exit 0
    fi
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "[$(date)] Timed out after ${MAX_WAIT}s. Submit manually."
exit 1

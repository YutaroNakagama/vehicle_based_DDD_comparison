#!/bin/bash
# ============================================================
# Submit remaining job after queue space opens up
# ============================================================
# Run this after some jobs complete to submit the last missing job
#
# Missing: swdous_s123 (smote|source_only|dtw|out_domain|0.5|123)
# ============================================================

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison.sh"

echo "Attempting to submit swdous_s123..."

for queue in SINGLE SMALL LONG DEFAULT; do
    result=$(qsub -N "swdous_s123" -l select=1:ncpus=4:mem=8gb -l walltime=12:00:00 -q "$queue" \
        -v CONDITION=smote,MODE=source_only,DISTANCE=dtw,DOMAIN=out_domain,RATIO=0.5,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true \
        "$JOB_SCRIPT" 2>&1)
    
    if [[ "$result" != *"would exceed"* ]]; then
        echo "SUCCESS: swdous_s123 -> $queue: $result"
        exit 0
    fi
done

echo "FAILED: All queues at limit. Try again later."
exit 1

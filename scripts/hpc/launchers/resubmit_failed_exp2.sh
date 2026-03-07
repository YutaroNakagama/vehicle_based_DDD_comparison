#!/bin/bash
# Resubmit 8 failed SMOTE walltime-exceeded jobs with 20h walltime
# These jobs hit the 10h walltime limit and need longer runtime.
# This script retries periodically until all 8 are submitted.
# Strategy: round-robin across SINGLE/DEFAULT/SMALL/LONG queues

set -u

PBS_SCRIPT="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
LOGFILE="scripts/hpc/launchers/resubmit_failed_exp2.log"
SLEEP_INTERVAL=60  # seconds between retry cycles (shorter than daemon's 120s to grab slots first)

# Queue limits (max_queued per user)
declare -A QUEUE_MAX=( [SINGLE]=40 [DEFAULT]=40 [SMALL]=30 [LONG]=15 )
QUEUES=(SINGLE DEFAULT SMALL LONG)

# Define the 8 failed jobs as: NAME CONDITION MODE DISTANCE DOMAIN RATIO SEED
JOBS=(
  "e2_smr5_mmmo_s7    smote_plain mixed mmd out_domain 0.5 7"
  "e2_swsr5_mmmo_s7   smote       mixed mmd out_domain 0.5 7"
  "e2_smr5_mmmo_s2024 smote_plain mixed mmd out_domain 0.5 2024"
  "e2_smr5_mmmo_s1337 smote_plain mixed mmd out_domain 0.5 1337"
  "e2_smr5_mmmi_s0    smote_plain mixed mmd in_domain  0.5 0"
  "e2_smr5_mmmi_s7    smote_plain mixed mmd in_domain  0.5 7"
  "e2_smr5_mmmi_s2024 smote_plain mixed mmd in_domain  0.5 2024"
  "e2_smr5_mdto_s2024 smote_plain mixed dtw out_domain 0.5 2024"
)

submitted=()
queue_idx=0

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOGFILE"
}

get_queue_count() {
    local q="$1"
    qstat -u "$USER" 2>/dev/null | grep -c "$q" || echo 0
}

log "=== Starting resubmission of ${#JOBS[@]} failed Exp2 jobs ==="

while [ ${#submitted[@]} -lt ${#JOBS[@]} ]; do
    for i in "${!JOBS[@]}"; do
        # Skip already submitted
        skip=false
        for s in "${submitted[@]+"${submitted[@]}"}"; do
            if [ "$s" = "$i" ]; then
                skip=true
                break
            fi
        done
        $skip && continue

        # Parse job params
        read -r NAME COND MODE DIST DOM RATIO SEED <<< "${JOBS[$i]}"

        # Try each queue in round-robin
        attempts=0
        while [ $attempts -lt ${#QUEUES[@]} ]; do
            q="${QUEUES[$queue_idx]}"
            queue_idx=$(( (queue_idx + 1) % ${#QUEUES[@]} ))
            
            count=$(get_queue_count "$q")
            max=${QUEUE_MAX[$q]}
            
            if [ "$count" -lt "$max" ]; then
                # Submit
                MEM="10gb"
                WALLTIME="20:00:00"
                
                result=$(qsub -N "$NAME" \
                    -l "select=1:ncpus=4:mem=${MEM}" \
                    -l "walltime=${WALLTIME}" \
                    -q "$q" \
                    -v "CONDITION=${COND},MODE=${MODE},DISTANCE=${DIST},DOMAIN=${DOM},RATIO=${RATIO},SEED=${SEED},N_TRIALS=100,RANKING=knn,RUN_EVAL=true" \
                    "$PBS_SCRIPT" 2>&1)
                
                if [ $? -eq 0 ]; then
                    log "OK: Submitted job $i ($NAME) -> $q : $result"
                    submitted+=("$i")
                    break
                else
                    log "WARN: qsub failed for $NAME on $q: $result"
                fi
            fi
            attempts=$((attempts + 1))
        done
    done

    if [ ${#submitted[@]} -lt ${#JOBS[@]} ]; then
        remaining=$(( ${#JOBS[@]} - ${#submitted[@]} ))
        log "Waiting... ${#submitted[@]}/${#JOBS[@]} submitted ($remaining remaining). Sleeping ${SLEEP_INTERVAL}s..."
        sleep $SLEEP_INTERVAL
    fi
done

log "=== All ${#JOBS[@]} failed jobs resubmitted successfully ==="

#!/bin/bash
# Round 2: Resubmit 12 walltime-exceeded SMOTE/sw_smote mixed mode jobs
# with 48h walltime (previous 10h was insufficient)
set -u

PBS_SCRIPT="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
LOGFILE="scripts/hpc/launchers/resubmit_failed_exp2_round2.log"
SLEEP_INTERVAL=60

declare -A QUEUE_MAX=( [SINGLE]=40 [DEFAULT]=40 [SMALL]=30 [LONG]=15 )
QUEUES=(SINGLE DEFAULT SMALL LONG)

# 12 uncovered walltime-exceeded jobs: NAME CONDITION MODE DISTANCE DOMAIN RATIO SEED
JOBS=(
  "e2_swr5_mdtwi_s2024  smote       mixed dtw         in_domain  0.5 2024"
  "e2_swr5_mwaso_s1     smote       mixed wasserstein out_domain 0.5 1"
  "e2_swr5_mwaso_s7     smote       mixed wasserstein out_domain 0.5 7"
  "e2_smr1_mdtwi_s7     smote_plain mixed dtw         in_domain  0.1 7"
  "e2_smr5_mdtwi_s0     smote_plain mixed dtw         in_domain  0.5 0"
  "e2_smr5_mdtwi_s2024  smote_plain mixed dtw         in_domain  0.5 2024"
  "e2_smr5_mdtwi_s7     smote_plain mixed dtw         in_domain  0.5 7"
  "e2_smr5_mdtwo_s0     smote_plain mixed dtw         out_domain 0.5 0"
  "e2_smr5_mdtwo_s1337  smote_plain mixed dtw         out_domain 0.5 1337"
  "e2_smr5_mdtwo_s7     smote_plain mixed dtw         out_domain 0.5 7"
  "e2_smr5_mwaso_s2024  smote_plain mixed wasserstein out_domain 0.5 2024"
  "e2_smr5_mwaso_s7     smote_plain mixed wasserstein out_domain 0.5 7"
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

log "=== Starting round 2 resubmission of ${#JOBS[@]} failed Exp2 jobs (48h walltime) ==="

while [ ${#submitted[@]} -lt ${#JOBS[@]} ]; do
    for i in "${!JOBS[@]}"; do
        skip=false
        for s in "${submitted[@]+"${submitted[@]}"}"; do
            if [ "$s" = "$i" ]; then
                skip=true
                break
            fi
        done
        $skip && continue

        read -r NAME COND MODE DIST DOM RATIO SEED <<< "${JOBS[$i]}"

        attempts=0
        while [ $attempts -lt ${#QUEUES[@]} ]; do
            q="${QUEUES[$queue_idx]}"
            queue_idx=$(( (queue_idx + 1) % ${#QUEUES[@]} ))
            
            count=$(get_queue_count "$q")
            max=${QUEUE_MAX[$q]}
            
            if [ "$count" -lt "$max" ]; then
                result=$(qsub -N "$NAME" \
                    -l "select=1:ncpus=4:mem=10gb" \
                    -l "walltime=48:00:00" \
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

log "=== All ${#JOBS[@]} round 2 jobs resubmitted successfully ==="

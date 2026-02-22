#!/bin/bash
# ============================================================
# Batch submit all pending domain_train jobs across 3 queues
# Distributes round-robin: DEFAULT → SINGLE → SMALL
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
DOMAIN_TRAIN_JOB="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
N_TRIALS=100
RANKING="knn"

PENDING_FILE="/tmp/pending_dt_all.txt"
if [[ ! -f "$PENDING_FILE" ]]; then
    echo "ERROR: $PENDING_FILE not found" >&2
    exit 1
fi

LOG="$PROJECT_ROOT/scripts/hpc/logs/train/batch_submit_3q_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Queue rotation: DEFAULT(max_queued=40), SINGLE(40), SMALL(30)
QUEUES=("DEFAULT" "SINGLE" "SMALL")
Q_IDX=0

SUBMITTED=0
FAILED=0
TOTAL=$(wc -l < "$PENDING_FILE")

log "Starting batch submit: $TOTAL jobs across queues: ${QUEUES[*]}"

# Read all lines into array first (avoid sed-while-read issues)
mapfile -t JOBS < "$PENDING_FILE"

for LINE in "${JOBS[@]}"; do
    IFS=: read -r MODEL CONDITION DISTANCE DOMAIN MODE RATIO SEED <<< "$LINE"

    # Resource specs per model
    case "$MODEL" in
        SvmA) NCPUS_MEM="ncpus=8:mem=32gb"; WALLTIME="48:00:00" ;;
        Lstm) NCPUS_MEM="ncpus=8:mem=32gb"; WALLTIME="20:00:00" ;;
        *)    NCPUS_MEM="ncpus=8:mem=32gb"; WALLTIME="48:00:00" ;;
    esac

    # Try queues in rotation, fallback to next if one fails
    SUBMITTED_THIS=false
    for attempt in 0 1 2; do
        QUEUE="${QUEUES[$(( (Q_IDX + attempt) % 3 ))]}"

        # Build job name
        local_model_short="${MODEL:0:2}"
        case "$CONDITION" in
            baseline)     local_cond_short="bs" ;;
            smote_plain)  local_cond_short="sp" ;;
            smote)        local_cond_short="sm" ;;
            undersample)  local_cond_short="un" ;;
        esac

        if [[ "$CONDITION" == "baseline" || "$RATIO" == "none" ]]; then
            JOB_NAME="${local_model_short}_${local_cond_short}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_s${SEED}"
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
            CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        else
            JOB_NAME="${local_model_short}_${local_cond_short}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_r${RATIO}_s${SEED}"
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
            CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true,RATIO=$RATIO"
        fi
        CMD="$CMD $DOMAIN_TRAIN_JOB"

        JOB_ID=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            log "[OK] #$((SUBMITTED+1))/$TOTAL $MODEL:$CONDITION:$DISTANCE:$DOMAIN:r$RATIO:s$SEED → $JOB_ID ($QUEUE)"
            ((SUBMITTED++))
            Q_IDX=$(( (Q_IDX + 1) % 3 ))  # rotate for next job
            SUBMITTED_THIS=true
            break
        else
            log "[RETRY] $QUEUE failed: $JOB_ID — trying next queue"
        fi
    done

    if [[ "$SUBMITTED_THIS" == false ]]; then
        log "[FAIL] ALL queues failed for $MODEL:$CONDITION:$DISTANCE:$DOMAIN:r$RATIO:s$SEED"
        ((FAILED++))
    fi

    sleep 0.1
done

log "=== DONE === Submitted: $SUBMITTED, Failed: $FAILED, Total: $TOTAL"

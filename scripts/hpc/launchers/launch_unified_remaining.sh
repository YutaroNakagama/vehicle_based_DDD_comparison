#!/bin/bash
# ============================================================
# Remaining unified jobs submission (SvmW remaining + SvmA + Lstm)
# ============================================================
# Distributes across DEFAULT, LONG, SMALL queues
# (SINGLE already has 40 SvmW jobs)
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")

# Already submitted SvmW combos (from first launch log)
SUBMITTED_LOG="$PROJECT_ROOT/scripts/hpc/logs/train/launch_prior_research_unified_20260221_111054.log"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
LOG_FILE="$LOG_DIR/launch_unified_remaining_${TIMESTAMP}.log"

echo "# Remaining launch at $(date)" > "$LOG_FILE"

JOB_COUNT=0
SKIP_COUNT=0

# Queue counter for spreading across DEFAULT, LONG, SMALL
QUEUE_IDX=0

get_queue() {
    local queues=("DEFAULT" "LONG" "DEFAULT" "SMALL" "DEFAULT" "LONG")
    local q="${queues[$((QUEUE_IDX % ${#queues[@]}))]}"
    ((QUEUE_IDX++))
    echo "$q"
}

get_walltime() {
    local model="$1"
    case "$model" in
        SvmA) echo "48:00:00" ;;
        SvmW) echo "12:00:00" ;;
        Lstm) echo "16:00:00" ;;
    esac
}

get_resources() {
    local model="$1"
    case "$model" in
        SvmA) echo "ncpus=8:mem=32gb" ;;
        SvmW) echo "ncpus=8:mem=16gb" ;;
        Lstm) echo "ncpus=8:mem=32gb" ;;
    esac
}

submit_job() {
    local MODEL="$1" CONDITION="$2" DISTANCE="$3" DOMAIN="$4" SEED="$5" RATIO="$6"
    
    # Check if already submitted
    local check_key="${MODEL}:${CONDITION}:${DISTANCE}:${DOMAIN}:domain_train"
    if [[ -n "$RATIO" ]]; then
        check_key="${check_key}:${RATIO}:${SEED}"
    else
        check_key="${check_key}:${SEED}"
    fi
    
    if grep -q "$check_key" "$SUBMITTED_LOG" 2>/dev/null; then
        return 0
    fi
    
    local QUEUE=$(get_queue)
    local WALLTIME=$(get_walltime "$MODEL")
    local RES=$(get_resources "$MODEL")
    
    local COND_SHORT="${CONDITION:0:2}"
    local JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_s${SEED}"
    if [[ -n "$RATIO" ]]; then
        JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_r${RATIO}_s${SEED}"
    fi
    
    local CMD="qsub -N $JOB_NAME -l select=1:$RES -l walltime=$WALLTIME -q $QUEUE"
    if [[ -n "$RATIO" ]]; then
        CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    else
        CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    fi
    CMD="$CMD $JOB_SCRIPT"
    
    JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] $CMD"; ((SKIP_COUNT++)); return 1; }
    echo "[SUBMIT] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | r=${RATIO:-n/a} | s$SEED | $QUEUE → $JOB_ID"
    echo "${check_key}:${JOB_ID}" >> "$LOG_FILE"
    ((JOB_COUNT++))
    sleep 0.2
}

echo "============================================================"
echo "Remaining unified jobs (skipping already submitted)"
echo "============================================================"

for MODEL in "SvmW" "SvmA" "Lstm"; do
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # Baseline
                submit_job "$MODEL" "baseline" "$DISTANCE" "$DOMAIN" "$SEED" ""
                
                # Ratio-based
                for RATIO in "${RATIOS[@]}"; do
                    for COND in "smote_plain" "smote" "undersample"; do
                        submit_job "$MODEL" "$COND" "$DISTANCE" "$DOMAIN" "$SEED" "$RATIO"
                    done
                done
            done
        done
    done
done

echo ""
echo "============================================================"
echo "Submitted: $JOB_COUNT | Skipped (already done or error): $SKIP_COUNT"
echo "Log: $LOG_FILE"
echo "============================================================"

#!/bin/bash
# ============================================================
# Auto-resub for unified domain_train jobs
# Polls every 5 minutes, fills available queue slots
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
cd "$PROJECT_ROOT"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODELS=("SvmW" "SvmA" "Lstm")

# Track submitted jobs across runs
SUBMITTED_FILE="/tmp/unified_submitted_keys.txt"
touch "$SUBMITTED_FILE"

# Import already submitted from launch logs
for logf in "$PROJECT_ROOT"/scripts/hpc/logs/train/launch_prior_research_unified_*.log \
            "$PROJECT_ROOT"/scripts/hpc/logs/train/launch_unified_remaining_*.log; do
    [[ -f "$logf" ]] || continue
    grep -E "^(SvmW|SvmA|Lstm)" "$logf" | while IFS=: read -r model cond dist dom mode rest; do
        # Build a key: MODEL:CONDITION:DISTANCE:DOMAIN:SEED[:RATIO]
        # Log format varies, extract seed and optional ratio
        echo "${model}:${cond}:${dist}:${dom}:${rest}" >> "$SUBMITTED_FILE"
    done
done
sort -u "$SUBMITTED_FILE" -o "$SUBMITTED_FILE"
echo "[INIT] $(wc -l < "$SUBMITTED_FILE") jobs already tracked"

POLL_INTERVAL=300  # 5 minutes
MAX_TOTAL=125      # Max total jobs we want in queue
TARGET_TOTAL=252

get_walltime() {
    case "$1" in
        SvmA) echo "48:00:00" ;;
        SvmW) echo "12:00:00" ;;
        Lstm) echo "16:00:00" ;;
    esac
}

get_resources() {
    case "$1" in
        SvmA) echo "ncpus=8:mem=32gb" ;;
        SvmW) echo "ncpus=8:mem=16gb" ;;
        Lstm) echo "ncpus=8:mem=32gb" ;;
    esac
}

QUEUE_IDX=0
get_queue() {
    local queues=("DEFAULT" "SINGLE" "LONG" "DEFAULT" "SMALL" "SINGLE")
    local q="${queues[$((QUEUE_IDX % ${#queues[@]}))]}"
    ((QUEUE_IDX++))
    echo "$q"
}

build_key() {
    local MODEL="$1" CONDITION="$2" DISTANCE="$3" DOMAIN="$4" SEED="$5" RATIO="$6"
    if [[ -n "$RATIO" ]]; then
        echo "${MODEL}:${CONDITION}:${DISTANCE}:${DOMAIN}:${RATIO}:${SEED}"
    else
        echo "${MODEL}:${CONDITION}:${DISTANCE}:${DOMAIN}:${SEED}"
    fi
}

# Generate all 252 job keys
ALL_KEYS=()
for MODEL in "${MODELS[@]}"; do
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                ALL_KEYS+=("$(build_key "$MODEL" "baseline" "$DISTANCE" "$DOMAIN" "$SEED" "")")
                for RATIO in "${RATIOS[@]}"; do
                    for COND in "smote_plain" "smote" "undersample"; do
                        ALL_KEYS+=("$(build_key "$MODEL" "$COND" "$DISTANCE" "$DOMAIN" "$SEED" "$RATIO")")
                    done
                done
            done
        done
    done
done
echo "[INIT] Total keys: ${#ALL_KEYS[@]}"

while true; do
    # Count current jobs
    CURRENT=$(qstat -u s2240011 2>/dev/null | awk 'NR>5' | wc -l)
    SUBMITTED_COUNT=$(wc -l < "$SUBMITTED_FILE")
    REMAINING=$((TARGET_TOTAL - SUBMITTED_COUNT))
    
    if [[ $REMAINING -le 0 ]]; then
        echo "[DONE] All $TARGET_TOTAL jobs submitted. Exiting."
        break
    fi
    
    SLOTS=$((MAX_TOTAL - CURRENT))
    if [[ $SLOTS -le 0 ]]; then
        echo "[WAIT] $(date +%H:%M) | jobs=$CURRENT | submitted=$SUBMITTED_COUNT | remaining=$REMAINING | no slots"
        sleep "$POLL_INTERVAL"
        continue
    fi
    
    echo "[FILL] $(date +%H:%M) | jobs=$CURRENT | slots=$SLOTS | submitted=$SUBMITTED_COUNT | remaining=$REMAINING"
    
    BATCH_COUNT=0
    for KEY in "${ALL_KEYS[@]}"; do
        [[ $BATCH_COUNT -ge $SLOTS ]] && break
        
        # Skip if already submitted (fuzzy match on key prefix)
        if grep -qF "$KEY" "$SUBMITTED_FILE" 2>/dev/null; then
            continue
        fi
        
        # Parse key
        IFS=: read -ra PARTS <<< "$KEY"
        MODEL="${PARTS[0]}"
        CONDITION="${PARTS[1]}"
        DISTANCE="${PARTS[2]}"
        DOMAIN="${PARTS[3]}"
        if [[ ${#PARTS[@]} -eq 6 ]]; then
            RATIO="${PARTS[4]}"
            SEED="${PARTS[5]}"
        else
            RATIO=""
            SEED="${PARTS[4]}"
        fi
        
        QUEUE=$(get_queue)
        WALLTIME=$(get_walltime "$MODEL")
        RES=$(get_resources "$MODEL")
        
        COND_SHORT="${CONDITION:0:2}"
        if [[ -n "$RATIO" ]]; then
            JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_r${RATIO}_s${SEED}"
        else
            JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_s${SEED}"
        fi
        
        CMD="qsub -N $JOB_NAME -l select=1:$RES -l walltime=$WALLTIME -q $QUEUE"
        if [[ -n "$RATIO" ]]; then
            CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        else
            CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        fi
        CMD="$CMD $JOB_SCRIPT"
        
        JOB_ID=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "$KEY" >> "$SUBMITTED_FILE"
            echo "  [SUB] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | r=${RATIO:-n/a} | s$SEED | $QUEUE → $JOB_ID"
            ((BATCH_COUNT++))
            sleep 0.2
        else
            # Queue full for this queue, try next
            continue
        fi
    done
    
    echo "  → Submitted $BATCH_COUNT jobs this round"
    sleep "$POLL_INTERVAL"
done

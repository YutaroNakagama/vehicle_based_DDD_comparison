#!/bin/bash
# ============================================================
# Auto-resub v2 for unified domain_train jobs
# - Routes Lstm jobs to GPU queues (GPU-1, GPU-1A, GPU-S, GPU-L, GPU-LA)
# - Routes SvmA jobs to CPU queues (DEFAULT, SINGLE, SMALL, LONG)
# - Polls every 5 minutes
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT_CPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
JOB_SCRIPT_GPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh"
cd "$PROJECT_ROOT"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")

# Track submitted jobs (reuse existing file)
SUBMITTED_FILE="/tmp/unified_submitted_keys.txt"
touch "$SUBMITTED_FILE"

SUBMITTED_COUNT_INIT=$(wc -l < "$SUBMITTED_FILE")
echo "[INIT-v2] $SUBMITTED_COUNT_INIT jobs already tracked"

POLL_INTERVAL=300  # 5 minutes

# ---------- Resource helpers ----------
get_walltime_cpu() {
    case "$1" in
        SvmA) echo "48:00:00" ;;
        SvmW) echo "12:00:00" ;;
        Lstm) echo "16:00:00" ;;
    esac
}

get_resources_cpu() {
    case "$1" in
        SvmA) echo "ncpus=8:mem=32gb" ;;
        SvmW) echo "ncpus=8:mem=16gb" ;;
        Lstm) echo "ncpus=8:mem=32gb" ;;
    esac
}

# GPU: Lstm only, request 1 GPU + 4 CPUs (data loading)
GPU_WALLTIME="08:00:00"
GPU_RESOURCES="ncpus=4:ngpus=1:mem=32gb"

# ---------- Queue rotation ----------
CPU_QUEUE_IDX=0
get_cpu_queue() {
    local queues=("DEFAULT" "SINGLE" "LONG" "DEFAULT" "SMALL" "SINGLE")
    local q="${queues[$((CPU_QUEUE_IDX % ${#queues[@]}))]}"
    ((CPU_QUEUE_IDX++))
    echo "$q"
}

GPU_QUEUE_IDX=0
get_gpu_queue() {
    # Route based on per-queue job count - pick the least loaded GPU queue
    local GPU1_JOBS=$(count_user_jobs_in_queue "GPU-1")
    local GPU1A_JOBS=$(count_user_jobs_in_queue "GPU-1A")
    local GPUS_JOBS=$(count_user_jobs_in_queue "GPU-S")
    local GPUL_JOBS=$(count_user_jobs_in_queue "GPU-L")
    local GPULA_JOBS=$(count_user_jobs_in_queue "GPU-LA")
    
    # Per-queue limits (max_run/user):
    #   GPU-1: 4, GPU-1A: 2, GPU-S: 2, GPU-L: 1, GPU-LA: 1  (total: 10)
    # Queue buffer: keep queued jobs reasonable per queue
    
    # First fill GPU-L and GPU-LA (1 slot each, use if empty)
    if [[ $GPUL_JOBS -lt 2 ]]; then
        echo "GPU-L"
    elif [[ $GPULA_JOBS -lt 2 ]]; then
        echo "GPU-LA"
    elif [[ $GPU1_JOBS -le $GPU1A_JOBS && $GPU1_JOBS -le $GPUS_JOBS && $GPU1_JOBS -lt 15 ]]; then
        echo "GPU-1"
    elif [[ $GPU1A_JOBS -le $GPUS_JOBS && $GPU1A_JOBS -lt 10 ]]; then
        echo "GPU-1A"
    elif [[ $GPUS_JOBS -lt 10 ]]; then
        echo "GPU-S"
    elif [[ $GPU1_JOBS -lt 15 ]]; then
        echo "GPU-1"
    else
        echo ""  # all full
    fi
}

# ---------- Key helpers ----------
build_key() {
    local MODEL="$1" CONDITION="$2" DISTANCE="$3" DOMAIN="$4" SEED="$5" RATIO="$6"
    if [[ -n "$RATIO" ]]; then
        echo "${MODEL}:${CONDITION}:${DISTANCE}:${DOMAIN}:${RATIO}:${SEED}"
    else
        echo "${MODEL}:${CONDITION}:${DISTANCE}:${DOMAIN}:${SEED}"
    fi
}

# ---------- Generate remaining job keys ----------
# Only SvmA and Lstm (SvmW already fully submitted)
ALL_KEYS=()
for MODEL in "Lstm" "SvmA"; do
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
echo "[INIT-v2] Candidate keys: ${#ALL_KEYS[@]} (Lstm + SvmA)"

# ---------- Count jobs per queue type ----------
count_user_jobs_in_queue() {
    local QNAME="$1"
    qstat -u s2240011 2>/dev/null | tail -n +6 | awk -v q="$QNAME" '$3==q' | wc -l
}

# ---------- Main loop ----------
while true; do
    CURRENT=$(qstat -u s2240011 2>/dev/null | awk 'NR>5' | wc -l)
    SUBMITTED_COUNT=$(wc -l < "$SUBMITTED_FILE")
    
    # Count remaining keys not yet submitted
    REMAINING=0
    for KEY in "${ALL_KEYS[@]}"; do
        if ! grep -qF "$KEY" "$SUBMITTED_FILE" 2>/dev/null; then
            ((REMAINING++))
        fi
    done
    
    if [[ $REMAINING -le 0 ]]; then
        echo "[DONE] All jobs submitted. Exiting."
        break
    fi
    
    echo "[POLL] $(date +%H:%M) | total_jobs=$CURRENT | submitted=$SUBMITTED_COUNT | remaining=$REMAINING"
    
    BATCH_COUNT=0
    
    for KEY in "${ALL_KEYS[@]}"; do
        # Skip already submitted
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
        
        # Route by model
        if [[ "$MODEL" == "Lstm" ]]; then
            QUEUE=$(get_gpu_queue)
            WALLTIME="$GPU_WALLTIME"
            RES="$GPU_RESOURCES"
            JOB_SCRIPT="$JOB_SCRIPT_GPU"
            
            # If no GPU queue available, skip this job
            if [[ -z "$QUEUE" ]]; then
                continue
            fi
        else
            # CPU (SvmA)
            QUEUE=$(get_cpu_queue)
            WALLTIME=$(get_walltime_cpu "$MODEL")
            RES=$(get_resources_cpu "$MODEL")
            JOB_SCRIPT="$JOB_SCRIPT_CPU"
            
            # Check total CPU queue slots
            CPU_TOTAL=$(qstat -u s2240011 2>/dev/null | tail -n +6 | awk '$3!~/GPU/' | wc -l)
            if [[ $CPU_TOTAL -ge 125 ]]; then
                continue
            fi
        fi
        
        # Build job name
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
            sleep 0.3
        fi
    done
    
    echo "  → Submitted $BATCH_COUNT jobs this round"
    
    if [[ $BATCH_COUNT -eq 0 ]]; then
        echo "  (all queues full, waiting...)"
    fi
    
    sleep "$POLL_INTERVAL"
done

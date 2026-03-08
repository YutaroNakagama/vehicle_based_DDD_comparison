#!/bin/bash
# ============================================================
# Auto-submit daemon — auto-submit missing Exp3 mixed jobs
# ============================================================
# Detect mixed jobs without evaluation results and based on queue availability
# Auto qsub. Check every 5 minutes.
#
# Usage:
#   nohup bash scripts/hpc/launchers/auto_resub_mixed_exp3.sh &
#   # Log: /tmp/auto_resub_mixed_exp3.log
# ============================================================

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
LOG="/tmp/auto_resub_mixed_exp3.log"
SUBMITTED_KEYS="/tmp/mixed_exp3_submitted_keys.txt"
POLL_INTERVAL=300  # 5 minutes
MODE="mixed"
N_TRIALS=100
RANKING="knn"

# ---- Error trap: output log on daemon death ----
trap 'echo "[$(date +%H:%M)] TRAP: daemon exiting unexpectedly (line $LINENO, exit=$?)" >> "$LOG"' EXIT
trap 'echo "[$(date +%H:%M)] TRAP: received signal, exiting" >> "$LOG"; exit 1' INT TERM HUP

# ---- Queue limit (queued per user) ----
declare -A QUEUE_MAX=( [SINGLE]=40 [DEFAULT]=40 [SMALL]=30 [LONG]=15 )

# ---- Global arrays (declared outside functions) ----
declare -A QUEUE_CURRENT=()

CPU_QUEUES=("SINGLE" "DEFAULT" "SMALL" "LONG")

# ---- Initialization: load already-submitted keys ----
touch "$SUBMITTED_KEYS"

# Recover keys from jobs already in queue
restore_keys_from_queue() {
    local qstat_output
    qstat_output=$(qstat -u s2240011 2>/dev/null | tail -n +6 || true)
    # mixed job name pattern: *_m_* (mixed mode marker)
    echo "$qstat_output" | awk '{print $4}' | grep '_m_' | while read -r name; do
        # Infer key from job name (name: Sv_sm_mi_m_r0.1_s42)
        echo "QUEUE:$name"
    done
}

# ---- Resource definitions ----
get_resources() {
    local model="$1"
    local cond="$2"
    local is_smote=false
    [[ "$cond" == "smote_plain" || "$cond" == "smote" ]] && is_smote=true

    case "$model" in
        SvmW)
            if $is_smote; then echo "ncpus=8:mem=24gb 24:00:00"
            else echo "ncpus=8:mem=24gb 16:00:00"; fi ;;
        SvmA)
            if $is_smote; then echo "ncpus=8:mem=48gb 48:00:00"
            else echo "ncpus=8:mem=48gb 30:00:00"; fi ;;
        Lstm)
            if $is_smote; then echo "ncpus=8:mem=48gb 24:00:00"
            else echo "ncpus=8:mem=48gb 20:00:00"; fi ;;
    esac
}

# ---- Enumerate all experiment conditions ----
ALL_JOBS=()
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("in_domain" "out_domain")
SEEDS=(42 123)
RATIOS=(0.1 0.5)
# Lstm mixed 84/84 complete — SvmW, SvmA onlysubmission target
MODELS=("SvmW" "SvmA")

for MODEL in "${MODELS[@]}"; do
    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # baseline
                ALL_JOBS+=("$MODEL|baseline|$DIST|$DOM|$SEED|")
                # ratio-based
                for COND in "smote_plain" "smote" "undersample"; do
                    for RATIO in "${RATIOS[@]}"; do
                        ALL_JOBS+=("$MODEL|$COND|$DIST|$DOM|$SEED|$RATIO")
                    done
                done
            done
        done
    done
done

echo "[$(date +%H:%M)] Daemon started. Total expected: ${#ALL_JOBS[@]} mixed jobs" >> "$LOG"
echo "[$(date +%H:%M)] Polling every ${POLL_INTERVAL}s" >> "$LOG"

# ---- Check if evaluation results exist ----
has_eval_result() {
    local model="$1" cond="$2" dist="$3" dom="$4" seed="$5" ratio="$6"
    local eval_dir="results/outputs/evaluation/$model"
    
    # condition → tag mapping
    local tag
    case "$cond" in
        baseline) tag="baseline" ;;
        smote_plain) tag="smote_plain" ;;
        smote) tag="imbalv3" ;;
        undersample) tag="undersample_rus" ;;
    esac
    
    local pattern
    if [[ -z "$ratio" ]]; then
        pattern="eval_results_${model}_mixed_prior_${model}_${tag}_knn_${dist}_${dom}_mixed_split2_s${seed}"
    else
        # *ratio matches both subjectwise_ratio and ratio with
        pattern="eval_results_${model}_mixed_prior_${model}_${tag}_knn_${dist}_${dom}_mixed_split2_*ratio${ratio}_s${seed}"
    fi
    
    # Check if matching file exists
    find "$eval_dir" -name "${pattern}*.json" 2>/dev/null | grep -q .
}

# ---- Check queue availability ----
get_queue_counts() {
    local qstat_output
    qstat_output=$(qstat -u s2240011 2>/dev/null | tail -n +6 || true)
    
    # QUEUE_CURRENT already declared globally — Only updating the value here
    for q in "${CPU_QUEUES[@]}"; do
        QUEUE_CURRENT[$q]=$(echo "$qstat_output" | awk -v q="$q" '$3==q' | wc -l || echo 0)
    done
}

find_available_queue() {
    for q in "${CPU_QUEUES[@]}"; do
        local current="${QUEUE_CURRENT[$q]:-0}"
        local max="${QUEUE_MAX[$q]:-0}"
        if (( current < max )); then
            echo "$q"
            return 0
        fi
    done
    return 1
}

# ---- Main loop ----
while true; do
    get_queue_counts || true
    
    SUBMITTED_THIS_ROUND=0
    REMAINING=0
    
    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r MODEL COND DIST DOM SEED RATIO <<< "$job_spec"
        
        # Create unique key
        KEY="${MODEL}:${COND}:${DIST}:${DOM}:mixed:${RATIO}:s${SEED}"
        
        # Skip if eval result already exists
        if has_eval_result "$MODEL" "$COND" "$DIST" "$DOM" "$SEED" "$RATIO"; then
            continue
        fi
        
        # Skip if already submitted
        if grep -qF "$KEY" "$SUBMITTED_KEYS" 2>/dev/null; then
            # Check if job is still in queue
            ((REMAINING++)) || true
            continue
        fi
        
        # Find available queue
        QUEUE=""
        QUEUE=$(find_available_queue) || true
        if [[ -z "$QUEUE" ]]; then
            ((REMAINING++)) || true
            continue
        fi
        
        # Get resources
        RES=$(get_resources "$MODEL" "$COND")
        NCPUS_MEM=$(echo "$RES" | cut -d' ' -f1)
        WALLTIME=$(echo "$RES" | cut -d' ' -f2)
        
        # Generate job name
        COND_SHORT="${COND:0:2}"
        if [[ -n "$RATIO" ]]; then
            JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DIST:0:1}${DOM:0:1}_m_r${RATIO}_s${SEED}"
        else
            JOB_NAME="${MODEL:0:2}_bs_${DIST:0:1}${DOM:0:1}_m_s${SEED}"
        fi
        
        # Build qsub command
        VARS="MODEL=$MODEL,CONDITION=$COND,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        [[ -n "$RATIO" ]] && VARS="$VARS,RATIO=$RATIO"
        
        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v $VARS $JOB_SCRIPT"
        
        JOB_ID=$(eval "$CMD" 2>&1) || {
            ((REMAINING++)) || true
            continue
        }
        
        # Record submission
        echo "$KEY:$JOB_ID" >> "$SUBMITTED_KEYS"
        ((SUBMITTED_THIS_ROUND++)) || true
        
        # Update queue count
        QUEUE_CURRENT[$QUEUE]=$(( ${QUEUE_CURRENT[$QUEUE]:-0} + 1 ))
        
        desc="$MODEL | $COND | $DIST | $DOM | r=${RATIO:-n/a} | s$SEED | $QUEUE"
        echo "  [SUB] $desc → $JOB_ID" >> "$LOG"
        
        sleep 0.3
    done
    
    TOTAL_QUEUED=$(qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l)
    TOTAL_SUBMITTED=$(wc -l < "$SUBMITTED_KEYS")
    echo "[POLL] $(date +%H:%M) | queued=$TOTAL_QUEUED | submitted=$TOTAL_SUBMITTED | new=$SUBMITTED_THIS_ROUND | remaining=$REMAINING" >> "$LOG"
    
    if [[ "$SUBMITTED_THIS_ROUND" -eq 0 && "$REMAINING" -eq 0 ]]; then
        echo "[DONE] All mixed exp3 jobs submitted or completed. Exiting." >> "$LOG"
        break
    fi
    
    if [[ "$SUBMITTED_THIS_ROUND" -eq 0 ]]; then
        echo "  (all queues full, waiting...)" >> "$LOG"
    fi
    
    sleep "$POLL_INTERVAL"
done

echo "[$(date +%H:%M)] Daemon finished." >> "$LOG"

#!/bin/bash
# =============================================================================
# Auto-submit remaining Imbalance V2 experiments when queue space available
# =============================================================================
# This script monitors queue status and submits remaining experiments
# when space becomes available.
# 
# Usage: nohup bash auto_submit_remaining.sh &
# =============================================================================

set -euo pipefail

PROJECT_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs/imbal_v2_fixed"
SCRIPT_DIR="${PROJECT_DIR}/scripts/hpc/logs/imbalance/tmp_jobs2"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR"
LOG_FILE="${LOG_DIR}/auto_submit_${TIMESTAMP}.txt"

N_TRIALS=75

# Maximum total jobs (including TINY which are just Hold jobs)
MAX_TOTAL_JOBS=100
# Wait between checks (seconds)
CHECK_INTERVAL=60

SUBMITTED=()

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

count_jobs() {
    qstat -u s2240011 2>/dev/null | grep -E "^[0-9]" | wc -l
}

# Submit a single experiment
submit_experiment() {
    local METHOD=$1
    local RATIO=$2
    local SEED=$3
    local MODEL=$4
    local OVERSAMPLE_FLAG=$5
    local OVERSAMPLE_METHOD=$6
    local QUEUE=$7
    
    local RATIO_SAFE="${RATIO//./_}"
    local TAG="imbal_v2_${METHOD}_ratio${RATIO_SAFE}_seed${SEED}"
    local SHORT="${METHOD:0:4}"
    local JOB_NAME="iv2${SHORT}${RATIO_SAFE:0:1}s${SEED:0:2}"
    
    case "$QUEUE" in
        "LONG")     WALLTIME="168:00:00"; MEM="8gb"; NCPUS="4" ;;
        "DEFAULT")  WALLTIME="24:00:00"; MEM="8gb"; NCPUS="4" ;;
        "SMALL")    WALLTIME="12:00:00"; MEM="8gb"; NCPUS="4" ;;
        "SINGLE")   WALLTIME="06:00:00"; MEM="8gb"; NCPUS="4" ;;
        *)          WALLTIME="06:00:00"; MEM="8gb"; NCPUS="4" ;;
    esac
    
    local TRAIN_SCRIPT="${SCRIPT_DIR}/tr_${METHOD}_${RATIO_SAFE}_s${SEED}.sh"
    
    cat > "$TRAIN_SCRIPT" << EOF
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -q ${QUEUE}
#PBS -l select=1:ncpus=${NCPUS}:mem=${MEM}
#PBS -l walltime=${WALLTIME}
#PBS -j oe
#PBS -o ${PROJECT_DIR}/scripts/hpc/logs/

set -euo pipefail
source /home/s2240011/conda/etc/profile.d/conda.sh
conda activate python310

cd ${PROJECT_DIR}
export PYTHONPATH="${PROJECT_DIR}:\${PYTHONPATH:-}"
export N_TRIALS_OVERRIDE=${N_TRIALS}
export OMP_NUM_THREADS=1

echo "[IMBAL V2] ${TAG} | MODEL=${MODEL} | Q=${QUEUE}"

EOF

    if [[ "$OVERSAMPLE_FLAG" == "yes" ]]; then
        cat >> "$TRAIN_SCRIPT" << EOF
python scripts/python/train/train.py --model ${MODEL} --mode pooled --tag ${TAG} --seed ${SEED} --time_stratify_labels --use_oversampling --oversample_method ${OVERSAMPLE_METHOD} --target_ratio ${RATIO}
EOF
    else
        cat >> "$TRAIN_SCRIPT" << EOF
python scripts/python/train/train.py --model ${MODEL} --mode pooled --tag ${TAG} --seed ${SEED} --time_stratify_labels
EOF
    fi
    
    echo "echo '[DONE] ${TAG}'" >> "$TRAIN_SCRIPT"
    chmod +x "$TRAIN_SCRIPT"
    
    TRAIN_JOB_ID=$(qsub "$TRAIN_SCRIPT" 2>&1)
    
    if [[ "$TRAIN_JOB_ID" =~ ^[0-9]+ ]]; then
        TRAIN_JOB_ID_CLEAN=$(echo "$TRAIN_JOB_ID" | grep -oE '^[0-9]+\.[a-zA-Z0-9-]+' || echo "$TRAIN_JOB_ID")
        log "TRAIN [${QUEUE}]: ${TAG} -> ${TRAIN_JOB_ID_CLEAN}"
        
        # Evaluation job
        local EVAL_SCRIPT="${SCRIPT_DIR}/ev_${METHOD}_${RATIO_SAFE}_s${SEED}.sh"
        
        cat > "$EVAL_SCRIPT" << EOF
#!/bin/bash
#PBS -N e${JOB_NAME}
#PBS -q TINY
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o ${PROJECT_DIR}/scripts/hpc/logs/
#PBS -W depend=afterok:${TRAIN_JOB_ID_CLEAN}

set -euo pipefail
source /home/s2240011/conda/etc/profile.d/conda.sh
conda activate python310

cd ${PROJECT_DIR}
export PYTHONPATH="${PROJECT_DIR}:\${PYTHONPATH:-}"

python scripts/python/evaluation/evaluate.py --model ${MODEL} --mode pooled --tag ${TAG}
echo "[DONE] Eval: ${TAG}"
EOF
        
        chmod +x "$EVAL_SCRIPT"
        qsub "$EVAL_SCRIPT" >/dev/null 2>&1
        return 0
    else
        log "[FAILED] ${TAG}: $TRAIN_JOB_ID"
        return 1
    fi
}

# Define remaining experiments (28 experiments total, after undersample_tomek 0.5 s123 was submitted)
# Each entry: "METHOD RATIO SEED MODEL OVERSAMPLE_FLAG OVERSAMPLE_METHOD QUEUE"
EXPERIMENTS=(
    # Undersample-Tomek remaining (4 more)
    "undersample_tomek 0.5 456 RF yes undersample_tomek LONG"
    "undersample_tomek 1.0 42 RF yes undersample_tomek SMALL"
    "undersample_tomek 1.0 123 RF yes undersample_tomek DEFAULT"
    "undersample_tomek 1.0 456 RF yes undersample_tomek LONG"
    
    # Undersample-ENN (9)
    "undersample_enn 0.1 42 RF yes undersample_enn SINGLE"
    "undersample_enn 0.1 123 RF yes undersample_enn DEFAULT"
    "undersample_enn 0.1 456 RF yes undersample_enn LONG"
    "undersample_enn 0.5 42 RF yes undersample_enn SMALL"
    "undersample_enn 0.5 123 RF yes undersample_enn DEFAULT"
    "undersample_enn 0.5 456 RF yes undersample_enn LONG"
    "undersample_enn 1.0 42 RF yes undersample_enn SINGLE"
    "undersample_enn 1.0 123 RF yes undersample_enn SMALL"
    "undersample_enn 1.0 456 RF yes undersample_enn DEFAULT"
    
    # BalancedRF (3)
    "balanced_rf 1.0 42 BalancedRF no none LONG"
    "balanced_rf 1.0 123 BalancedRF no none SINGLE"
    "balanced_rf 1.0 456 BalancedRF no none SMALL"
    
    # EasyEnsemble (3)
    "easy_ensemble 1.0 42 EasyEnsemble no none DEFAULT"
    "easy_ensemble 1.0 123 EasyEnsemble no none LONG"
    "easy_ensemble 1.0 456 EasyEnsemble no none SINGLE"
    
    # SMOTE+BalancedRF (9)
    "smote_balanced_rf 0.1 42 BalancedRF yes smote SMALL"
    "smote_balanced_rf 0.1 123 BalancedRF yes smote DEFAULT"
    "smote_balanced_rf 0.1 456 BalancedRF yes smote LONG"
    "smote_balanced_rf 0.5 42 BalancedRF yes smote SINGLE"
    "smote_balanced_rf 0.5 123 BalancedRF yes smote SMALL"
    "smote_balanced_rf 0.5 456 BalancedRF yes smote DEFAULT"
    "smote_balanced_rf 1.0 42 BalancedRF yes smote LONG"
    "smote_balanced_rf 1.0 123 BalancedRF yes smote SINGLE"
    "smote_balanced_rf 1.0 456 BalancedRF yes smote SMALL"
)

log "============================================================"
log "Auto-submit remaining Imbalance V2 experiments"
log "Total experiments to submit: ${#EXPERIMENTS[@]}"
log "Max jobs: $MAX_TOTAL_JOBS"
log "Check interval: ${CHECK_INTERVAL}s"
log "============================================================"

NEXT_IDX=0
TOTAL=${#EXPERIMENTS[@]}

while [[ $NEXT_IDX -lt $TOTAL ]]; do
    CURRENT_JOBS=$(count_jobs)
    log "Current jobs: $CURRENT_JOBS, Pending: $((TOTAL - NEXT_IDX)) experiments"
    
    # Submit jobs if we have room (leave buffer of 10 for safety)
    while [[ $CURRENT_JOBS -lt $((MAX_TOTAL_JOBS - 10)) ]] && [[ $NEXT_IDX -lt $TOTAL ]]; do
        EXP="${EXPERIMENTS[$NEXT_IDX]}"
        read -r METHOD RATIO SEED MODEL OVERSAMPLE_FLAG OVERSAMPLE_METHOD QUEUE <<< "$EXP"
        
        if submit_experiment "$METHOD" "$RATIO" "$SEED" "$MODEL" "$OVERSAMPLE_FLAG" "$OVERSAMPLE_METHOD" "$QUEUE"; then
            NEXT_IDX=$((NEXT_IDX + 1))
            CURRENT_JOBS=$((CURRENT_JOBS + 2))  # train + eval
        else
            log "Submission failed, will retry later"
            break
        fi
        
        sleep 1
    done
    
    if [[ $NEXT_IDX -lt $TOTAL ]]; then
        log "Waiting ${CHECK_INTERVAL}s for queue space..."
        sleep $CHECK_INTERVAL
    fi
done

log "============================================================"
log "All ${TOTAL} experiments submitted!"
log "============================================================"

#!/bin/bash
# =============================================================================
# Imbalance V2 - Submit remaining experiments (after queue limit)
# =============================================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
LOG_DIR="${PROJECT_DIR}/logs/imbal_v2_fixed"
SCRIPT_DIR="${PROJECT_DIR}/scripts/hpc/logs/imbalance/tmp_jobs"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR"
LOG_FILE="${LOG_DIR}/submit_remaining_${TIMESTAMP}.txt"

echo "============================================================" | tee "$LOG_FILE"
echo "Imbalance V2 - Remaining Experiments" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

N_TRIALS=75
SEEDS=("42" "123" "456")
RATIOS=("0.1" "0.5" "1.0")

QUEUES=("LONG" "LONG" "DEFAULT" "LONG" "SMALL" "DEFAULT")
QUEUE_IDX=0

TRAIN_COUNT=0
EVAL_COUNT=0

submit_job() {
    local METHOD=$1
    local RATIO=$2
    local SEED=$3
    local MODEL=$4
    local OVERSAMPLE_FLAG=$5
    local OVERSAMPLE_METHOD=$6
    
    local RATIO_SAFE="${RATIO//./_}"
    local TAG="imbal_v2_${METHOD}_ratio${RATIO_SAFE}_seed${SEED}"
    local JOB_NAME="iv2_${METHOD:0:4}_r${RATIO_SAFE:0:3}_s${SEED:0:2}"
    
    local QUEUE="${QUEUES[$((QUEUE_IDX % ${#QUEUES[@]}))]}"
    QUEUE_IDX=$((QUEUE_IDX + 1))
    
    case "$QUEUE" in
        "LONG")     WALLTIME="168:00:00"; MEM="8gb"; NCPUS="4" ;;
        "DEFAULT")  WALLTIME="24:00:00"; MEM="8gb"; NCPUS="4" ;;
        "SMALL")    WALLTIME="12:00:00"; MEM="8gb"; NCPUS="4" ;;
        *)          WALLTIME="06:00:00"; MEM="8gb"; NCPUS="4" ;;
    esac
    
    local TRAIN_SCRIPT="${SCRIPT_DIR}/train_${METHOD}_${RATIO_SAFE}_s${SEED}.sh"
    
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

echo "[IMBALANCE V2] Training: ${TAG}"
echo "METHOD: ${METHOD}, MODEL: ${MODEL}, RATIO: ${RATIO}, SEED: ${SEED}"

EOF

    if [[ "$OVERSAMPLE_FLAG" == "yes" ]]; then
        cat >> "$TRAIN_SCRIPT" << EOF
python scripts/python/train.py --model ${MODEL} --mode pooled --tag ${TAG} --seed ${SEED} --time_stratify_labels --use_oversampling --oversample_method ${OVERSAMPLE_METHOD} --target_ratio ${RATIO}
EOF
    else
        cat >> "$TRAIN_SCRIPT" << EOF
python scripts/python/train.py --model ${MODEL} --mode pooled --tag ${TAG} --seed ${SEED} --time_stratify_labels
EOF
    fi
    
    cat >> "$TRAIN_SCRIPT" << EOF

echo "[DONE] Training complete: ${TAG}"
EOF
    
    chmod +x "$TRAIN_SCRIPT"
    
    TRAIN_JOB_ID=$(qsub "$TRAIN_SCRIPT" 2>&1)
    
    if [[ "$TRAIN_JOB_ID" =~ ^[0-9]+ ]]; then
        TRAIN_JOB_ID_CLEAN=$(echo "$TRAIN_JOB_ID" | grep -oE '^[0-9]+\.[a-zA-Z0-9-]+' || echo "$TRAIN_JOB_ID")
        echo "TRAIN [${QUEUE}]: ${TAG} -> ${TRAIN_JOB_ID_CLEAN}" | tee -a "$LOG_FILE"
        TRAIN_COUNT=$((TRAIN_COUNT + 1))
        
        local EVAL_SCRIPT="${SCRIPT_DIR}/eval_${METHOD}_${RATIO_SAFE}_s${SEED}.sh"
        
        cat > "$EVAL_SCRIPT" << EOF
#!/bin/bash
#PBS -N ev_${JOB_NAME}
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

python scripts/python/evaluate.py --model ${MODEL} --mode pooled --tag ${TAG}

echo "[DONE] Evaluation: ${TAG}"
EOF
        
        chmod +x "$EVAL_SCRIPT"
        
        EVAL_JOB_ID=$(qsub "$EVAL_SCRIPT" 2>&1)
        if [[ "$EVAL_JOB_ID" =~ ^[0-9]+ ]]; then
            EVAL_COUNT=$((EVAL_COUNT + 1))
        fi
    else
        echo "[ERROR] Failed: ${TAG}: $TRAIN_JOB_ID" | tee -a "$LOG_FILE"
    fi
    
    sleep 0.3
}

# =============================================================================
# REMAINING experiments (continue from where we stopped)
# Already submitted: undersample_rus (9), undersample_tomek (4)
# =============================================================================

echo "" | tee -a "$LOG_FILE"
echo "=== Undersample-Tomek (remaining) ===" | tee -a "$LOG_FILE"
# Already done: ratio0_1 (all seeds), ratio0_5 seed42
for SEED in "123" "456"; do
    submit_job "undersample_tomek" "0.5" "$SEED" "RF" "yes" "undersample_tomek"
done
for SEED in "${SEEDS[@]}"; do
    submit_job "undersample_tomek" "1.0" "$SEED" "RF" "yes" "undersample_tomek"
done

echo "" | tee -a "$LOG_FILE"
echo "=== Undersample-ENN (all) ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "undersample_enn" "$RATIO" "$SEED" "RF" "yes" "undersample_enn"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== BalancedRF ===" | tee -a "$LOG_FILE"
for SEED in "${SEEDS[@]}"; do
    submit_job "balanced_rf" "1.0" "$SEED" "BalancedRF" "no" ""
done

echo "" | tee -a "$LOG_FILE"
echo "=== EasyEnsemble ===" | tee -a "$LOG_FILE"
for SEED in "${SEEDS[@]}"; do
    submit_job "easy_ensemble" "1.0" "$SEED" "EasyEnsemble" "no" ""
done

echo "" | tee -a "$LOG_FILE"
echo "=== SMOTE+BalancedRF ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "smote_balanced_rf" "$RATIO" "$SEED" "BalancedRF" "yes" "smote"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Training jobs submitted: $TRAIN_COUNT" | tee -a "$LOG_FILE"
echo "Evaluation jobs submitted: $EVAL_COUNT" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

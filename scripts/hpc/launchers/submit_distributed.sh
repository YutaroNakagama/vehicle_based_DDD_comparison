#!/bin/bash
# =============================================================================
# Imbalance V2 - Distributed submission across all available queues
# =============================================================================
# Submits remaining 29 experiments distributed across SINGLE, SMALL, DEFAULT, LONG
# Uses individual jobs (not array) to work around queue limits
# =============================================================================

set -euo pipefail

PROJECT_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs/imbal_v2_fixed"
SCRIPT_DIR="${PROJECT_DIR}/scripts/hpc/logs/imbalance/tmp_jobs2"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR"
LOG_FILE="${LOG_DIR}/distributed_${TIMESTAMP}.txt"

N_TRIALS=75

# Queue rotation: use less congested queues
# SINGLE: 13 jobs (40 max), SMALL: 8 jobs (30 max), DEFAULT: 8 jobs (40 max), LONG: 15 jobs
# Available: SINGLE +27, SMALL +22, DEFAULT +32, LONG +many
QUEUES=("DEFAULT" "LONG" "SMALL" "DEFAULT" "LONG" "SINGLE")
QUEUE_IDX=0

TRAIN_COUNT=0
EVAL_COUNT=0

echo "============================================================" | tee "$LOG_FILE"
echo "Imbalance V2 - Distributed Submission" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

submit_job() {
    local METHOD=$1
    local RATIO=$2
    local SEED=$3
    local MODEL=$4
    local OVERSAMPLE_FLAG=$5
    local OVERSAMPLE_METHOD=$6
    
    local RATIO_SAFE="${RATIO//./_}"
    local TAG="imbal_v2_${METHOD}_ratio${RATIO_SAFE}_seed${SEED}"
    local SHORT="${METHOD:0:4}"
    local JOB_NAME="iv2${SHORT}${RATIO_SAFE:0:1}s${SEED:0:2}"
    
    local QUEUE="${QUEUES[$((QUEUE_IDX % ${#QUEUES[@]}))]}"
    QUEUE_IDX=$((QUEUE_IDX + 1))
    
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
        echo "TRAIN [${QUEUE}]: ${TAG} -> ${TRAIN_JOB_ID_CLEAN}" | tee -a "$LOG_FILE"
        TRAIN_COUNT=$((TRAIN_COUNT + 1))
        
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
        EVAL_JOB_ID=$(qsub "$EVAL_SCRIPT" 2>&1)
        if [[ "$EVAL_JOB_ID" =~ ^[0-9]+ ]]; then
            EVAL_COUNT=$((EVAL_COUNT + 1))
        fi
    else
        echo "[ERROR] ${TAG}: $TRAIN_JOB_ID" | tee -a "$LOG_FILE"
    fi
    
    sleep 0.5
}

# =============================================================================
# Submit all 29 remaining experiments
# =============================================================================

echo "" | tee -a "$LOG_FILE"
echo "=== Undersample-Tomek (5 remaining) ===" | tee -a "$LOG_FILE"
submit_job "undersample_tomek" "0.5" "123" "RF" "yes" "undersample_tomek"
submit_job "undersample_tomek" "0.5" "456" "RF" "yes" "undersample_tomek"
submit_job "undersample_tomek" "1.0" "42" "RF" "yes" "undersample_tomek"
submit_job "undersample_tomek" "1.0" "123" "RF" "yes" "undersample_tomek"
submit_job "undersample_tomek" "1.0" "456" "RF" "yes" "undersample_tomek"

echo "" | tee -a "$LOG_FILE"
echo "=== Undersample-ENN (9) ===" | tee -a "$LOG_FILE"
for RATIO in "0.1" "0.5" "1.0"; do
    for SEED in "42" "123" "456"; do
        submit_job "undersample_enn" "$RATIO" "$SEED" "RF" "yes" "undersample_enn"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== BalancedRF (3) ===" | tee -a "$LOG_FILE"
for SEED in "42" "123" "456"; do
    submit_job "balanced_rf" "1.0" "$SEED" "BalancedRF" "no" ""
done

echo "" | tee -a "$LOG_FILE"
echo "=== EasyEnsemble (3) ===" | tee -a "$LOG_FILE"
for SEED in "42" "123" "456"; do
    submit_job "easy_ensemble" "1.0" "$SEED" "EasyEnsemble" "no" ""
done

echo "" | tee -a "$LOG_FILE"
echo "=== SMOTE+BalancedRF (9) ===" | tee -a "$LOG_FILE"
for RATIO in "0.1" "0.5" "1.0"; do
    for SEED in "42" "123" "456"; do
        submit_job "smote_balanced_rf" "$RATIO" "$SEED" "BalancedRF" "yes" "smote"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Summary" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Training jobs: $TRAIN_COUNT / 29" | tee -a "$LOG_FILE"
echo "Evaluation jobs: $EVAL_COUNT" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

#!/bin/bash
# =============================================================================
# Imbalance V2 Fixed - All missing experiments
# =============================================================================
# Fixed: undersample now uses --use_oversampling --oversample_method
# Uses multiple queues to avoid limits
# =============================================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
LOG_DIR="${PROJECT_DIR}/logs/imbal_v2_fixed"
SCRIPT_DIR="${PROJECT_DIR}/scripts/hpc/logs/imbalance/tmp_jobs"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR"
LOG_FILE="${LOG_DIR}/submit_${TIMESTAMP}.txt"

echo "============================================================" | tee "$LOG_FILE"
echo "Imbalance V2 Fixed Experiment Submission" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Configuration
N_TRIALS=75
SEEDS=("42" "123" "456")
RATIOS=("0.1" "0.5" "1.0")

# Queue rotation to avoid limits
QUEUES=("SINGLE" "SINGLE" "DEFAULT" "SINGLE" "SINGLE" "SMALL")
QUEUE_IDX=0

TRAIN_COUNT=0
EVAL_COUNT=0

submit_job() {
    local METHOD=$1
    local RATIO=$2
    local SEED=$3
    local MODEL=$4
    local OVERSAMPLE_FLAG=$5  # "yes" or "no"
    local OVERSAMPLE_METHOD=$6  # method name or empty
    
    local RATIO_SAFE="${RATIO//./_}"
    local TAG="imbal_v2_${METHOD}_ratio${RATIO_SAFE}_seed${SEED}"
    local JOB_NAME="iv2_${METHOD:0:4}_r${RATIO_SAFE:0:3}_s${SEED:0:2}"
    
    # Select queue
    local QUEUE="${QUEUES[$((QUEUE_IDX % ${#QUEUES[@]}))]}"
    QUEUE_IDX=$((QUEUE_IDX + 1))
    
    # Set resources based on queue
    case "$QUEUE" in
        "TINY")     WALLTIME="00:30:00"; MEM="4gb"; NCPUS="2" ;;
        "SINGLE")   WALLTIME="06:00:00"; MEM="8gb"; NCPUS="4" ;;
        "SMALL")    WALLTIME="12:00:00"; MEM="8gb"; NCPUS="4" ;;
        "DEFAULT")  WALLTIME="24:00:00"; MEM="8gb"; NCPUS="4" ;;
        *)          WALLTIME="06:00:00"; MEM="8gb"; NCPUS="4" ;;
    esac
    
    # Create training script
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
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "============================================================"
echo "[IMBALANCE V2] Training with N_TRIALS=${N_TRIALS}"
echo "============================================================"
echo "METHOD: ${METHOD}"
echo "MODEL: ${MODEL}"
echo "RATIO: ${RATIO}"
echo "SEED: ${SEED}"
echo "TAG: ${TAG}"
echo "QUEUE: ${QUEUE}"
echo "PBS_JOBID: \${PBS_JOBID:-local}"
echo "============================================================"

EOF

    # Add training command
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
    
    # Submit training job
    TRAIN_JOB_ID=$(qsub "$TRAIN_SCRIPT" 2>&1)
    
    if [[ "$TRAIN_JOB_ID" =~ ^[0-9]+ ]]; then
        TRAIN_JOB_ID_CLEAN=$(echo "$TRAIN_JOB_ID" | grep -oE '^[0-9]+\.[a-zA-Z0-9-]+' || echo "$TRAIN_JOB_ID")
        echo "TRAIN [${QUEUE}]: ${TAG} -> ${TRAIN_JOB_ID_CLEAN}" | tee -a "$LOG_FILE"
        TRAIN_COUNT=$((TRAIN_COUNT + 1))
        
        # Create evaluation script
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

echo "============================================================"
echo "[IMBALANCE V2] Evaluation"
echo "============================================================"
echo "TAG: ${TAG}"
echo "============================================================"

python scripts/python/evaluate.py --model ${MODEL} --mode pooled --tag ${TAG}

echo "[DONE] Evaluation complete: ${TAG}"
EOF
        
        chmod +x "$EVAL_SCRIPT"
        
        EVAL_JOB_ID=$(qsub "$EVAL_SCRIPT" 2>&1)
        if [[ "$EVAL_JOB_ID" =~ ^[0-9]+ ]]; then
            EVAL_COUNT=$((EVAL_COUNT + 1))
        fi
    else
        echo "[ERROR] Failed: ${TAG}: $TRAIN_JOB_ID" | tee -a "$LOG_FILE"
    fi
    
    sleep 0.2
}

# =============================================================================
# Submit all experiments
# =============================================================================

echo "" | tee -a "$LOG_FILE"
echo "=== Submitting BASELINE experiments ===" | tee -a "$LOG_FILE"
for SEED in "${SEEDS[@]}"; do
    submit_job "baseline" "1.0" "$SEED" "RF" "no" ""
done

echo "" | tee -a "$LOG_FILE"
echo "=== Submitting SMOTE experiments ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "smote" "$RATIO" "$SEED" "RF" "yes" "smote"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== Submitting SMOTE-Tomek experiments ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "smote_tomek" "$RATIO" "$SEED" "RF" "yes" "smote_tomek"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== Submitting SMOTE-ENN experiments ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "smote_enn" "$RATIO" "$SEED" "RF" "yes" "smote_enn"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== Submitting SMOTE-RUS experiments ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "smote_rus" "$RATIO" "$SEED" "RF" "yes" "smote_rus"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== Submitting Undersample-RUS experiments (FIXED) ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # FIX: Use --use_oversampling --oversample_method undersample_rus
        submit_job "undersample_rus" "$RATIO" "$SEED" "RF" "yes" "undersample_rus"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== Submitting Undersample-Tomek experiments (FIXED) ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "undersample_tomek" "$RATIO" "$SEED" "RF" "yes" "undersample_tomek"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== Submitting Undersample-ENN experiments (FIXED) ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "undersample_enn" "$RATIO" "$SEED" "RF" "yes" "undersample_enn"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== Submitting BalancedRF experiments ===" | tee -a "$LOG_FILE"
for SEED in "${SEEDS[@]}"; do
    submit_job "balanced_rf" "1.0" "$SEED" "BalancedRF" "no" ""
done

echo "" | tee -a "$LOG_FILE"
echo "=== Submitting EasyEnsemble experiments ===" | tee -a "$LOG_FILE"
for SEED in "${SEEDS[@]}"; do
    submit_job "easy_ensemble" "1.0" "$SEED" "EasyEnsemble" "no" ""
done

echo "" | tee -a "$LOG_FILE"
echo "=== Submitting SMOTE+BalancedRF experiments ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "smote_balanced_rf" "$RATIO" "$SEED" "BalancedRF" "yes" "smote"
    done
done

# =============================================================================
# Summary
# =============================================================================
echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Submission Summary" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Training jobs submitted: $TRAIN_COUNT" | tee -a "$LOG_FILE"
echo "Evaluation jobs submitted: $EVAL_COUNT" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

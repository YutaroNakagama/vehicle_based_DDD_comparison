#!/bin/bash
# =============================================================================
# Imbalance V2 - Submit ONLY missing experiments
# =============================================================================
# Focus on:
# 1. Undersample (RUS, Tomek, ENN) - script bug was fixed
# 2. BalancedRF, EasyEnsemble - were not submitted due to queue limits
# 3. smote_balanced_rf - were cancelled
# =============================================================================

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
LOG_DIR="${PROJECT_DIR}/logs/imbal_v2_fixed"
SCRIPT_DIR="${PROJECT_DIR}/scripts/hpc/logs/imbalance/tmp_jobs"
mkdir -p "$LOG_DIR" "$SCRIPT_DIR"
LOG_FILE="${LOG_DIR}/submit_missing_${TIMESTAMP}.txt"

echo "============================================================" | tee "$LOG_FILE"
echo "Imbalance V2 - Missing Experiments Only" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Configuration
N_TRIALS=75
SEEDS=("42" "123" "456")
RATIOS=("0.1" "0.5" "1.0")

# Use LONG queue primarily (less congested) + DEFAULT
QUEUES=("LONG" "LONG" "DEFAULT" "LONG" "LONG" "SMALL")
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
    
    # Select queue
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
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "============================================================"
echo "[IMBALANCE V2] Training - N_TRIALS=${N_TRIALS}"
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

echo "[IMBALANCE V2] Evaluation: ${TAG}"

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
# Submit ONLY missing experiments
# =============================================================================

echo "" | tee -a "$LOG_FILE"
echo "=== Undersample-RUS (FIXED: use --oversample_method) ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "undersample_rus" "$RATIO" "$SEED" "RF" "yes" "undersample_rus"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== Undersample-Tomek (FIXED) ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "undersample_tomek" "$RATIO" "$SEED" "RF" "yes" "undersample_tomek"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== Undersample-ENN (FIXED) ===" | tee -a "$LOG_FILE"
for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_job "undersample_enn" "$RATIO" "$SEED" "RF" "yes" "undersample_enn"
    done
done

echo "" | tee -a "$LOG_FILE"
echo "=== BalancedRF (was not submitted) ===" | tee -a "$LOG_FILE"
for SEED in "${SEEDS[@]}"; do
    submit_job "balanced_rf" "1.0" "$SEED" "BalancedRF" "no" ""
done

echo "" | tee -a "$LOG_FILE"
echo "=== EasyEnsemble (was not submitted) ===" | tee -a "$LOG_FILE"
for SEED in "${SEEDS[@]}"; do
    submit_job "easy_ensemble" "1.0" "$SEED" "EasyEnsemble" "no" ""
done

echo "" | tee -a "$LOG_FILE"
echo "=== SMOTE+BalancedRF (was cancelled) ===" | tee -a "$LOG_FILE"
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
echo "Submission Summary (Missing experiments only)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Undersample-RUS: 9 jobs" | tee -a "$LOG_FILE"
echo "Undersample-Tomek: 9 jobs" | tee -a "$LOG_FILE"
echo "Undersample-ENN: 9 jobs" | tee -a "$LOG_FILE"
echo "BalancedRF: 3 jobs" | tee -a "$LOG_FILE"
echo "EasyEnsemble: 3 jobs" | tee -a "$LOG_FILE"
echo "SMOTE+BalancedRF: 9 jobs" | tee -a "$LOG_FILE"
echo "---" | tee -a "$LOG_FILE"
echo "Training jobs submitted: $TRAIN_COUNT / 42" | tee -a "$LOG_FILE"
echo "Evaluation jobs submitted: $EVAL_COUNT" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

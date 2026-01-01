#!/bin/bash
# =============================================================================
# N_TRIALS=75 Full Imbalance Experiment Submission
# =============================================================================
# All methods × 3 ratios × 3 seeds with optimized resources
# Distributed across TINY, SINGLE, DEFAULT queues for faster completion
# =============================================================================

# Don't exit on error - continue with other jobs
# set -e
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/imbalance_v2_n75_jobs_${TIMESTAMP}.txt"
PROJECT_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"

mkdir -p logs

echo "============================================================" | tee -a $LOG_FILE
echo "Imbalance V2 Full Experiment Submission (N_TRIALS=75)" | tee -a $LOG_FILE
echo "Timestamp: $TIMESTAMP" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

# N_TRIALS setting
export N_TRIALS_OVERRIDE=75

# Methods and ratios
SAMPLING_METHODS=("smote" "smote_tomek" "smote_enn" "smote_rus")
UNDERSAMPLING_METHODS=("undersample_rus" "undersample_tomek" "undersample_enn")
ENSEMBLE_METHODS=("balanced_rf" "easy_ensemble" "smote_balanced_rf")
RATIOS=("0.1" "0.5" "1.0")
SEEDS=("42" "123" "456")

# Resource settings (optimized)
MEM_SAMPLING="8gb"
MEM_UNDERSAMPLE="4gb"
MEM_ENSEMBLE="8gb"
NCPUS=4
WALLTIME_TRAIN="06:00:00"  # Increased for N_TRIALS=75
WALLTIME_EVAL="00:30:00"   # TINY queue max is 30 min

# Queue distribution strategy
# TINY: 30min, 1CPU, 4GB - use for eval jobs
# SINGLE: 4hr, 4CPU, 32GB - use for small training
# DEFAULT: 8hr, 16CPU, 62GB - use for larger training

train_job_count=0
eval_job_count=0

submit_train_job() {
    local METHOD=$1
    local RATIO=$2
    local SEED=$3
    local MEM=$4
    local QUEUE=$5
    local MODEL_TYPE=$6
    
    local TAG="imbal_v2_${METHOD}_ratio${RATIO//./_}_seed${SEED}"
    local JOB_NAME="imbal_v2_${METHOD}_s${SEED}"
    
    # Create job script
    local JOB_SCRIPT=$(mktemp)
    cat > $JOB_SCRIPT << EOF
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -q ${QUEUE}
#PBS -l select=1:ncpus=${NCPUS}:mem=${MEM}
#PBS -l walltime=${WALLTIME_TRAIN}
#PBS -j oe
#PBS -o ${PROJECT_DIR}/scripts/hpc/logs/

cd ${PROJECT_DIR}
source /home/s2240011/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH
export N_TRIALS_OVERRIDE=75

echo "============================================================"
echo "[IMBALANCE V2] Training with N_TRIALS=75"
echo "============================================================"
echo "METHOD: ${METHOD}"
echo "RATIO: ${RATIO}"
echo "SEED: ${SEED}"
echo "MODEL: ${MODEL_TYPE}"
echo "N_TRIALS: 75"
echo "Python: \$(which python)"
echo "============================================================"

EOF
    
    # Add appropriate training command based on method type
    if [[ "$METHOD" == "baseline" ]]; then
        cat >> $JOB_SCRIPT << EOF
python scripts/python/train.py \\
    --model ${MODEL_TYPE} \\
    --mode pooled \\
    --tag ${TAG} \\
    --seed ${SEED} \\
    --time_stratify_labels
EOF
    elif [[ "$METHOD" == "balanced_rf" ]]; then
        cat >> $JOB_SCRIPT << EOF
python scripts/python/train.py \\
    --model BalancedRF \\
    --mode pooled \\
    --tag ${TAG} \\
    --seed ${SEED} \\
    --time_stratify_labels
EOF
    elif [[ "$METHOD" == "easy_ensemble" ]]; then
        cat >> $JOB_SCRIPT << EOF
python scripts/python/train.py \\
    --model EasyEnsemble \\
    --mode pooled \\
    --tag ${TAG} \\
    --seed ${SEED} \\
    --time_stratify_labels
EOF
    elif [[ "$METHOD" == "smote_balanced_rf" ]]; then
        cat >> $JOB_SCRIPT << EOF
python scripts/python/train.py \\
    --model BalancedRF \\
    --mode pooled \\
    --tag ${TAG} \\
    --seed ${SEED} \\
    --time_stratify_labels \\
    --use_oversampling \\
    --oversample_method smote \\
    --target_ratio ${RATIO}
EOF
    elif [[ "$METHOD" == smote* ]]; then
        cat >> $JOB_SCRIPT << EOF
python scripts/python/train.py \\
    --model ${MODEL_TYPE} \\
    --mode pooled \\
    --tag ${TAG} \\
    --seed ${SEED} \\
    --time_stratify_labels \\
    --use_oversampling \\
    --oversample_method ${METHOD} \\
    --target_ratio ${RATIO}
EOF
    elif [[ "$METHOD" == undersample* ]]; then
        cat >> $JOB_SCRIPT << EOF
python scripts/python/train.py \\
    --model ${MODEL_TYPE} \\
    --mode pooled \\
    --tag ${TAG} \\
    --seed ${SEED} \\
    --time_stratify_labels \\
    --use_undersampling \\
    --undersample_method ${METHOD} \\
    --target_ratio ${RATIO}
EOF
    fi
    
    cat >> $JOB_SCRIPT << EOF

echo "[DONE] Training complete: ${TAG}"
EOF
    
    TRAIN_JOB_ID=$(qsub $JOB_SCRIPT 2>&1)
    QSUB_STATUS=$?
    rm -f $JOB_SCRIPT
    
    if [[ $QSUB_STATUS -ne 0 ]] || [[ ! "$TRAIN_JOB_ID" =~ ^[0-9]+ ]]; then
        echo "TRAIN: ${TAG} -> FAILED (qsub error: ${TRAIN_JOB_ID})" | tee -a $LOG_FILE
        return 1
    fi
    
    # Clean job ID
    TRAIN_JOB_ID=$(echo "$TRAIN_JOB_ID" | tr -d '\n' | tr -d ' ')
    
    echo "TRAIN: ${TAG} -> ${TRAIN_JOB_ID} (${QUEUE})" | tee -a $LOG_FILE
    train_job_count=$((train_job_count + 1))
    
    # Submit evaluation job with dependency
    submit_eval_job "$METHOD" "$RATIO" "$SEED" "$MODEL_TYPE" "$TRAIN_JOB_ID"
}

submit_eval_job() {
    local METHOD=$1
    local RATIO=$2
    local SEED=$3
    local MODEL_TYPE=$4
    local TRAIN_JOB_ID=$5
    
    # Clean the job ID (remove whitespace/newlines)
    TRAIN_JOB_ID=$(echo "$TRAIN_JOB_ID" | tr -d '\n' | tr -d ' ')
    
    if [[ -z "$TRAIN_JOB_ID" ]]; then
        echo "  EVAL: SKIPPED (no train job ID)" | tee -a $LOG_FILE
        return 1
    fi
    
    local TAG="imbal_v2_${METHOD}_ratio${RATIO//./_}_seed${SEED}"
    local JOB_NAME="eval_${METHOD}_s${SEED}"
    
    local JOB_SCRIPT=$(mktemp)
    cat > $JOB_SCRIPT << EOF
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -q TINY
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=${WALLTIME_EVAL}
#PBS -j oe
#PBS -o ${PROJECT_DIR}/scripts/hpc/logs/
#PBS -W depend=afterok:${TRAIN_JOB_ID}

cd ${PROJECT_DIR}
source /home/s2240011/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH

echo "[EVAL] Starting evaluation for ${TAG}"

python scripts/python/evaluate.py \\
    --model ${MODEL_TYPE} \\
    --mode pooled \\
    --tag ${TAG}

echo "[DONE] Evaluation complete: ${TAG}"
EOF
    
    EVAL_JOB_ID=$(qsub $JOB_SCRIPT 2>&1)
    QSUB_STATUS=$?
    rm -f $JOB_SCRIPT
    
    if [[ $QSUB_STATUS -ne 0 ]] || [[ ! "$EVAL_JOB_ID" =~ ^[0-9]+ ]]; then
        echo "  EVAL: ${TAG} -> FAILED (qsub error: ${EVAL_JOB_ID})" | tee -a $LOG_FILE
        return 1
    fi
    
    echo "  EVAL: ${TAG} -> ${EVAL_JOB_ID} (TINY, depends on ${TRAIN_JOB_ID})" | tee -a $LOG_FILE
    eval_job_count=$((eval_job_count + 1))
}

# =============================================================================
# Submit all jobs with queue distribution
# =============================================================================

echo "" | tee -a $LOG_FILE
echo "Submitting BASELINE experiments..." | tee -a $LOG_FILE
for SEED in "${SEEDS[@]}"; do
    submit_train_job "baseline" "1.0" "$SEED" "8gb" "SINGLE" "RF"
done

echo "" | tee -a $LOG_FILE
echo "Submitting SAMPLING (SMOTE-based) experiments..." | tee -a $LOG_FILE
queue_idx=0
QUEUES=("SINGLE" "SINGLE" "DEFAULT")  # Rotate through queues
for METHOD in "${SAMPLING_METHODS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            QUEUE=${QUEUES[$((queue_idx % 3))]}
            submit_train_job "$METHOD" "$RATIO" "$SEED" "$MEM_SAMPLING" "$QUEUE" "RF"
            ((queue_idx++))
        done
    done
done

echo "" | tee -a $LOG_FILE
echo "Submitting UNDERSAMPLING experiments..." | tee -a $LOG_FILE
queue_idx=0
for METHOD in "${UNDERSAMPLING_METHODS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            QUEUE=${QUEUES[$((queue_idx % 3))]}
            submit_train_job "$METHOD" "$RATIO" "$SEED" "$MEM_UNDERSAMPLE" "$QUEUE" "RF"
            ((queue_idx++))
        done
    done
done

echo "" | tee -a $LOG_FILE
echo "Submitting ENSEMBLE experiments..." | tee -a $LOG_FILE
for SEED in "${SEEDS[@]}"; do
    submit_train_job "balanced_rf" "1.0" "$SEED" "$MEM_ENSEMBLE" "SINGLE" "BalancedRF"
done

for SEED in "${SEEDS[@]}"; do
    submit_train_job "easy_ensemble" "1.0" "$SEED" "$MEM_ENSEMBLE" "SINGLE" "EasyEnsemble"
done

for RATIO in "${RATIOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        submit_train_job "smote_balanced_rf" "$RATIO" "$SEED" "$MEM_ENSEMBLE" "DEFAULT" "BalancedRF"
    done
done

# =============================================================================
# Summary
# =============================================================================

echo "" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "SUBMISSION COMPLETE" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "Training jobs submitted: ${train_job_count}" | tee -a $LOG_FILE
echo "Evaluation jobs submitted: ${eval_job_count}" | tee -a $LOG_FILE
echo "Total jobs: $((train_job_count + eval_job_count))" | tee -a $LOG_FILE
echo "Log file: ${LOG_FILE}" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

# Check queue status
echo "" | tee -a $LOG_FILE
echo "Current queue status:" | tee -a $LOG_FILE
qstat -u s2240011 | tee -a $LOG_FILE

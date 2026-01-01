#!/bin/bash
# =============================================================================
# Submit remaining jobs that failed due to queue limits
# Run after some jobs complete to free up queue slots
# =============================================================================

PROJECT_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
LOG_FILE="logs/imbalance_v2_n75_remaining_$(date +%Y%m%d_%H%M%S).txt"
WALLTIME_TRAIN="06:00:00"
WALLTIME_EVAL="00:30:00"  # TINY queue max is 30 minutes

echo "============================================================" | tee -a $LOG_FILE
echo "Submitting remaining N_TRIALS=75 jobs" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE

submit_train_job() {
    local METHOD=$1
    local RATIO=$2
    local SEED=$3
    local MEM=$4
    local QUEUE=$5
    local MODEL_TYPE=$6
    
    local TAG="imbal_v2_${METHOD}_ratio${RATIO//./_}_seed${SEED}"
    local JOB_NAME="imbal_v2_${METHOD}_s${SEED}"
    
    local JOB_SCRIPT=$(mktemp)
    cat > $JOB_SCRIPT << EOF
#!/bin/bash
#PBS -N ${JOB_NAME}
#PBS -q ${QUEUE}
#PBS -l select=1:ncpus=4:mem=${MEM}
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
echo "METHOD: ${METHOD}, RATIO: ${RATIO}, SEED: ${SEED}"
echo "============================================================"

EOF
    
    if [[ "$METHOD" == "balanced_rf" ]]; then
        cat >> $JOB_SCRIPT << EOF
python scripts/python/train/train.py \\
    --model BalancedRF \\
    --mode pooled \\
    --tag ${TAG} \\
    --seed ${SEED} \\
    --time_stratify_labels
EOF
    elif [[ "$METHOD" == "easy_ensemble" ]]; then
        cat >> $JOB_SCRIPT << EOF
python scripts/python/train/train.py \\
    --model EasyEnsemble \\
    --mode pooled \\
    --tag ${TAG} \\
    --seed ${SEED} \\
    --time_stratify_labels
EOF
    elif [[ "$METHOD" == undersample* ]]; then
        cat >> $JOB_SCRIPT << EOF
python scripts/python/train/train.py \\
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
        echo "TRAIN: ${TAG} -> FAILED (${TRAIN_JOB_ID})" | tee -a $LOG_FILE
        return 1
    fi
    
    TRAIN_JOB_ID=$(echo "$TRAIN_JOB_ID" | tr -d '\n')
    echo "TRAIN: ${TAG} -> ${TRAIN_JOB_ID} (${QUEUE})" | tee -a $LOG_FILE
    
    # Submit eval job
    local EVAL_SCRIPT=$(mktemp)
    cat > $EVAL_SCRIPT << EOF
#!/bin/bash
#PBS -N eval_${METHOD}_s${SEED}
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

python scripts/python/evaluation/evaluate.py --model ${MODEL_TYPE} --mode pooled --tag ${TAG}
echo "[DONE] Evaluation: ${TAG}"
EOF
    
    EVAL_JOB_ID=$(qsub $EVAL_SCRIPT 2>&1)
    rm -f $EVAL_SCRIPT
    echo "  EVAL: ${TAG} -> ${EVAL_JOB_ID}" | tee -a $LOG_FILE
}

echo "" | tee -a $LOG_FILE
echo "--- undersample_enn jobs ---" | tee -a $LOG_FILE
submit_train_job "undersample_enn" "0.1" "123" "4gb" "DEFAULT" "RF"
submit_train_job "undersample_enn" "0.5" "42" "4gb" "DEFAULT" "RF"
submit_train_job "undersample_enn" "0.5" "123" "4gb" "DEFAULT" "RF"
submit_train_job "undersample_enn" "1.0" "42" "4gb" "DEFAULT" "RF"
submit_train_job "undersample_enn" "1.0" "123" "4gb" "DEFAULT" "RF"

echo "" | tee -a $LOG_FILE
echo "--- balanced_rf jobs ---" | tee -a $LOG_FILE
submit_train_job "balanced_rf" "1.0" "42" "8gb" "DEFAULT" "BalancedRF"
submit_train_job "balanced_rf" "1.0" "123" "8gb" "DEFAULT" "BalancedRF"
submit_train_job "balanced_rf" "1.0" "456" "8gb" "DEFAULT" "BalancedRF"

echo "" | tee -a $LOG_FILE
echo "--- easy_ensemble jobs ---" | tee -a $LOG_FILE
submit_train_job "easy_ensemble" "1.0" "42" "8gb" "DEFAULT" "EasyEnsemble"
submit_train_job "easy_ensemble" "1.0" "123" "8gb" "DEFAULT" "EasyEnsemble"
submit_train_job "easy_ensemble" "1.0" "456" "8gb" "DEFAULT" "EasyEnsemble"

echo "" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "Done. Check queue: qstat -u s2240011" | tee -a $LOG_FILE

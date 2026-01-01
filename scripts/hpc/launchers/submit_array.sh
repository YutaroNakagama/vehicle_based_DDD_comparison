#!/bin/bash
# =============================================================================
# Imbalance V2 - Array Job Submission for remaining experiments
# =============================================================================
# Uses array jobs to efficiently submit all remaining experiments
# Distributes across LONG-L, LARGE, XLARGE queues (less congested)
# =============================================================================

set -euo pipefail

PROJECT_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs/imbal_v2_fixed"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/array_submit_${TIMESTAMP}.txt"

# Define all remaining experiments as indexed array
# Format: METHOD:RATIO:SEED:MODEL:OVERSAMPLE_FLAG:OVERSAMPLE_METHOD
declare -a EXPERIMENTS=(
    # Undersample-Tomek remaining (5)
    "undersample_tomek:0.5:123:RF:yes:undersample_tomek"
    "undersample_tomek:0.5:456:RF:yes:undersample_tomek"
    "undersample_tomek:1.0:42:RF:yes:undersample_tomek"
    "undersample_tomek:1.0:123:RF:yes:undersample_tomek"
    "undersample_tomek:1.0:456:RF:yes:undersample_tomek"
    # Undersample-ENN (9)
    "undersample_enn:0.1:42:RF:yes:undersample_enn"
    "undersample_enn:0.1:123:RF:yes:undersample_enn"
    "undersample_enn:0.1:456:RF:yes:undersample_enn"
    "undersample_enn:0.5:42:RF:yes:undersample_enn"
    "undersample_enn:0.5:123:RF:yes:undersample_enn"
    "undersample_enn:0.5:456:RF:yes:undersample_enn"
    "undersample_enn:1.0:42:RF:yes:undersample_enn"
    "undersample_enn:1.0:123:RF:yes:undersample_enn"
    "undersample_enn:1.0:456:RF:yes:undersample_enn"
    # BalancedRF (3)
    "balanced_rf:1.0:42:BalancedRF:no:"
    "balanced_rf:1.0:123:BalancedRF:no:"
    "balanced_rf:1.0:456:BalancedRF:no:"
    # EasyEnsemble (3)
    "easy_ensemble:1.0:42:EasyEnsemble:no:"
    "easy_ensemble:1.0:123:EasyEnsemble:no:"
    "easy_ensemble:1.0:456:EasyEnsemble:no:"
    # SMOTE+BalancedRF (9)
    "smote_balanced_rf:0.1:42:BalancedRF:yes:smote"
    "smote_balanced_rf:0.1:123:BalancedRF:yes:smote"
    "smote_balanced_rf:0.1:456:BalancedRF:yes:smote"
    "smote_balanced_rf:0.5:42:BalancedRF:yes:smote"
    "smote_balanced_rf:0.5:123:BalancedRF:yes:smote"
    "smote_balanced_rf:0.5:456:BalancedRF:yes:smote"
    "smote_balanced_rf:1.0:42:BalancedRF:yes:smote"
    "smote_balanced_rf:1.0:123:BalancedRF:yes:smote"
    "smote_balanced_rf:1.0:456:BalancedRF:yes:smote"
)

# Save experiment list to file for array job to read
EXPERIMENT_LIST="${PROJECT_DIR}/scripts/hpc/logs/imbalance/experiment_list.txt"
printf "%s\n" "${EXPERIMENTS[@]}" > "$EXPERIMENT_LIST"

NUM_EXPERIMENTS=${#EXPERIMENTS[@]}
echo "Total experiments to submit: $NUM_EXPERIMENTS" | tee "$LOG_FILE"

# Create array job script for training
TRAIN_ARRAY_SCRIPT="${PROJECT_DIR}/scripts/hpc/launchers/train_array.sh"

cat > "$TRAIN_ARRAY_SCRIPT" << 'TRAINEOF'
#!/bin/bash
#PBS -N imbal_v2_array
#PBS -q LONG-L
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=168:00:00
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -J 1-29

set -euo pipefail

PROJECT_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
EXPERIMENT_LIST="${PROJECT_DIR}/scripts/hpc/logs/imbalance/experiment_list.txt"

source /home/s2240011/conda/etc/profile.d/conda.sh
conda activate python310

cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export N_TRIALS_OVERRIDE=75
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Get experiment parameters from list
LINE=$(sed -n "${PBS_ARRAY_INDEX}p" "$EXPERIMENT_LIST")
IFS=':' read -r METHOD RATIO SEED MODEL OVERSAMPLE_FLAG OVERSAMPLE_METHOD <<< "$LINE"

RATIO_SAFE="${RATIO//./_}"
TAG="imbal_v2_${METHOD}_ratio${RATIO_SAFE}_seed${SEED}"

echo "============================================================"
echo "[IMBALANCE V2 ARRAY] Training"
echo "============================================================"
echo "PBS_ARRAY_INDEX: ${PBS_ARRAY_INDEX}"
echo "METHOD: ${METHOD}"
echo "MODEL: ${MODEL}"
echo "RATIO: ${RATIO}"
echo "SEED: ${SEED}"
echo "TAG: ${TAG}"
echo "OVERSAMPLE: ${OVERSAMPLE_FLAG}"
echo "============================================================"

if [[ "$OVERSAMPLE_FLAG" == "yes" ]]; then
    python scripts/python/train/train.py \
        --model "$MODEL" \
        --mode pooled \
        --tag "$TAG" \
        --seed "$SEED" \
        --time_stratify_labels \
        --use_oversampling \
        --oversample_method "$OVERSAMPLE_METHOD" \
        --target_ratio "$RATIO"
else
    python scripts/python/train/train.py \
        --model "$MODEL" \
        --mode pooled \
        --tag "$TAG" \
        --seed "$SEED" \
        --time_stratify_labels
fi

echo "[DONE] Training complete: ${TAG}"
TRAINEOF

chmod +x "$TRAIN_ARRAY_SCRIPT"

echo "Submitting array job for training..." | tee -a "$LOG_FILE"

# Submit array job
TRAIN_JOB_ID=$(qsub "$TRAIN_ARRAY_SCRIPT" 2>&1)

if [[ "$TRAIN_JOB_ID" =~ ^[0-9]+ ]]; then
    echo "TRAIN ARRAY: $TRAIN_JOB_ID" | tee -a "$LOG_FILE"
    TRAIN_JOB_ID_BASE=$(echo "$TRAIN_JOB_ID" | grep -oE '^[0-9]+')
    
    # Create evaluation array job that depends on training
    EVAL_ARRAY_SCRIPT="${PROJECT_DIR}/scripts/hpc/launchers/eval_array.sh"
    
    cat > "$EVAL_ARRAY_SCRIPT" << EVALEOF
#!/bin/bash
#PBS -N ev_imbal_v2_arr
#PBS -q TINY
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -J 1-29
#PBS -W depend=afterok:${TRAIN_JOB_ID}

set -euo pipefail

PROJECT_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
EXPERIMENT_LIST="\${PROJECT_DIR}/scripts/hpc/logs/imbalance/experiment_list.txt"

source /home/s2240011/conda/etc/profile.d/conda.sh
conda activate python310

cd "\$PROJECT_DIR"
export PYTHONPATH="\${PROJECT_DIR}:\${PYTHONPATH:-}"

LINE=\$(sed -n "\${PBS_ARRAY_INDEX}p" "\$EXPERIMENT_LIST")
IFS=':' read -r METHOD RATIO SEED MODEL OVERSAMPLE_FLAG OVERSAMPLE_METHOD <<< "\$LINE"

RATIO_SAFE="\${RATIO//./_}"
TAG="imbal_v2_\${METHOD}_ratio\${RATIO_SAFE}_seed\${SEED}"

echo "[IMBALANCE V2] Evaluation: \${TAG}"

python scripts/python/evaluation/evaluate.py \
    --model "\$MODEL" \
    --mode pooled \
    --tag "\$TAG"

echo "[DONE] Evaluation complete: \${TAG}"
EVALEOF

    chmod +x "$EVAL_ARRAY_SCRIPT"
    
    EVAL_JOB_ID=$(qsub "$EVAL_ARRAY_SCRIPT" 2>&1)
    
    if [[ "$EVAL_JOB_ID" =~ ^[0-9]+ ]]; then
        echo "EVAL ARRAY: $EVAL_JOB_ID (depends on training)" | tee -a "$LOG_FILE"
    else
        echo "[ERROR] Eval array submission failed: $EVAL_JOB_ID" | tee -a "$LOG_FILE"
    fi
else
    echo "[ERROR] Train array submission failed: $TRAIN_JOB_ID" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Array Job Submission Complete" | tee -a "$LOG_FILE"
echo "Experiments: $NUM_EXPERIMENTS" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

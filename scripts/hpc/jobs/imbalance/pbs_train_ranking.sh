#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
# Note: -N, -l select, -l walltime, -q are passed dynamically via qsub options

# ============================================================
# Unified Ranking-based Training Script
# For source_only/target_only mode with ranking-based subject selection
# ============================================================
# Usage:
#   qsub -N smote_knn_top8 -l select=1:ncpus=4:mem=8gb -l walltime=10:00:00 -q SINGLE \
#        -v METHOD=smote,MODE=source_only,SUBJECT_FILE=/path/to/subjects.txt,TAG=my_tag pbs_train_ranking.sh
#
# Environment Variables:
#   METHOD       : Sampling method (smote, smote_subjectwise, smote_balanced_rf)
#   MODE         : Training mode (source_only, target_only)
#   SUBJECT_FILE : Path to subject list file (required for non-pooled modes)
#   RATIO        : Target ratio for oversampling (default: 0.33)
#   SEED         : Random seed (default: 42)
#   TAG          : Experiment tag (required)
#   N_TRIALS     : Override N_TRIALS_OVERRIDE (default: 50)
# ============================================================
set -euo pipefail

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

# Thread optimization for HPC
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export JOBLIB_MULTIPROCESSING=0
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export N_TRIALS_OVERRIDE="${N_TRIALS:-50}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/s2240011/git/ddd/vehicle_based_DDD_comparison}"
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

# === Parameters ===
METHOD="${METHOD:-smote}"
MODE="${MODE:-source_only}"
SUBJECT_FILE="${SUBJECT_FILE:-}"
RATIO="${RATIO:-0.33}"
SEED="${SEED:-42}"
TAG="${TAG:-ranking_test}"

# Auto-select model based on method
case "$METHOD" in
    smote_balanced_rf)
        MODEL="BalancedRF"
        ;;
    *)
        MODEL="RF"
        ;;
esac

echo "============================================================"
echo "[RANKING TRAINING] ${METHOD^^}"
echo "============================================================"
echo "MODEL: $MODEL"
echo "METHOD: $METHOD"
echo "MODE: $MODE"
echo "SUBJECT_FILE: $SUBJECT_FILE"
echo "RATIO: $RATIO"
echo "SEED: $SEED"
echo "TAG: $TAG"
echo "N_TRIALS: $N_TRIALS_OVERRIDE"
echo "JOBID: $PBS_JOBID"
echo "============================================================"

# === Build Command ===
CMD="python scripts/python/train/train.py \
    --model $MODEL \
    --mode $MODE \
    --seed $SEED \
    --time_stratify_labels \
    --tag $TAG"

# Add method-specific options
case "$METHOD" in
    smote)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO"
        ;;
    smote_subjectwise)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO --subject_wise_oversampling"
        ;;
    smote_balanced_rf)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO"
        ;;
    *)
        echo "[ERROR] Unknown method: $METHOD"
        echo "Supported methods: smote, smote_subjectwise, smote_balanced_rf"
        exit 1
        ;;
esac

# Add target_file if mode is not pooled
if [[ "$MODE" != "pooled" && -n "$SUBJECT_FILE" ]]; then
    if [[ -f "$SUBJECT_FILE" ]]; then
        CMD="$CMD --target_file $SUBJECT_FILE"
        echo "[INFO] Using subject file: $SUBJECT_FILE"
        echo "[INFO] Subject count: $(wc -l < "$SUBJECT_FILE")"
    else
        echo "[WARN] Subject file not found: $SUBJECT_FILE"
        echo "[INFO] Falling back to default subjects"
    fi
fi

echo ""
echo "[CMD] $CMD"
echo ""

eval $CMD

echo "=== RANKING TRAINING DONE ==="

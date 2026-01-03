#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
# Note: -N, -l select, -l walltime, -q are passed dynamically via qsub options

# ============================================================
# Unified Imbalance Training Script
# Supports all imbalance methods with configurable parameters
# ============================================================
# Usage:
#   qsub -N job_name -l select=1:ncpus=4:mem=8gb -l walltime=10:00:00 -q SINGLE \
#        -v METHOD=smote,SEED=42,RATIO=0.33,TAG=my_experiment pbs_train.sh
#
# Environment Variables:
#   METHOD    : Sampling method (required)
#               - baseline, smote, smote_subjectwise, smote_tomek, smote_enn,
#                 smote_rus, balanced_rf, easy_ensemble, smote_balanced_rf,
#                 jitter_scale, undersample_enn, undersample_rus, undersample_tomek
#   MODEL     : Model type (default: RF, or BalancedRF/EasyEnsemble for ensemble methods)
#   SEED      : Random seed (default: 42)
#   RATIO     : Target ratio for oversampling (default: 0.33)
#   TAG       : Experiment tag (default: auto-generated)
#   MODE      : Training mode (default: pooled)
#   N_TRIALS  : Override N_TRIALS_OVERRIDE (default: 50)
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

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

# === Parameters ===
METHOD="${METHOD:-baseline}"
SEED="${SEED:-42}"
RATIO="${RATIO:-0.33}"
MODE="${MODE:-pooled}"

# Auto-select model based on method
case "$METHOD" in
    balanced_rf|smote_balanced_rf)
        MODEL="${MODEL:-BalancedRF}"
        ;;
    easy_ensemble)
        MODEL="${MODEL:-EasyEnsemble}"
        ;;
    *)
        MODEL="${MODEL:-RF}"
        ;;
esac

# Auto-generate tag if not provided
if [[ -z "${TAG:-}" ]]; then
    TAG="imbal_v2_${METHOD}_seed${SEED}"
fi

echo "============================================================"
echo "[IMBALANCE TRAINING] ${METHOD^^}"
echo "============================================================"
echo "MODEL: $MODEL"
echo "METHOD: $METHOD"
echo "SEED: $SEED"
echo "RATIO: $RATIO"
echo "MODE: $MODE"
echo "TAG: $TAG"
echo "N_TRIALS: $N_TRIALS_OVERRIDE"
echo "JOBID: $PBS_JOBID"
echo "============================================================"

# === Build Command ===
CMD="python scripts/python/train/train.py \
    --model $MODEL \
    --mode $MODE \
    --subject_wise_split \
    --seed $SEED \
    --time_stratify_labels \
    --tag $TAG"

# Add method-specific options
case "$METHOD" in
    baseline)
        # No oversampling
        ;;
    balanced_rf|easy_ensemble)
        # Internal balancing, no oversampling needed
        ;;
    smote)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO"
        ;;
    smote_subjectwise)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO --subject_wise_oversampling"
        ;;
    smote_tomek)
        CMD="$CMD --use_oversampling --oversample_method smote_tomek --target_ratio $RATIO"
        ;;
    smote_enn)
        CMD="$CMD --use_oversampling --oversample_method smote_enn --target_ratio $RATIO"
        ;;
    smote_rus)
        CMD="$CMD --use_oversampling --oversample_method smote_rus --target_ratio $RATIO"
        ;;
    smote_balanced_rf)
        CMD="$CMD --use_oversampling --oversample_method smote --target_ratio $RATIO"
        ;;
    jitter_scale|jitter_scale_adaptive)
        CMD="$CMD --use_oversampling --oversample_method jitter_scale --target_ratio $RATIO"
        ;;
    undersample_enn)
        CMD="$CMD --use_oversampling --oversample_method undersample_enn"
        ;;
    undersample_rus)
        CMD="$CMD --use_oversampling --oversample_method undersample_rus --target_ratio $RATIO"
        ;;
    undersample_tomek)
        CMD="$CMD --use_oversampling --oversample_method undersample_tomek"
        ;;
    *)
        echo "[ERROR] Unknown method: $METHOD"
        echo "Supported methods: baseline, smote, smote_subjectwise, smote_tomek, smote_enn,"
        echo "                   smote_rus, balanced_rf, easy_ensemble, smote_balanced_rf,"
        echo "                   jitter_scale, undersample_enn, undersample_rus, undersample_tomek"
        exit 1
        ;;
esac

echo ""
echo "[CMD] $CMD"
echo ""

eval $CMD

echo "=== TRAINING DONE ==="

#!/bin/bash
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe
# Note: -N, -l select, -l walltime, -q are passed dynamically via qsub options

# ==============================================================================
# Unified Domain Analysis Training Script
# ==============================================================================
# Supports two experiment types:
#   1. ranking_comparison: Compare ranking methods (knn, lof, etc.)
#   2. smote_comparison: Compare SMOTE methods with ranking-based subject selection
#
# Usage Examples:
#   # Ranking comparison (which ranking method is best?)
#   qsub -N rank_knn -l select=1:ncpus=4:mem=16gb -l walltime=12:00:00 -q SINGLE \
#        -v EXPERIMENT=ranking_comparison,RANKING_METHOD=knn,DISTANCE_METRIC=mmd,DOMAIN_LEVEL=out_domain pbs_train.sh
#
#   # SMOTE comparison with ranking (which SMOTE method is best for ranked subjects?)
#   qsub -N smote_knn -l select=1:ncpus=4:mem=8gb -l walltime=10:00:00 -q SINGLE \
#        -v EXPERIMENT=smote_comparison,METHOD=smote,SUBJECT_FILE=/path/to/subjects.txt pbs_train.sh
#
# Environment Variables:
#   EXPERIMENT      : ranking_comparison | smote_comparison (required)
#   MODE            : source_only, target_only (default: source_only)
#   SEED            : Random seed (default: 42)
#   N_TRIALS        : Optuna trials (default: 50)
#
#   [ranking_comparison only]
#   RANKING_METHOD  : knn, lof, mean_distance, median_distance, isolation_forest, centroid_umap
#   DISTANCE_METRIC : mmd, dtw, wasserstein
#   DOMAIN_LEVEL    : out_domain, in_domain, cross_domain
#
#   [smote_comparison only]
#   METHOD          : smote, smote_subjectwise, smote_balanced_rf
#   SUBJECT_FILE    : Path to subject list file
#   RATIO           : Target ratio for oversampling (default: 0.33)
#   TAG             : Experiment tag (auto-generated if not set)
# ==============================================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Environment setup
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

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

# Common parameters
EXPERIMENT="${EXPERIMENT:-}"
MODE="${MODE:-source_only}"
SEED="${SEED:-42}"
export N_TRIALS_OVERRIDE="${N_TRIALS:-50}"

# Validate experiment type
if [[ -z "$EXPERIMENT" ]]; then
    echo "[ERROR] EXPERIMENT is required (ranking_comparison | smote_comparison)"
    exit 1
fi

echo "============================================================"
echo "[DOMAIN ANALYSIS] ${EXPERIMENT^^}"
echo "============================================================"
echo "MODE: $MODE"
echo "SEED: $SEED"
echo "N_TRIALS: $N_TRIALS_OVERRIDE"
echo "JOBID: $PBS_JOBID"
echo "============================================================"

case "$EXPERIMENT" in
    ranking_comparison)
        # Parameters for ranking comparison
        RANKING_METHOD="${RANKING_METHOD:-knn}"
        DISTANCE_METRIC="${DISTANCE_METRIC:-mmd}"
        DOMAIN_LEVEL="${DOMAIN_LEVEL:-out_domain}"
        
        echo "RANKING_METHOD: $RANKING_METHOD"
        echo "DISTANCE_METRIC: $DISTANCE_METRIC"
        echo "DOMAIN_LEVEL: $DOMAIN_LEVEL"
        
        # Build subject list path
        SUBJECT_FILE="results/analysis/exp2_domain_shift/distance/subject-wise/ranks/ranks29/${RANKING_METHOD}/${DISTANCE_METRIC}_${DOMAIN_LEVEL}.txt"
        
        if [[ ! -f "$SUBJECT_FILE" ]]; then
            echo "[ERROR] Subject list not found: $SUBJECT_FILE"
            exit 1
        fi
        
        TAG="rank_cmp_${RANKING_METHOD}_${DISTANCE_METRIC}_${DOMAIN_LEVEL}_s${SEED}"
        MODEL="RF"
        
        echo "[INFO] Using subject file: $SUBJECT_FILE"
        echo "[INFO] Subject count: $(wc -l < "$SUBJECT_FILE")"
        echo "[INFO] Tag: $TAG"
        
        # Build command (no oversampling)
        CMD="python scripts/python/train/train.py \
            --model $MODEL \
            --mode $MODE \
            --seed $SEED \
            --target_file $SUBJECT_FILE \
            --tag $TAG"
        ;;
        
    smote_comparison)
        # Parameters for SMOTE comparison
        METHOD="${METHOD:-smote}"
        SUBJECT_FILE="${SUBJECT_FILE:-}"
        RATIO="${RATIO:-0.33}"
        TAG="${TAG:-}"
        
        echo "METHOD: $METHOD"
        echo "SUBJECT_FILE: $SUBJECT_FILE"
        echo "RATIO: $RATIO"
        
        # Auto-select model based on method
        case "$METHOD" in
            smote_balanced_rf)
                MODEL="BalancedRF"
                ;;
            *)
                MODEL="RF"
                ;;
        esac
        
        # Auto-generate tag if not provided
        if [[ -z "$TAG" ]]; then
            TAG="smote_rank_${METHOD}_s${SEED}"
        fi
        
        echo "[INFO] Model: $MODEL"
        echo "[INFO] Tag: $TAG"
        
        # Build base command
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
                echo "Supported: smote, smote_subjectwise, smote_balanced_rf"
                exit 1
                ;;
        esac
        
        # Add target_file if provided
        if [[ -n "$SUBJECT_FILE" ]]; then
            if [[ -f "$SUBJECT_FILE" ]]; then
                CMD="$CMD --target_file $SUBJECT_FILE"
                echo "[INFO] Using subject file: $SUBJECT_FILE"
                echo "[INFO] Subject count: $(wc -l < "$SUBJECT_FILE")"
            else
                echo "[WARN] Subject file not found: $SUBJECT_FILE"
                exit 1
            fi
        fi
        ;;
        
    *)
        echo "[ERROR] Unknown experiment type: $EXPERIMENT"
        echo "Supported: ranking_comparison, smote_comparison"
        exit 1
        ;;
esac

echo ""
echo "[CMD] $CMD"
echo ""

eval $CMD

echo "============================================================"
echo "[INFO] Training Completed at $(date)"
echo "============================================================"

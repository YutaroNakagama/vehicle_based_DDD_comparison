#!/bin/bash
# =============================================================================
# Subject-wise SMOTE experiment (login node execution)
# =============================================================================
# Usage:
#   ./scripts/hpc/launchers/run_subject_wise_smote.sh [OPTIONS]
#
# Options:
#   --ratio <float>     Target minority/majority ratio (default: 0.5)
#   --model <name>      Model name: RF, BalancedRF, EasyEnsemble (default: RF)
#   --trials <int>      Number of Optuna trials (default: 10)
#   --seed <int>        Random seed (default: 42)
#   --tag <string>      Experiment tag suffix (default: subject_wise_smote)
#
# Example:
#   ./scripts/hpc/launchers/run_subject_wise_smote.sh --ratio 0.5 --trials 20
# =============================================================================

set -e

# --- Default parameters ---
RATIO=0.5
MODEL=RF
TRIALS=10
SEED=42
TAG="subject_wise_smote"

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --ratio)
            RATIO="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --- Setup environment ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=============================================="
echo " Subject-wise SMOTE Experiment"
echo "=============================================="
echo " Model:        $MODEL"
echo " Mode:         pooled"
echo " Ratio:        $RATIO"
echo " Optuna:       $TRIALS trials"
echo " Seed:         $SEED"
echo " Tag:          ${TAG}_ratio${RATIO}"
echo " Project:      $PROJECT_ROOT"
echo "=============================================="

# --- Activate conda ---
if [ -f ~/conda/etc/profile.d/conda.sh ]; then
    source ~/conda/etc/profile.d/conda.sh
    conda activate python310
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate python310
else
    echo "Warning: conda not found, using system Python"
fi

# --- Set environment variables ---
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export N_TRIALS_OVERRIDE="$TRIALS"

# --- Run training ---
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..."
echo ""

python scripts/python/train/train.py \
    --model "$MODEL" \
    --mode pooled \
    --subject_wise_split \
    --use_oversampling \
    --oversample_method smote \
    --target_ratio "$RATIO" \
    --subject_wise_oversampling \
    --seed "$SEED" \
    --tag "${TAG}_ratio${RATIO}"

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed!"
echo "=============================================="

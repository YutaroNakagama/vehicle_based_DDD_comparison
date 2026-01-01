#!/bin/bash
#PBS -N imbal_v2_ensemble
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

# ============================================================
# Imbalance Comparison V2: Ensemble Evaluation
# Combine predictions from multiple models
# ============================================================
set -euo pipefail

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export JOBLIB_MULTIPROCESSING=0

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export PBS_JOBID="${PBS_JOBID:-manual_$(date +%Y%m%d_%H%M%S)}"

cd "$PROJECT_ROOT"

# Seed for reproducibility
SEED="${SEED:-42}"

echo "============================================================"
echo "[IMBALANCE V2] Ensemble Evaluation"
echo "============================================================"
echo "JOB_ID: $PBS_JOBID"
echo "============================================================"

# Training Job IDs (from V2 experiments)
BASELINE_JOBID="14468417"
SMOTE_TOMEK_JOBID="14468418"
SMOTE_ENN_JOBID="14468419"
BALANCED_RF_JOBID="14468420"
SMOTE_RUS_JOBID="14468421"
EASY_ENSEMBLE_JOBID="14468501"

# Ensemble 1: All 6 methods (best F2)
echo ""
echo "=== Ensemble: All 6 Methods ==="
python scripts/python/archive/ensemble_evaluate.py \
    --models RF RF RF RF BalancedRF EasyEnsemble \
    --tags imbal_v2_baseline imbal_v2_smote_tomek imbal_v2_smote_enn imbal_v2_smote_rus imbal_v2_balanced_rf imbal_v2_easy_ensemble \
    --jobids $BASELINE_JOBID $SMOTE_TOMEK_JOBID $SMOTE_ENN_JOBID $SMOTE_RUS_JOBID $BALANCED_RF_JOBID $EASY_ENSEMBLE_JOBID \
    --mode pooled \
    --strategy prob_average \
    --threshold 0.5 \
    --seed "$SEED"

# Ensemble 2: Top 2 (SMOTE+RUS + EasyEnsemble)
echo ""
echo "=== Ensemble: SMOTE+RUS + EasyEnsemble ==="
python scripts/python/archive/ensemble_evaluate.py \
    --models RF EasyEnsemble \
    --tags imbal_v2_smote_rus imbal_v2_easy_ensemble \
    --jobids $SMOTE_RUS_JOBID $EASY_ENSEMBLE_JOBID \
    --mode pooled \
    --strategy prob_average \
    --threshold 0.45 \
    --seed "$SEED"

# Ensemble 3: RF + EasyEnsemble (high recall)
echo ""
echo "=== Ensemble: RF Baseline + EasyEnsemble ==="
python scripts/python/archive/ensemble_evaluate.py \
    --models RF EasyEnsemble \
    --tags imbal_v2_baseline imbal_v2_easy_ensemble \
    --jobids $BASELINE_JOBID $EASY_ENSEMBLE_JOBID \
    --mode pooled \
    --strategy prob_average \
    --threshold 0.5 \
    --seed "$SEED"

# Ensemble 4: any_positive strategy (max recall)
echo ""
echo "=== Ensemble: All 6 Methods (any_positive strategy) ==="
python scripts/python/archive/ensemble_evaluate.py \
    --models RF RF RF RF BalancedRF EasyEnsemble \
    --tags imbal_v2_baseline imbal_v2_smote_tomek imbal_v2_smote_enn imbal_v2_smote_rus imbal_v2_balanced_rf imbal_v2_easy_ensemble \
    --jobids $BASELINE_JOBID $SMOTE_TOMEK_JOBID $SMOTE_ENN_JOBID $SMOTE_RUS_JOBID $BALANCED_RF_JOBID $EASY_ENSEMBLE_JOBID \
    --mode pooled \
    --strategy any_positive \
    --seed "$SEED"

echo ""
echo "=== ALL ENSEMBLE EVALUATIONS DONE ==="

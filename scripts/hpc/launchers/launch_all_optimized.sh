#!/bin/bash
# ============================================================
# Optimized: Launch ALL missing seed experiments
# Memory optimized, distributed across queues
# ============================================================

set -euo pipefail

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_IDS_FILE="scripts/hpc/logs/imbalance/job_ids_all_optimized_${TIMESTAMP}.txt"

echo "============================================================"
echo "[OPTIMIZED] All Missing Seed Experiments"
echo "Timestamp: $TIMESTAMP"
echo "============================================================"

echo "# Optimized All Jobs - $TIMESTAMP" > "$JOB_IDS_FILE"

declare -A TRAIN_JOBS

# ============================================================
# 1. BalancedRF + EasyEnsemble (already running: 14619088-91)
# ============================================================
echo ""
echo "=== BalancedRF/EasyEnsemble already running ==="
echo "14619088: BalancedRF seed=123"
echo "14619089: BalancedRF seed=456"
echo "14619090: EasyEnsemble seed=123"
echo "14619091: EasyEnsemble seed=456"

TRAIN_JOBS["balanced_rf_seed123"]="14619088.spcc-adm1"
TRAIN_JOBS["balanced_rf_seed456"]="14619089.spcc-adm1"
TRAIN_JOBS["easy_ensemble_seed123"]="14619090.spcc-adm1"
TRAIN_JOBS["easy_ensemble_seed456"]="14619091.spcc-adm1"

# ============================================================
# 2. SMOTE + BalancedRF (seed 456, ratios 0.1, 0.5, 1.0)
# Queue: SINGLE, Memory: 4GB, CPUs: 4
# ============================================================
echo ""
echo "=== SMOTE + BalancedRF (seed 456) ==="

PBS_OPTS_SMALL="-l select=1:ncpus=4:mem=4gb -l walltime=04:00:00 -q SINGLE"

for RATIO in 0.1 0.5 1.0; do
    RATIO_TAG=$(echo $RATIO | tr '.' '_')
    TAG="imbal_v2_smote_balanced_rf_ratio${RATIO_TAG}_seed456"
    JOB_ID=$(qsub $PBS_OPTS_SMALL \
        -v SEED=456,RATIO=$RATIO,RATIO_TAG=$RATIO_TAG \
        scripts/hpc/jobs/imbalance/pbs_train_smote_balanced_rf_ratio.sh)
    echo "Submitted smote_balanced_rf ratio=$RATIO seed=456: $JOB_ID"
    echo "smote_balanced_rf_ratio${RATIO_TAG}_seed456: $JOB_ID" >> "$JOB_IDS_FILE"
    TRAIN_JOBS["smote_balanced_rf_ratio${RATIO_TAG}_seed456"]="$JOB_ID"
    sleep 0.3
done

# ============================================================
# 3. Undersample methods (seed 42)
# Queue: TINY (very small jobs), Memory: 4GB, CPUs: 4
# ============================================================
echo ""
echo "=== Undersample methods (seed 42) ==="

# Try TINY queue first, fall back to SINGLE
PBS_OPTS_TINY="-l select=1:ncpus=4:mem=4gb -l walltime=04:00:00 -q SINGLE"

# undersample_enn: ratio 0.1, 0.5
for RATIO in 0.1 0.5; do
    RATIO_TAG=$(echo $RATIO | tr '.' '_')
    TAG="imbal_v2_undersample_enn_ratio${RATIO_TAG}_seed42"
    JOB_ID=$(qsub $PBS_OPTS_TINY \
        -v MODEL=RF,RATIO=$RATIO,METHOD=undersample_enn,TAG=$TAG,SEED=42 \
        scripts/hpc/jobs/imbalance/pbs_train_generic_ratio.sh)
    echo "Submitted undersample_enn ratio=$RATIO seed=42: $JOB_ID"
    echo "undersample_enn_ratio${RATIO_TAG}_seed42: $JOB_ID" >> "$JOB_IDS_FILE"
    TRAIN_JOBS["undersample_enn_ratio${RATIO_TAG}_seed42"]="$JOB_ID"
    sleep 0.3
done

# undersample_rus: ratio 0.5
RATIO=0.5
RATIO_TAG=$(echo $RATIO | tr '.' '_')
TAG="imbal_v2_undersample_rus_ratio${RATIO_TAG}_seed42"
JOB_ID=$(qsub $PBS_OPTS_TINY \
    -v MODEL=RF,RATIO=$RATIO,METHOD=undersample_rus,TAG=$TAG,SEED=42 \
    scripts/hpc/jobs/imbalance/pbs_train_generic_ratio.sh)
echo "Submitted undersample_rus ratio=$RATIO seed=42: $JOB_ID"
echo "undersample_rus_ratio${RATIO_TAG}_seed42: $JOB_ID" >> "$JOB_IDS_FILE"
TRAIN_JOBS["undersample_rus_ratio${RATIO_TAG}_seed42"]="$JOB_ID"
sleep 0.3

# undersample_tomek: ratio 0.5
TAG="imbal_v2_undersample_tomek_ratio${RATIO_TAG}_seed42"
JOB_ID=$(qsub $PBS_OPTS_TINY \
    -v MODEL=RF,RATIO=$RATIO,METHOD=undersample_tomek,TAG=$TAG,SEED=42 \
    scripts/hpc/jobs/imbalance/pbs_train_generic_ratio.sh)
echo "Submitted undersample_tomek ratio=$RATIO seed=42: $JOB_ID"
echo "undersample_tomek_ratio${RATIO_TAG}_seed42: $JOB_ID" >> "$JOB_IDS_FILE"
TRAIN_JOBS["undersample_tomek_ratio${RATIO_TAG}_seed42"]="$JOB_ID"

echo ""
echo "=== Submitting evaluation jobs with dependencies ==="

# ============================================================
# Evaluation jobs
# ============================================================

# BalancedRF evaluations (depend on existing jobs)
for SEED in 123 456; do
    TAG="imbal_v2_balanced_rf_seed${SEED}"
    TRAIN_ID="${TRAIN_JOBS[balanced_rf_seed${SEED}]}"
    TRAIN_JOBID=$(echo "$TRAIN_ID" | cut -d'.' -f1)
    EVAL_JOB=$(qsub -v MODEL=BalancedRF,TAG=$TAG,TRAIN_JOBID=$TRAIN_JOBID,SEED=$SEED \
        -W depend=afterok:$TRAIN_ID \
        scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
    echo "Eval balanced_rf_seed${SEED}: $EVAL_JOB"
    sleep 0.2
done

# EasyEnsemble evaluations
for SEED in 123 456; do
    TAG="imbal_v2_easy_ensemble_seed${SEED}"
    TRAIN_ID="${TRAIN_JOBS[easy_ensemble_seed${SEED}]}"
    TRAIN_JOBID=$(echo "$TRAIN_ID" | cut -d'.' -f1)
    EVAL_JOB=$(qsub -v MODEL=EasyEnsemble,TAG=$TAG,TRAIN_JOBID=$TRAIN_JOBID,SEED=$SEED \
        -W depend=afterok:$TRAIN_ID \
        scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
    echo "Eval easy_ensemble_seed${SEED}: $EVAL_JOB"
    sleep 0.2
done

# SMOTE + BalancedRF evaluations
for RATIO in 0.1 0.5 1.0; do
    RATIO_TAG=$(echo $RATIO | tr '.' '_')
    TAG="imbal_v2_smote_balanced_rf_ratio${RATIO_TAG}_seed456"
    TRAIN_ID="${TRAIN_JOBS[smote_balanced_rf_ratio${RATIO_TAG}_seed456]}"
    TRAIN_JOBID=$(echo "$TRAIN_ID" | cut -d'.' -f1)
    EVAL_JOB=$(qsub -v MODEL=BalancedRF,TAG=$TAG,TRAIN_JOBID=$TRAIN_JOBID,SEED=456 \
        -W depend=afterok:$TRAIN_ID \
        scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
    echo "Eval smote_balanced_rf_ratio${RATIO_TAG}_seed456: $EVAL_JOB"
    sleep 0.2
done

# Undersample evaluations
for METHOD in undersample_enn undersample_rus undersample_tomek; do
    if [[ "$METHOD" == "undersample_enn" ]]; then
        RATIOS="0.1 0.5"
    else
        RATIOS="0.5"
    fi
    
    for RATIO in $RATIOS; do
        RATIO_TAG=$(echo $RATIO | tr '.' '_')
        TAG="imbal_v2_${METHOD}_ratio${RATIO_TAG}_seed42"
        KEY="${METHOD}_ratio${RATIO_TAG}_seed42"
        TRAIN_ID="${TRAIN_JOBS[$KEY]}"
        TRAIN_JOBID=$(echo "$TRAIN_ID" | cut -d'.' -f1)
        EVAL_JOB=$(qsub -v MODEL=RF,TAG=$TAG,TRAIN_JOBID=$TRAIN_JOBID,SEED=42 \
            -W depend=afterok:$TRAIN_ID \
            scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
        echo "Eval ${METHOD}_ratio${RATIO_TAG}_seed42: $EVAL_JOB"
        sleep 0.2
    done
done

echo ""
echo "============================================================"
echo "[ALL DONE] Training: 7 new jobs, Evaluation: 11 jobs"
echo "Job IDs saved to: $JOB_IDS_FILE"
echo "============================================================"

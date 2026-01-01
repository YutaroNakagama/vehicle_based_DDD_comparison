#!/bin/bash
# ============================================================
# Launch evaluation for missing seed experiments
# ============================================================
# Job IDs from launch_missing_seeds.sh:
# 14619088: BalancedRF seed=123
# 14619089: BalancedRF seed=456
# 14619090: EasyEnsemble seed=123
# 14619091: EasyEnsemble seed=456
# 14619092: smote_balanced_rf ratio=0.1 seed=456
# 14619093: smote_balanced_rf ratio=0.5 seed=456
# 14619094: smote_balanced_rf ratio=1.0 seed=456
# 14619095: undersample_enn ratio=0.1 seed=42
# 14619096: undersample_enn ratio=0.5 seed=42
# 14619097: undersample_rus ratio=0.5 seed=42
# 14619098: undersample_tomek ratio=0.5 seed=42
# ============================================================

set -euo pipefail

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_IDS_FILE="scripts/hpc/logs/imbalance/job_ids_eval_missing_seeds_${TIMESTAMP}.txt"

echo "============================================================"
echo "[MISSING SEEDS] Submitting evaluation jobs"
echo "Timestamp: $TIMESTAMP"
echo "Job IDs will be saved to: $JOB_IDS_FILE"
echo "============================================================"

echo "# Missing Seed Evaluation Jobs - $TIMESTAMP" > "$JOB_IDS_FILE"

# ============================================================
# BalancedRF evaluations
# ============================================================
echo ""
echo "=== BalancedRF Evaluations ==="

# seed 123
JOB_ID=$(qsub -v MODEL=BalancedRF,TAG=imbal_v2_balanced_rf_seed123,TRAIN_JOBID=14619088,SEED=123 \
    -W depend=afterok:14619088.spcc-adm1 \
    scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
echo "Submitted eval BalancedRF seed=123: $JOB_ID"
echo "eval_balanced_rf_seed123: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 1

# seed 456
JOB_ID=$(qsub -v MODEL=BalancedRF,TAG=imbal_v2_balanced_rf_seed456,TRAIN_JOBID=14619089,SEED=456 \
    -W depend=afterok:14619089.spcc-adm1 \
    scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
echo "Submitted eval BalancedRF seed=456: $JOB_ID"
echo "eval_balanced_rf_seed456: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 1

# ============================================================
# EasyEnsemble evaluations
# ============================================================
echo ""
echo "=== EasyEnsemble Evaluations ==="

# seed 123
JOB_ID=$(qsub -v MODEL=EasyEnsemble,TAG=imbal_v2_easy_ensemble_seed123,TRAIN_JOBID=14619090,SEED=123 \
    -W depend=afterok:14619090.spcc-adm1 \
    scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
echo "Submitted eval EasyEnsemble seed=123: $JOB_ID"
echo "eval_easy_ensemble_seed123: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 1

# seed 456
JOB_ID=$(qsub -v MODEL=EasyEnsemble,TAG=imbal_v2_easy_ensemble_seed456,TRAIN_JOBID=14619091,SEED=456 \
    -W depend=afterok:14619091.spcc-adm1 \
    scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
echo "Submitted eval EasyEnsemble seed=456: $JOB_ID"
echo "eval_easy_ensemble_seed456: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 1

# ============================================================
# SMOTE + BalancedRF evaluations (seed 456)
# ============================================================
echo ""
echo "=== SMOTE + BalancedRF Evaluations (seed 456) ==="

# ratio 0.1
JOB_ID=$(qsub -v MODEL=BalancedRF,TAG=imbal_v2_smote_balanced_rf_ratio0_1_seed456,TRAIN_JOBID=14619092,SEED=456 \
    -W depend=afterok:14619092.spcc-adm1 \
    scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
echo "Submitted eval smote_balanced_rf ratio=0.1 seed=456: $JOB_ID"
echo "eval_smote_balanced_rf_ratio0_1_seed456: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 1

# ratio 0.5
JOB_ID=$(qsub -v MODEL=BalancedRF,TAG=imbal_v2_smote_balanced_rf_ratio0_5_seed456,TRAIN_JOBID=14619093,SEED=456 \
    -W depend=afterok:14619093.spcc-adm1 \
    scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
echo "Submitted eval smote_balanced_rf ratio=0.5 seed=456: $JOB_ID"
echo "eval_smote_balanced_rf_ratio0_5_seed456: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 1

# ratio 1.0
JOB_ID=$(qsub -v MODEL=BalancedRF,TAG=imbal_v2_smote_balanced_rf_ratio1_0_seed456,TRAIN_JOBID=14619094,SEED=456 \
    -W depend=afterok:14619094.spcc-adm1 \
    scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
echo "Submitted eval smote_balanced_rf ratio=1.0 seed=456: $JOB_ID"
echo "eval_smote_balanced_rf_ratio1_0_seed456: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 1

# ============================================================
# Undersample ENN evaluations (seed 42)
# ============================================================
echo ""
echo "=== Undersample ENN Evaluations (seed 42) ==="

# ratio 0.1
JOB_ID=$(qsub -v MODEL=RF,TAG=imbal_v2_undersample_enn_ratio0_1_seed42,TRAIN_JOBID=14619095,SEED=42 \
    -W depend=afterok:14619095.spcc-adm1 \
    scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
echo "Submitted eval undersample_enn ratio=0.1 seed=42: $JOB_ID"
echo "eval_undersample_enn_ratio0_1_seed42: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 1

# ratio 0.5
JOB_ID=$(qsub -v MODEL=RF,TAG=imbal_v2_undersample_enn_ratio0_5_seed42,TRAIN_JOBID=14619096,SEED=42 \
    -W depend=afterok:14619096.spcc-adm1 \
    scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
echo "Submitted eval undersample_enn ratio=0.5 seed=42: $JOB_ID"
echo "eval_undersample_enn_ratio0_5_seed42: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 1

# ============================================================
# Undersample RUS evaluation (seed 42)
# ============================================================
echo ""
echo "=== Undersample RUS Evaluation (seed 42) ==="

JOB_ID=$(qsub -v MODEL=RF,TAG=imbal_v2_undersample_rus_ratio0_5_seed42,TRAIN_JOBID=14619097,SEED=42 \
    -W depend=afterok:14619097.spcc-adm1 \
    scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
echo "Submitted eval undersample_rus ratio=0.5 seed=42: $JOB_ID"
echo "eval_undersample_rus_ratio0_5_seed42: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 1

# ============================================================
# Undersample Tomek evaluation (seed 42)
# ============================================================
echo ""
echo "=== Undersample Tomek Evaluation (seed 42) ==="

JOB_ID=$(qsub -v MODEL=RF,TAG=imbal_v2_undersample_tomek_ratio0_5_seed42,TRAIN_JOBID=14619098,SEED=42 \
    -W depend=afterok:14619098.spcc-adm1 \
    scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
echo "Submitted eval undersample_tomek ratio=0.5 seed=42: $JOB_ID"
echo "eval_undersample_tomek_ratio0_5_seed42: $JOB_ID" >> "$JOB_IDS_FILE"

echo ""
echo "============================================================"
echo "[DONE] All evaluation jobs submitted"
echo "Total jobs: 11"
echo "Job IDs saved to: $JOB_IDS_FILE"
echo "Note: Evaluation jobs will start after training jobs complete"
echo "============================================================"

#!/bin/bash
# ============================================================
# Launch missing seed experiments for imbal_v2
# ============================================================
# Missing experiments identified:
# 1. BalancedRF: seed 123, 456
# 2. EasyEnsemble: seed 123, 456
# 3. smote_balanced_rf: seed 456 (ratio 0.1, 0.5, 1.0)
# 4. undersample_enn: seed 42 (ratio 0.1, 0.5)
# 5. undersample_rus: seed 42 (ratio 0.5)
# 6. undersample_tomek: seed 42 (ratio 0.5)
# ============================================================

set -euo pipefail

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_IDS_FILE="scripts/hpc/logs/imbalance/job_ids_missing_seeds_${TIMESTAMP}.txt"

echo "============================================================"
echo "[MISSING SEEDS] Submitting experiments"
echo "Timestamp: $TIMESTAMP"
echo "Job IDs will be saved to: $JOB_IDS_FILE"
echo "============================================================"

# Initialize job IDs file
echo "# Missing Seed Experiments - $TIMESTAMP" > "$JOB_IDS_FILE"

# ============================================================
# 1. BalancedRF: seed 123, 456
# ============================================================
echo ""
echo "=== BalancedRF (seeds 123, 456) ==="

for SEED in 123 456; do
    JOB_ID=$(qsub -v SEED=$SEED scripts/hpc/jobs/imbalance/pbs_train_balanced_rf.sh)
    echo "Submitted BalancedRF seed=$SEED: $JOB_ID"
    echo "balanced_rf_seed${SEED}: $JOB_ID" >> "$JOB_IDS_FILE"
    sleep 1
done

# ============================================================
# 2. EasyEnsemble: seed 123, 456
# ============================================================
echo ""
echo "=== EasyEnsemble (seeds 123, 456) ==="

for SEED in 123 456; do
    JOB_ID=$(qsub -v SEED=$SEED scripts/hpc/jobs/imbalance/pbs_train_easy_ensemble.sh)
    echo "Submitted EasyEnsemble seed=$SEED: $JOB_ID"
    echo "easy_ensemble_seed${SEED}: $JOB_ID" >> "$JOB_IDS_FILE"
    sleep 1
done

# ============================================================
# 3. smote_balanced_rf: seed 456 (ratio 0.1, 0.5, 1.0)
# ============================================================
echo ""
echo "=== SMOTE + BalancedRF (seed 456, ratios 0.1, 0.5, 1.0) ==="

for RATIO in 0.1 0.5 1.0; do
    RATIO_TAG=$(echo $RATIO | tr '.' '_')
    JOB_ID=$(qsub -v SEED=456,RATIO=$RATIO,RATIO_TAG=$RATIO_TAG scripts/hpc/jobs/imbalance/pbs_train_smote_balanced_rf_ratio.sh)
    echo "Submitted smote_balanced_rf ratio=$RATIO seed=456: $JOB_ID"
    echo "smote_balanced_rf_ratio${RATIO_TAG}_seed456: $JOB_ID" >> "$JOB_IDS_FILE"
    sleep 1
done

# ============================================================
# 4. undersample_enn: seed 42 (ratio 0.1, 0.5)
# ============================================================
echo ""
echo "=== undersample_enn (seed 42, ratios 0.1, 0.5) ==="

for RATIO in 0.1 0.5; do
    RATIO_TAG=$(echo $RATIO | tr '.' '_')
    JOB_ID=$(qsub -v SEED=42,RATIO=$RATIO,RATIO_TAG=$RATIO_TAG,METHOD=undersample_enn scripts/hpc/jobs/imbalance/pbs_train_generic_ratio.sh)
    echo "Submitted undersample_enn ratio=$RATIO seed=42: $JOB_ID"
    echo "undersample_enn_ratio${RATIO_TAG}_seed42: $JOB_ID" >> "$JOB_IDS_FILE"
    sleep 1
done

# ============================================================
# 5. undersample_rus: seed 42 (ratio 0.5)
# ============================================================
echo ""
echo "=== undersample_rus (seed 42, ratio 0.5) ==="

RATIO=0.5
RATIO_TAG=$(echo $RATIO | tr '.' '_')
JOB_ID=$(qsub -v SEED=42,RATIO=$RATIO,RATIO_TAG=$RATIO_TAG,METHOD=undersample_rus scripts/hpc/jobs/imbalance/pbs_train_generic_ratio.sh)
echo "Submitted undersample_rus ratio=$RATIO seed=42: $JOB_ID"
echo "undersample_rus_ratio${RATIO_TAG}_seed42: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 1

# ============================================================
# 6. undersample_tomek: seed 42 (ratio 0.5)
# ============================================================
echo ""
echo "=== undersample_tomek (seed 42, ratio 0.5) ==="

RATIO=0.5
RATIO_TAG=$(echo $RATIO | tr '.' '_')
JOB_ID=$(qsub -v SEED=42,RATIO=$RATIO,RATIO_TAG=$RATIO_TAG,METHOD=undersample_tomek scripts/hpc/jobs/imbalance/pbs_train_generic_ratio.sh)
echo "Submitted undersample_tomek ratio=$RATIO seed=42: $JOB_ID"
echo "undersample_tomek_ratio${RATIO_TAG}_seed42: $JOB_ID" >> "$JOB_IDS_FILE"

echo ""
echo "============================================================"
echo "[DONE] All missing seed experiments submitted"
echo "Total jobs: 11"
echo "Job IDs saved to: $JOB_IDS_FILE"
echo "============================================================"

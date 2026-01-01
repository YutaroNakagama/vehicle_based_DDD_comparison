#!/bin/bash
# ============================================================
# Optimized: Launch missing undersample experiments
# Memory: 4GB (actual usage ~1-2GB)
# CPUs: 4 (sufficient for single-threaded Optuna)
# Queue: SINGLE (faster scheduling for small jobs)
# Walltime: 4 hours (actual ~2-3 hours)
# ============================================================

set -euo pipefail

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_IDS_FILE="scripts/hpc/logs/imbalance/job_ids_undersample_optimized_${TIMESTAMP}.txt"

echo "============================================================"
echo "[OPTIMIZED] Submitting undersample experiments"
echo "Memory: 4GB, CPUs: 4, Queue: SINGLE"
echo "Timestamp: $TIMESTAMP"
echo "============================================================"

echo "# Optimized Undersample Jobs - $TIMESTAMP" > "$JOB_IDS_FILE"

# Common PBS options for optimization
PBS_OPTS="-l select=1:ncpus=4:mem=4gb -l walltime=04:00:00 -q SINGLE"

# ============================================================
# undersample_enn: seed 42 (ratio 0.1, 0.5)
# ============================================================
echo ""
echo "=== undersample_enn (seed 42, ratios 0.1, 0.5) ==="

for RATIO in 0.1 0.5; do
    RATIO_TAG=$(echo $RATIO | tr '.' '_')
    TAG="imbal_v2_undersample_enn_ratio${RATIO_TAG}_seed42"
    JOB_ID=$(qsub $PBS_OPTS \
        -v MODEL=RF,RATIO=$RATIO,METHOD=undersample_enn,TAG=$TAG,SEED=42 \
        scripts/hpc/jobs/imbalance/pbs_train_generic_ratio.sh)
    echo "Submitted undersample_enn ratio=$RATIO seed=42: $JOB_ID"
    echo "undersample_enn_ratio${RATIO_TAG}_seed42: $JOB_ID" >> "$JOB_IDS_FILE"
    sleep 0.5
done

# ============================================================
# undersample_rus: seed 42 (ratio 0.5)
# ============================================================
echo ""
echo "=== undersample_rus (seed 42, ratio 0.5) ==="

RATIO=0.5
RATIO_TAG=$(echo $RATIO | tr '.' '_')
TAG="imbal_v2_undersample_rus_ratio${RATIO_TAG}_seed42"
JOB_ID=$(qsub $PBS_OPTS \
    -v MODEL=RF,RATIO=$RATIO,METHOD=undersample_rus,TAG=$TAG,SEED=42 \
    scripts/hpc/jobs/imbalance/pbs_train_generic_ratio.sh)
echo "Submitted undersample_rus ratio=$RATIO seed=42: $JOB_ID"
echo "undersample_rus_ratio${RATIO_TAG}_seed42: $JOB_ID" >> "$JOB_IDS_FILE"
sleep 0.5

# ============================================================
# undersample_tomek: seed 42 (ratio 0.5)
# ============================================================
echo ""
echo "=== undersample_tomek (seed 42, ratio 0.5) ==="

RATIO=0.5
RATIO_TAG=$(echo $RATIO | tr '.' '_')
TAG="imbal_v2_undersample_tomek_ratio${RATIO_TAG}_seed42"
JOB_ID=$(qsub $PBS_OPTS \
    -v MODEL=RF,RATIO=$RATIO,METHOD=undersample_tomek,TAG=$TAG,SEED=42 \
    scripts/hpc/jobs/imbalance/pbs_train_generic_ratio.sh)
echo "Submitted undersample_tomek ratio=$RATIO seed=42: $JOB_ID"
echo "undersample_tomek_ratio${RATIO_TAG}_seed42: $JOB_ID" >> "$JOB_IDS_FILE"

echo ""
echo "============================================================"
echo "[DONE] 4 optimized jobs submitted"
echo "Job IDs saved to: $JOB_IDS_FILE"
echo "============================================================"

# Extract job IDs for evaluation dependency
echo ""
echo "=== Now submitting evaluation jobs ==="

# Read job IDs from file and submit evaluations
sleep 1

while IFS=': ' read -r name job_id; do
    [[ "$name" =~ ^# ]] && continue
    [[ -z "$name" ]] && continue
    
    # Parse components
    if [[ "$name" =~ undersample_enn_ratio([0-9_]+)_seed([0-9]+) ]]; then
        RATIO_TAG="${BASH_REMATCH[1]}"
        SEED="${BASH_REMATCH[2]}"
        TAG="imbal_v2_undersample_enn_ratio${RATIO_TAG}_seed${SEED}"
    elif [[ "$name" =~ undersample_rus_ratio([0-9_]+)_seed([0-9]+) ]]; then
        RATIO_TAG="${BASH_REMATCH[1]}"
        SEED="${BASH_REMATCH[2]}"
        TAG="imbal_v2_undersample_rus_ratio${RATIO_TAG}_seed${SEED}"
    elif [[ "$name" =~ undersample_tomek_ratio([0-9_]+)_seed([0-9]+) ]]; then
        RATIO_TAG="${BASH_REMATCH[1]}"
        SEED="${BASH_REMATCH[2]}"
        TAG="imbal_v2_undersample_tomek_ratio${RATIO_TAG}_seed${SEED}"
    else
        continue
    fi
    
    # Extract numeric job ID
    TRAIN_JOBID=$(echo "$job_id" | cut -d'.' -f1)
    
    EVAL_JOB=$(qsub -v MODEL=RF,TAG=$TAG,TRAIN_JOBID=$TRAIN_JOBID,SEED=$SEED \
        -W depend=afterok:$job_id \
        scripts/hpc/jobs/imbalance/pbs_evaluate.sh)
    echo "Submitted eval $name: $EVAL_JOB (depends on $job_id)"
    
done < "$JOB_IDS_FILE"

echo ""
echo "============================================================"
echo "[ALL DONE] Training + Evaluation jobs submitted"
echo "============================================================"

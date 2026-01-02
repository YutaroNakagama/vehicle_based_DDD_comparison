#!/bin/bash
# ============================================================
# SMOTE Comparison Experiments Launcher
# ============================================================
# Experiment 1: Pooled mode (3 methods × 3 seeds)
# Experiment 2: Source/Target mode (3 methods × 2 rankings × 1 metric × 2 levels × 2 modes)
# ============================================================
# Queue Strategy:
#   - SINGLE: 4 waiting, good capacity -> Pooled experiments
#   - DEFAULT: 2 free slots -> Source/Target (BalancedRF)
#   - SMALL: full -> skip
#   - LONG-L: empty -> Source/Target (long jobs if needed)
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="scripts/hpc/logs/smote_comparison"
mkdir -p "$LOG_DIR"
JOB_LOG="$LOG_DIR/job_ids_${TIMESTAMP}.txt"

echo "# SMOTE Comparison Experiment - $(date)" > "$JOB_LOG"
echo "============================================================"
echo "SMOTE Comparison Experiment Launcher"
echo "============================================================"

# ============================================================
# Experiment 1: Pooled Mode (SINGLE queue)
# 3 methods × 3 seeds = 9 jobs
# ============================================================
echo ""
echo "=== Experiment 1: Pooled Mode (SINGLE queue) ==="

SEEDS=(42 123 456)

for SEED in "${SEEDS[@]}"; do
    echo "  Submitting pooled experiments for seed=$SEED..."
    
    # 1a. Subject-wise SMOTE
    JOB1=$(qsub -v SEED="$SEED" scripts/hpc/jobs/imbalance/pbs_train_smote_subjectwise.sh)
    echo "    [SW-SMOTE] seed=$SEED -> $JOB1"
    echo "POOLED:SW_SMOTE:seed${SEED}:${JOB1}" >> "$JOB_LOG"
    
    # 1b. Simple SMOTE
    JOB2=$(qsub -v SEED="$SEED" scripts/hpc/jobs/imbalance/pbs_train_smote.sh)
    echo "    [SMOTE] seed=$SEED -> $JOB2"
    echo "POOLED:SMOTE:seed${SEED}:${JOB2}" >> "$JOB_LOG"
    
    # 1c. SMOTE + BalancedRF (-> DEFAULT queue)
    JOB3=$(qsub -v SEED="$SEED" scripts/hpc/jobs/imbalance/pbs_train_smote_balanced_rf.sh)
    echo "    [SMOTE+BRF] seed=$SEED -> $JOB3"
    echo "POOLED:SMOTE_BRF:seed${SEED}:${JOB3}" >> "$JOB_LOG"
done

echo "  Pooled experiments submitted: $((${#SEEDS[@]} * 3)) jobs"

# ============================================================
# Experiment 2: Source/Target Mode
# Focus on: knn, lof × mmd (most reliable) × out/in_domain × 2 modes
# 3 methods × 2 rankings × 1 metric × 2 levels × 2 modes = 24 jobs
# ============================================================
echo ""
echo "=== Experiment 2: Source/Target Mode ==="

RANKING_METHODS=("knn" "lof")
METRICS=("mmd")  # Focus on MMD (most stable)
LEVELS=("out_domain" "in_domain")  # Skip mid_domain for efficiency
MODES=("source_only" "target_only")
SAMPLING_METHODS=("sw_smote" "smote" "smote_brf")
SEED=42

RANKING_BASE="results/analysis/domain/distance/subject-wise"

count=0
for RANKING in "${RANKING_METHODS[@]}"; do
    for METRIC in "${METRICS[@]}"; do
        for LEVEL in "${LEVELS[@]}"; do
            for MODE in "${MODES[@]}"; do
                for SAMPLING in "${SAMPLING_METHODS[@]}"; do
                    # Construct subject file path
                    SUBJECT_FILE="$PROJECT_ROOT/$RANKING_BASE/$METRIC/groups/clustering_ranked/${METRIC}_${RANKING}_${LEVEL}.txt"
                    
                    if [[ ! -f "$SUBJECT_FILE" ]]; then
                        echo "    [WARN] File not found: $SUBJECT_FILE"
                        continue
                    fi
                    
                    TAG="rank_${RANKING}_${METRIC}_${LEVEL}_${SAMPLING}"
                    
                    # Submit job based on sampling method
                    if [[ "$SAMPLING" == "sw_smote" ]]; then
                        JOB=$(qsub -v MODE="$MODE",SUBJECT_FILE="$SUBJECT_FILE",TAG="$TAG",SEED="$SEED" \
                            scripts/hpc/jobs/imbalance/pbs_train_sw_smote_ranking.sh)
                    elif [[ "$SAMPLING" == "smote" ]]; then
                        JOB=$(qsub -v MODE="$MODE",SUBJECT_FILE="$SUBJECT_FILE",TAG="$TAG",SEED="$SEED" \
                            scripts/hpc/jobs/imbalance/pbs_train_smote_ranking.sh)
                    elif [[ "$SAMPLING" == "smote_brf" ]]; then
                        JOB=$(qsub -v MODE="$MODE",SUBJECT_FILE="$SUBJECT_FILE",TAG="$TAG",SEED="$SEED" \
                            scripts/hpc/jobs/imbalance/pbs_train_smote_brf_ranking.sh)
                    fi
                    
                    echo "    [${SAMPLING}] ${MODE}/${RANKING}/${METRIC}/${LEVEL} -> $JOB"
                    echo "${MODE}:${SAMPLING}:${RANKING}:${METRIC}:${LEVEL}:${JOB}" >> "$JOB_LOG"
                    ((count++))
                done
            done
        done
    done
done

echo ""
echo "  Source/Target experiments submitted: $count jobs"

echo ""
echo "============================================================"
echo "Total jobs submitted: $((${#SEEDS[@]} * 3 + count))"
echo "Job log: $JOB_LOG"
echo "============================================================"

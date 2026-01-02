#!/bin/bash
# Submit remaining Source/Target jobs
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
RANKING_BASE="$PROJECT_ROOT/results/analysis/domain/distance/subject-wise"
SEED=42
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/smote_comparison"
mkdir -p "$LOG_DIR"
JOB_LOG="$LOG_DIR/job_ids_remaining_$(date +%Y%m%d_%H%M%S).txt"

echo "# Remaining Source/Target jobs - $(date)" > "$JOB_LOG"

cd "$PROJECT_ROOT"

RANKING_METHODS=("knn" "lof")
MODES=("source_only" "target_only")
LEVELS=("out_domain" "in_domain")
METRIC="mmd"

count=0
for RANKING in "${RANKING_METHODS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for LEVEL in "${LEVELS[@]}"; do
            SUBJECT_FILE="$RANKING_BASE/$METRIC/groups/clustering_ranked/${METRIC}_${RANKING}_${LEVEL}.txt"
            
            if [[ ! -f "$SUBJECT_FILE" ]]; then
                echo "[WARN] Not found: $SUBJECT_FILE"
                continue
            fi
            
            # Subject-wise SMOTE
            TAG="rank_${RANKING}_${METRIC}_${LEVEL}_sw_smote"
            JOB1=$(qsub -v MODE="$MODE",SUBJECT_FILE="$SUBJECT_FILE",TAG="$TAG",SEED="$SEED" \
                scripts/hpc/jobs/imbalance/pbs_train_sw_smote_ranking.sh 2>/dev/null || echo "FAILED")
            echo "[SW-SMOTE] ${MODE}/${RANKING}/${LEVEL} -> $JOB1"
            echo "${MODE}:sw_smote:${RANKING}:${LEVEL}:${JOB1}" >> "$JOB_LOG"
            
            # Simple SMOTE
            TAG="rank_${RANKING}_${METRIC}_${LEVEL}_smote"
            JOB2=$(qsub -v MODE="$MODE",SUBJECT_FILE="$SUBJECT_FILE",TAG="$TAG",SEED="$SEED" \
                scripts/hpc/jobs/imbalance/pbs_train_smote_ranking.sh 2>/dev/null || echo "FAILED")
            echo "[SMOTE] ${MODE}/${RANKING}/${LEVEL} -> $JOB2"
            echo "${MODE}:smote:${RANKING}:${LEVEL}:${JOB2}" >> "$JOB_LOG"
            
            # SMOTE + BalancedRF
            TAG="rank_${RANKING}_${METRIC}_${LEVEL}_smote_brf"
            JOB3=$(qsub -v MODE="$MODE",SUBJECT_FILE="$SUBJECT_FILE",TAG="$TAG",SEED="$SEED" \
                scripts/hpc/jobs/imbalance/pbs_train_smote_brf_ranking.sh 2>/dev/null || echo "FAILED")
            echo "[SMOTE+BRF] ${MODE}/${RANKING}/${LEVEL} -> $JOB3"
            echo "${MODE}:smote_brf:${RANKING}:${LEVEL}:${JOB3}" >> "$JOB_LOG"
            
            ((count+=3))
        done
    done
done

echo ""
echo "Total jobs submitted: $count"
echo "Job log: $JOB_LOG"

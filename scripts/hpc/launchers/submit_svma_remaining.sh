#!/bin/bash
# Submit remaining SvmA domain_train jobs to SINGLE and SMALL queues
# These queues are currently unused and have available slots
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
SUBMITTED_FILE="/tmp/unified_submitted_keys.txt"
RANKING="knn"
N_TRIALS=100

# Alternate between SINGLE (max_run=10) and SMALL (max_run=7)
QUEUES=("SINGLE" "SMALL" "SINGLE" "SMALL" "SINGLE" "SMALL" "SINGLE")
Q_IDX=0

SUBMITTED=0
SKIPPED=0
set +e

for DISTANCE in mmd dtw wasserstein; do
  for DOMAIN in out_domain in_domain; do
    for SEED in 42 123; do
      # baseline (no ratio)
      KEY="SvmA:baseline:${DISTANCE}:${DOMAIN}:${SEED}"
      if ! grep -qF "$KEY" "$SUBMITTED_FILE" 2>/dev/null; then
        QUEUE="${QUEUES[$((Q_IDX % ${#QUEUES[@]}))]}"
        JOB_NAME="Sv_ba_${DISTANCE:0:2}${DOMAIN:0:1}_dt_s${SEED}"
        JOB_ID=$(qsub -N "$JOB_NAME" \
          -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q "$QUEUE" \
          -v MODEL=SvmA,CONDITION=baseline,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true \
          "$JOB_SCRIPT" 2>&1)
        if [[ $? -eq 0 ]]; then
          echo "$KEY" >> "$SUBMITTED_FILE"
          echo "[SUB] baseline | $DISTANCE | $DOMAIN | s$SEED | $QUEUE → $JOB_ID"
          ((SUBMITTED++))
          ((Q_IDX++))
          sleep 0.3
        else
          echo "[FAIL] $KEY | $QUEUE: $JOB_ID"
        fi
      else
        ((SKIPPED++))
      fi

      # ratio-based conditions
      for RATIO in 0.1 0.5; do
        for COND in smote_plain smote undersample; do
          KEY="SvmA:${COND}:${DISTANCE}:${DOMAIN}:${RATIO}:${SEED}"
          if ! grep -qF "$KEY" "$SUBMITTED_FILE" 2>/dev/null; then
            QUEUE="${QUEUES[$((Q_IDX % ${#QUEUES[@]}))]}"
            COND_SHORT="${COND:0:2}"
            JOB_NAME="Sv_${COND_SHORT}_${DISTANCE:0:2}${DOMAIN:0:1}_r${RATIO}_s${SEED}"
            
            # SMOTE conditions get 48h walltime
            WALLTIME="48:00:00"
            
            JOB_ID=$(qsub -N "$JOB_NAME" \
              -l select=1:ncpus=8:mem=32gb -l walltime=$WALLTIME -q "$QUEUE" \
              -v MODEL=SvmA,CONDITION=$COND,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true \
              "$JOB_SCRIPT" 2>&1)
            if [[ $? -eq 0 ]]; then
              echo "$KEY" >> "$SUBMITTED_FILE"
              echo "[SUB] $COND r$RATIO | $DISTANCE | $DOMAIN | s$SEED | $QUEUE → $JOB_ID"
              ((SUBMITTED++))
              ((Q_IDX++))
              sleep 0.3
            else
              echo "[FAIL] $KEY | $QUEUE: $JOB_ID"
            fi
          else
            ((SKIPPED++))
          fi
        done
      done
    done
  done
done

echo ""
echo "=== Summary ==="
echo "Submitted: $SUBMITTED"
echo "Skipped (already submitted): $SKIPPED"
echo "Total SvmA in tracking file: $(grep -c '^SvmA' "$SUBMITTED_FILE")"

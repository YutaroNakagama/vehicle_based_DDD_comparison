#!/bin/bash
# Submit remaining exp2 RF jobs across all available queues optimally.
# Queue capacities (RF only — BalancedRF excluded from paper):
#   SMALL:  30 max (4-core, <=8h) — primary RF queue
#   SINGLE: 40 max (4-core, <=8h) — overflow for RF
#   LONG:   15 max (8-core, 8h)   — overflow for RF
#   LARGE:  15 max (8-core+)      — overflow for RF
#   DEF:    40 max (QoS=def, MaxJobsPU=300) — independent QOS, used as additional overflow
#
# Run without args to submit; --dry-run to preview.

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
SUBMITTED_FILE="$LOG_DIR/submitted_exp2_v2.txt"
WAVE_LOG="$LOG_DIR/submit_exp2_multiqueue_$(date +%Y%m%d_%H%M%S).log"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

N_TRIALS=100
RANKING="knn"
SEEDS=(0 1 3 7 13 42 123 256 512 999 1337 2024)
RATIOS=(0.1 0.5)
DISTANCES=(mmd dtw wasserstein)
DOMAINS=(out_domain in_domain)
MODES=(source_only target_only mixed)

is_submitted() { grep -qxF "$1" "$SUBMITTED_FILE" 2>/dev/null; }

current_queue_count() {
    local q="$1"
    squeue -u s2240011 -p "$q" -h 2>/dev/null | wc -l
}

submit_one() {
    local CONDITION="$1" DISTANCE="$2" DOMAIN="$3" MODE="$4" RATIO="$5" SEED="$6" QUEUE="$7"
    local NCPUS_MEM WALLTIME KEY JOB_NAME CMD JOB_ID

    # Walltime tuned from observed runtimes (n=1383 COMPLETED + 34 TIMEOUTs):
    #   smote_plain mixed often exceeds 8h → 24h (QOS allows up to 7d)
    #   baseline mixed occasionally exceeds 6h → 8h
    case "$CONDITION" in
        balanced_rf) NCPUS_MEM="ncpus=8:mem=12gb"; WALLTIME="08:00:00" ;;
        smote_plain) NCPUS_MEM="ncpus=4:mem=10gb"; WALLTIME="24:00:00" ;;
        smote)       NCPUS_MEM="ncpus=4:mem=10gb"; WALLTIME="12:00:00" ;;
        *)           NCPUS_MEM="ncpus=4:mem=8gb";  WALLTIME="08:00:00" ;;
    esac

    if [[ -n "$RATIO" ]]; then
        KEY="${CONDITION}:${DISTANCE}:${DOMAIN}:${MODE}:${RATIO}:${SEED}"
        JOB_NAME="${CONDITION:0:2}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_r${RATIO}_s${SEED}"
        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE \
          -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true \
          $JOB_SCRIPT"
    else
        KEY="${CONDITION}:${DISTANCE}:${DOMAIN}:${MODE}:${SEED}"
        JOB_NAME="${CONDITION:0:2}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_s${SEED}"
        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE \
          -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true \
          $JOB_SCRIPT"
    fi

    is_submitted "$KEY" && return 2  # already submitted

    if $DRY_RUN; then
        echo "[DRY-RUN] $KEY → $QUEUE"
        return 0
    fi

    JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] $KEY → $JOB_ID" | tee -a "$WAVE_LOG"; return 1; }
    echo "$KEY" >> "$SUBMITTED_FILE"
    echo "[SUBMITTED] $KEY → $JOB_ID (queue=$QUEUE)" | tee -a "$WAVE_LOG"
    sleep 0.3
}

# ---- Current queue counts ----
LONG_CURRENT=$(current_queue_count "LONG")
LARGE_CURRENT=$(current_queue_count "LARGE")
SMALL_CURRENT=$(current_queue_count "SMALL")
SINGLE_CURRENT=$(current_queue_count "SINGLE")
DEF_CURRENT=$(current_queue_count "DEF")
LONG_SLOTS=$(( 15 - LONG_CURRENT ))
LARGE_SLOTS=$(( 15 - LARGE_CURRENT ))
SMALL_SLOTS=$(( 30 - SMALL_CURRENT ))
SINGLE_SLOTS=$(( 40 - SINGLE_CURRENT ))
DEF_SLOTS=$(( 40 - DEF_CURRENT ))

echo "[$(date)] Queue status: SMALL=$SMALL_CURRENT/30 (+$SMALL_SLOTS), SINGLE=$SINGLE_CURRENT/40 (+$SINGLE_SLOTS), LONG=$LONG_CURRENT/15 (+$LONG_SLOTS), LARGE=$LARGE_CURRENT/15 (+$LARGE_SLOTS), DEF=$DEF_CURRENT/40 (+$DEF_SLOTS)" | tee -a "$WAVE_LOG"
echo "[$(date)] Total available slots: $(( SMALL_SLOTS + SINGLE_SLOTS + LONG_SLOTS + LARGE_SLOTS + DEF_SLOTS ))" | tee -a "$WAVE_LOG"

LONG_USED=0
LARGE_USED=0
SMALL_USED=0
SINGLE_USED=0
DEF_USED=0
SKIP=0
TOTAL_SUBMITTED=0

for DISTANCE in "${DISTANCES[@]}"; do
  for DOMAIN in "${DOMAINS[@]}"; do
    for MODE in "${MODES[@]}"; do
      for SEED in "${SEEDS[@]}"; do

        # --- baseline (4-core) → SMALL → SINGLE → LONG → LARGE → DEF ---
        if [[ $SMALL_USED -lt $SMALL_SLOTS ]]; then
          submit_one "baseline" "$DISTANCE" "$DOMAIN" "$MODE" "" "$SEED" "SMALL"
          rc=$?; [[ $rc -eq 0 ]] && ((SMALL_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
        elif [[ $SINGLE_USED -lt $SINGLE_SLOTS ]]; then
          submit_one "baseline" "$DISTANCE" "$DOMAIN" "$MODE" "" "$SEED" "SINGLE"
          rc=$?; [[ $rc -eq 0 ]] && ((SINGLE_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
        elif [[ $LONG_USED -lt $LONG_SLOTS ]]; then
          submit_one "baseline" "$DISTANCE" "$DOMAIN" "$MODE" "" "$SEED" "LONG"
          rc=$?; [[ $rc -eq 0 ]] && ((LONG_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
        elif [[ $LARGE_USED -lt $LARGE_SLOTS ]]; then
          submit_one "baseline" "$DISTANCE" "$DOMAIN" "$MODE" "" "$SEED" "LARGE"
          rc=$?; [[ $rc -eq 0 ]] && ((LARGE_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
        elif [[ $DEF_USED -lt $DEF_SLOTS ]]; then
          submit_one "baseline" "$DISTANCE" "$DOMAIN" "$MODE" "" "$SEED" "DEF"
          rc=$?; [[ $rc -eq 0 ]] && ((DEF_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
        fi

        for RATIO in "${RATIOS[@]}"; do
          # --- smote_plain (4-core) → SMALL → SINGLE → LONG → LARGE → DEF ---
          if [[ $SMALL_USED -lt $SMALL_SLOTS ]]; then
            submit_one "smote_plain" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "SMALL"
            rc=$?; [[ $rc -eq 0 ]] && ((SMALL_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $SINGLE_USED -lt $SINGLE_SLOTS ]]; then
            submit_one "smote_plain" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "SINGLE"
            rc=$?; [[ $rc -eq 0 ]] && ((SINGLE_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $LONG_USED -lt $LONG_SLOTS ]]; then
            submit_one "smote_plain" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "LONG"
            rc=$?; [[ $rc -eq 0 ]] && ((LONG_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $LARGE_USED -lt $LARGE_SLOTS ]]; then
            submit_one "smote_plain" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "LARGE"
            rc=$?; [[ $rc -eq 0 ]] && ((LARGE_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $DEF_USED -lt $DEF_SLOTS ]]; then
            submit_one "smote_plain" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "DEF"
            rc=$?; [[ $rc -eq 0 ]] && ((DEF_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          fi
          # --- smote (4-core) → SMALL → SINGLE → LONG → LARGE → DEF ---
          if [[ $SMALL_USED -lt $SMALL_SLOTS ]]; then
            submit_one "smote" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "SMALL"
            rc=$?; [[ $rc -eq 0 ]] && ((SMALL_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $SINGLE_USED -lt $SINGLE_SLOTS ]]; then
            submit_one "smote" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "SINGLE"
            rc=$?; [[ $rc -eq 0 ]] && ((SINGLE_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $LONG_USED -lt $LONG_SLOTS ]]; then
            submit_one "smote" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "LONG"
            rc=$?; [[ $rc -eq 0 ]] && ((LONG_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $LARGE_USED -lt $LARGE_SLOTS ]]; then
            submit_one "smote" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "LARGE"
            rc=$?; [[ $rc -eq 0 ]] && ((LARGE_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $DEF_USED -lt $DEF_SLOTS ]]; then
            submit_one "smote" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "DEF"
            rc=$?; [[ $rc -eq 0 ]] && ((DEF_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          fi
          # --- undersample (4-core) → SMALL → SINGLE → LONG → LARGE → DEF ---
          if [[ $SMALL_USED -lt $SMALL_SLOTS ]]; then
            submit_one "undersample" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "SMALL"
            rc=$?; [[ $rc -eq 0 ]] && ((SMALL_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $SINGLE_USED -lt $SINGLE_SLOTS ]]; then
            submit_one "undersample" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "SINGLE"
            rc=$?; [[ $rc -eq 0 ]] && ((SINGLE_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $LONG_USED -lt $LONG_SLOTS ]]; then
            submit_one "undersample" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "LONG"
            rc=$?; [[ $rc -eq 0 ]] && ((LONG_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $LARGE_USED -lt $LARGE_SLOTS ]]; then
            submit_one "undersample" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "LARGE"
            rc=$?; [[ $rc -eq 0 ]] && ((LARGE_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          elif [[ $DEF_USED -lt $DEF_SLOTS ]]; then
            submit_one "undersample" "$DISTANCE" "$DOMAIN" "$MODE" "$RATIO" "$SEED" "DEF"
            rc=$?; [[ $rc -eq 0 ]] && ((DEF_USED++)) && ((TOTAL_SUBMITTED++)) || [[ $rc -eq 2 ]] && ((SKIP++))
          fi
        done

      done
    done
  done
done

TOTAL_IN_FILE=$(grep -c . "$SUBMITTED_FILE" 2>/dev/null || echo 0)
echo "" | tee -a "$WAVE_LOG"
echo "==== Summary ====" | tee -a "$WAVE_LOG"
echo "Submitted this run: $TOTAL_SUBMITTED (LONG:$LONG_USED, LARGE:$LARGE_USED, SMALL:$SMALL_USED, SINGLE:$SINGLE_USED, DEF:$DEF_USED)" | tee -a "$WAVE_LOG"
echo "Already submitted (skipped): $SKIP" | tee -a "$WAVE_LOG"
echo "Total submitted across all runs: $TOTAL_IN_FILE / 1512" | tee -a "$WAVE_LOG"
echo "Log: $WAVE_LOG" | tee -a "$WAVE_LOG"

#!/bin/bash
# Auto-submission daemon for remaining 14 SvmA exp3 jobs
# Watches for free slots and submits jobs as space becomes available
# Usage: nohup bash scripts/hpc/jobs/domain_analysis/auto_submit_svma_exp3_round2.sh &

set -uo pipefail

PBS_SCRIPT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
COMMON_SELECT="select=1:ncpus=8:mem=32gb"
COMMON_WALLTIME="walltime=48:00:00"
MAX_JOBS=170
CHECK_INTERVAL=120  # seconds between checks

# Define all 14 missing jobs: NAME|QUEUE|CONDITION|DISTANCE|DOMAIN|SEED|RATIO
JOBS=(
  "Sv_iv_di_dt_r05_s42|SINGLE|smote|dtw|in_domain|42|0.5"
  "Sv_iv_di_dt_r05_s123|DEFAULT|smote|dtw|in_domain|123|0.5"
  "Sv_iv_do_dt_r05_s42|LONG|smote|dtw|out_domain|42|0.5"
  "Sv_iv_wo_wa_r05_s123|SINGLE|smote|wasserstein|out_domain|123|0.5"
  "Sv_sp_di_dt_r05_s42|DEFAULT|smote_plain|dtw|in_domain|42|0.5"
  "Sv_sp_di_dt_r05_s123|LONG|smote_plain|dtw|in_domain|123|0.5"
  "Sv_sp_do_dt_r05_s42|SINGLE|smote_plain|dtw|out_domain|42|0.5"
  "Sv_sp_do_dt_r05_s123|DEFAULT|smote_plain|dtw|out_domain|123|0.5"
  "Sv_sp_mi_mm_r05_s42|LONG|smote_plain|mmd|in_domain|42|0.5"
  "Sv_sp_mi_mm_r05_s123|SINGLE|smote_plain|mmd|in_domain|123|0.5"
  "Sv_sp_mo_mm_r05_s42|DEFAULT|smote_plain|mmd|out_domain|42|0.5"
  "Sv_sp_mo_mm_r05_s123|LONG|smote_plain|mmd|out_domain|123|0.5"
  "Sv_sp_wi_wa_r05_s42|SINGLE|smote_plain|wasserstein|in_domain|42|0.5"
  "Sv_sp_wo_wa_r05_s42|DEFAULT|smote_plain|wasserstein|out_domain|42|0.5"
)

NEXT_IDX=0
TOTAL=${#JOBS[@]}

echo "[DAEMON] SvmA exp3 auto-submitter started at $(date)"
echo "[DAEMON] Total jobs to submit: $TOTAL"
echo "[DAEMON] Max user jobs: $MAX_JOBS"
echo "[DAEMON] Check interval: ${CHECK_INTERVAL}s"
echo ""

while [[ $NEXT_IDX -lt $TOTAL ]]; do
    # Count current user jobs
    CURRENT=$(qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l)

    if [[ $CURRENT -lt $MAX_JOBS ]]; then
        SLOTS=$(( MAX_JOBS - CURRENT ))
        echo "[$(date '+%H:%M:%S')] Current jobs: $CURRENT / $MAX_JOBS — $SLOTS slot(s) available"

        # Submit as many as we have slots
        while [[ $SLOTS -gt 0 && $NEXT_IDX -lt $TOTAL ]]; do
            IFS='|' read -r NAME QUEUE CONDITION DISTANCE DOMAIN SEED RATIO <<< "${JOBS[$NEXT_IDX]}"

            echo "[SUBMIT] ($((NEXT_IDX+1))/$TOTAL) $NAME → $QUEUE"
            qsub -N "$NAME" \
                -l "$COMMON_SELECT" \
                -l "$COMMON_WALLTIME" \
                -q "$QUEUE" \
                -v "MODEL=SvmA,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=$RATIO" \
                "$PBS_SCRIPT"

            RESULT=$?
            if [[ $RESULT -eq 0 ]]; then
                NEXT_IDX=$((NEXT_IDX + 1))
                SLOTS=$((SLOTS - 1))
                sleep 0.3
            else
                echo "[WARN] qsub failed (rc=$RESULT), will retry next cycle"
                break
            fi
        done
    else
        echo "[$(date '+%H:%M:%S')] Queue full ($CURRENT/$MAX_JOBS). Remaining: $((TOTAL - NEXT_IDX))/$TOTAL. Waiting ${CHECK_INTERVAL}s..."
    fi

    # If all submitted, break
    [[ $NEXT_IDX -ge $TOTAL ]] && break

    sleep $CHECK_INTERVAL
done

echo ""
echo "[DAEMON] All $TOTAL SvmA exp3 jobs submitted. Exiting at $(date)."

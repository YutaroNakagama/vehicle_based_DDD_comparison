#!/bin/bash
# Resubmit 20 failed SvmA jobs, waiting for queue slots
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SPLIT2_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
export TMPDIR="$HOME/tmp"; mkdir -p "$TMPDIR"

QUEUES=("SINGLE" "LONG" "DEFAULT" "SMALL")
QI=0
next_q() { echo "${QUEUES[$((QI++ % ${#QUEUES[@]}))]}" ; }

MAX_TOTAL=165
POLL=120

wait_for_slot() {
    while true; do
        local n=$(qstat -u "$USER" 2>/dev/null | awk '$1 ~ /^[0-9]+\./ {c++} END {print c+0}')
        if [[ $n -lt $MAX_TOTAL ]]; then
            return
        fi
        echo "[WAIT] $n/$MAX_TOTAL jobs — sleeping ${POLL}s ($(date +%H:%M:%S))"
        sleep "$POLL"
    done
}

submit() {
    local NAME="$1" COND="$2" DIST="$3" DOM="$4" MODE="$5" SEED="$6" RATIO="${7:-}"
    wait_for_slot

    local VARS="MODEL=SvmA,CONDITION=$COND,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=100,RANKING=knn,RUN_EVAL=true"
    [[ -n "$RATIO" ]] && VARS="$VARS,RATIO=$RATIO"

    local attempts=0
    while true; do
        local QUEUE=$(next_q)
        JOB_ID=$(qsub -N "$NAME" \
            -l "select=1:ncpus=8:mem=32gb" -l "walltime=24:00:00" \
            -q "$QUEUE" -v "$VARS" "$SPLIT2_SCRIPT" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[OK] $NAME → $QUEUE → $JOB_ID"
            return
        fi
        ((attempts++))
        if [[ $attempts -ge ${#QUEUES[@]} ]]; then
            echo "[FULL] All queues full for $NAME — waiting ${POLL}s"
            sleep "$POLL"
            attempts=0
        fi
    done
}

echo "=== SvmA failed-job resubmission ($(date)) ==="

submit Sa_un_mi_s_r0.5_s42 undersample mmd in_domain source_only 42 0.5

submit Sa_bs_mi_s_s42  baseline mmd in_domain source_only 42
submit Sa_bs_mi_s_s123 baseline mmd in_domain source_only 123
submit Sa_bs_mi_t_s42  baseline mmd in_domain target_only 42
submit Sa_bs_mi_t_s123 baseline mmd in_domain target_only 123

submit Sa_bs_do_s_s42  baseline dtw out_domain source_only 42
submit Sa_bs_do_s_s123 baseline dtw out_domain source_only 123
submit Sa_bs_do_t_s42  baseline dtw out_domain target_only 42
submit Sa_bs_do_t_s123 baseline dtw out_domain target_only 123

submit Sa_bs_di_s_s42  baseline dtw in_domain source_only 42
submit Sa_bs_di_s_s123 baseline dtw in_domain source_only 123
submit Sa_bs_di_t_s42  baseline dtw in_domain target_only 42
submit Sa_bs_di_t_s123 baseline dtw in_domain target_only 123

submit Sa_bs_wo_s_s42  baseline wasserstein out_domain source_only 42
submit Sa_bs_wo_s_s123 baseline wasserstein out_domain source_only 123
submit Sa_bs_wo_t_s42  baseline wasserstein out_domain target_only 42
submit Sa_bs_wo_t_s123 baseline wasserstein out_domain target_only 123

submit Sa_bs_wi_s_s42  baseline wasserstein in_domain source_only 42
submit Sa_bs_wi_s_s123 baseline wasserstein in_domain source_only 123
submit Sa_bs_wi_t_s42  baseline wasserstein in_domain target_only 42
submit Sa_bs_wi_t_s123 baseline wasserstein in_domain target_only 123

echo "=== All 20 resubmitted ($(date)) ==="

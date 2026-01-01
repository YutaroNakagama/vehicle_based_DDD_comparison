#!/bin/bash
# 失敗した5ジョブの再実行（DEFAULTキュー、8時間）

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$(dirname "$0")"

echo "============================================================"
echo "Rerun Failed Jobs (DEFAULT queue, 8h walltime)"
echo "Time: $(date)"
echo "============================================================"

submit_job() {
    local METHOD="$1"
    local SEED="$2"
    local RATIO="$3"
    
    [[ "$METHOD" == "smote_balanced_rf" ]] && MODEL="BalancedRF" || MODEL="RF"
    
    if [[ "$RATIO" == "default" ]]; then
        TAG="imbal_v2_${METHOD}_seed${SEED}"
    else
        TAG="imbal_v2_${METHOD}_ratio${RATIO//./_}_seed${SEED}"
    fi
    
    QUEUE="DEFAULT"
    MEM="4gb"
    WALLTIME="08:00:00"
    NCPUS="4"
    
    TRAIN_JOB=$(qsub -q "$QUEUE" \
        -l select=1:ncpus=${NCPUS}:mem=${MEM} \
        -l walltime=${WALLTIME} \
        -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL="$MODEL",RATIO="$RATIO",METHOD="$METHOD",TAG="$TAG",SEED="$SEED" \
        pbs_train_generic_ratio.sh)
    
    TRAIN_ID="${TRAIN_JOB%%.*}"
    echo "  [$METHOD] seed=$SEED ratio=$RATIO -> $TRAIN_ID"
    
    # Submit evaluation job
    qsub -q DEFAULT \
        -l select=1:ncpus=2:mem=4gb \
        -l walltime=02:00:00 \
        -W depend=afterok:$TRAIN_JOB \
        -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL="$MODEL",TAG="$TAG",TRAIN_JOBID="$TRAIN_ID",SEED="$SEED" \
        pbs_evaluate.sh
    
    sleep 0.5
}

echo ""
echo "=== Rerunning 5 failed jobs ==="

# baseline (3 seeds)
submit_job "baseline" "42" "default"
submit_job "baseline" "123" "default"
submit_job "baseline" "456" "default"

# undersample_rus ratio=0.1 seed=42
submit_job "undersample_rus" "42" "0.1"

# undersample_tomek ratio=0.1 seed=42
submit_job "undersample_tomek" "42" "0.1"

echo ""
echo "============================================================"
echo "Done! 5 training + 5 evaluation jobs submitted"
echo "============================================================"

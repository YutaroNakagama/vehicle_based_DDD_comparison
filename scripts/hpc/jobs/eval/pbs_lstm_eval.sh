#!/bin/bash
# ============================================================
# PBS Lstm Evaluation Job
# Usage: qsub -v MODEL=Lstm,MODE=target_only,TAG=prior_...,JOBID=12345 pbs_lstm_eval.sh
# ============================================================
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=01:00:00
#PBS -q SEMINAR
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/results/outputs/evaluation/logs/

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "============================================================"
echo "  Lstm Evaluation Job"
echo "  MODEL=$MODEL  MODE=$MODE  JOBID=$JOBID"
echo "  TAG=$TAG"
echo "  $(date)"
echo "============================================================"

python scripts/python/evaluation/evaluate.py \
    --model "$MODEL" \
    --tag "$TAG" \
    --mode "$MODE" \
    --jobid "$JOBID" 2>&1

EXIT_CODE=$?
echo ""
echo "[$(date)] Evaluation finished with exit code $EXIT_CODE"
exit $EXIT_CODE

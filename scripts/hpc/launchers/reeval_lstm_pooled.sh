#!/bin/bash
#PBS -N reeval_lstm_pooled
#PBS -q SINGLE
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/models/Lstm/reeval_pooled.log

set -uo pipefail
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "=== Lstm pooled re-evaluation ($(date)) ==="

python scripts/python/evaluation/evaluate.py \
    --model Lstm --tag prior_research_s42 --mode pooled --seed 42 --jobid 14674645 2>&1 | tail -5
echo "--- s42 done ---"

python scripts/python/evaluation/evaluate.py \
    --model Lstm --tag prior_research_s123 --mode pooled --seed 123 --jobid 14674646 2>&1 | tail -5
echo "--- s123 done ---"

echo "=== Finished: $(date) ==="

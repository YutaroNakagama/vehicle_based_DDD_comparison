#!/bin/bash
#PBS -N train_job
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/jobs/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/jobs/log/

set -euo pipefail

# ===== env =====
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

PROJECT_ROOT="${PBS_O_WORKDIR}"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-RF}"   # qsub -v MODEL=... で外から指定
SEED="${SEED:-42}"

echo "=== JOB START (MODEL=$MODEL, SEED=$SEED) ==="

python bin/train.py \
    --model "$MODEL" \
    --subject_wise_split \
    --seed "$SEED"

echo "=== JOB DONE (MODEL=$MODEL) ==="

# ===== ログを整理 =====
LOG_BASE="$PBS_O_WORKDIR/jobs/log/train_${MODEL}_${PBS_JOBID}.log"
cat /home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/jobs/log/train_job.o${PBS_JOBID} \
    > "$LOG_BASE" 2>/dev/null || true


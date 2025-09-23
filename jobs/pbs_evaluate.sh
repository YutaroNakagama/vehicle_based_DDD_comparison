#!/bin/bash
#PBS -N eval_RF
#PBS -J 1-5
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=02:00:00
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

# ===== settings =====
MODEL="RF"

# タグ一覧（PBS_ARRAY_INDEX=1..5 に対応）
TAG_LIST=(
  "rank_mmd_mean_middle_only_target"
  "rank_mmd_std_high_only_target"
  "rank_wasserstein_mean_low_only_target"
  "rank_wasserstein_std_middle_only_target"
  "rank_dtw_mean_high_only_target"
)

TAG="${TAG_LIST[$PBS_ARRAY_INDEX-1]}"

# ===== run evaluation =====
python bin/evaluate.py \
  --model "$MODEL" \
  --tag "$TAG" \
  --subject_wise_split


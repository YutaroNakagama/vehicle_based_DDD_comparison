#!/bin/bash
#PBS -N only10_6groups
#PBS -J 1-6
#PBS -l select=1:ncpus=8:mem=128gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -q DEFAULT
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/jobs/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/jobs/log/

set -euo pipefail

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

# ★ここがポイント：spoolではなく送信元ディレクトリを起点にする
PROJECT_ROOT="${PBS_O_WORKDIR}"
cd "$PROJECT_ROOT"

NAMES_FILE="$PROJECT_ROOT/misc/pretrain_groups/group_names.txt"
if [[ ! -f "$NAMES_FILE" ]]; then
  echo "NG: $NAMES_FILE not found"; exit 1
fi

IDX="${PBS_ARRAY_INDEX}"
NAME="$(sed -n "${IDX}p" "$NAMES_FILE" | tr -d '\r\n' || true)"
if [[ -z "${NAME:-}" ]]; then
  echo "NG: group name empty for index=$IDX"; exit 1
fi

GROUP_FILE="$PROJECT_ROOT/misc/pretrain_groups/${NAME}.txt"
if [[ ! -f "$GROUP_FILE" ]]; then
  echo "NG: $GROUP_FILE not found"; exit 1
fi

# 10名のIDをスペース区切りに
TARGETS="$(tr '\n' ' ' < "$GROUP_FILE" | sed 's/[[:space:]]\+/ /g' | sed 's/^ //; s/ $//')"
if [[ -z "$TARGETS" ]]; then
  echo "NG: targets empty in $GROUP_FILE"; exit 1
fi

echo "IDX=$IDX"
echo "NAME=$NAME"
echo "GROUP_FILE=$GROUP_FILE"
echo "TARGETS(first few)=$(echo "$TARGETS" | cut -c1-160)"

python "$PROJECT_ROOT/bin/train.py" \
  --model RF \
  --subject_split_strategy subject_time_split \
  --target_subjects $TARGETS \
  --tag "only10_${NAME}"


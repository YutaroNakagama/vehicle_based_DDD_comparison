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

cd "$PBS_O_WORKDIR"
ROOT="$PBS_O_WORKDIR"

GROUP_NAMES_FILE="$ROOT/misc/pretrain_groups/group_names.txt"
SUBJECT_LIST_FILE="$ROOT/misc/subject_list.txt"

if [[ ! -f "$GROUP_NAMES_FILE" ]]; then echo "NG: $GROUP_NAMES_FILE not found"; exit 1; fi
if [[ ! -f "$SUBJECT_LIST_FILE" ]]; then echo "NG: $SUBJECT_LIST_FILE not found"; exit 1; fi

IDX="${PBS_ARRAY_INDEX}"
NAME="$(sed -n "${IDX}p" "$GROUP_NAMES_FILE" | tr -d '\r\n')"
GROUP_FILE="$ROOT/misc/pretrain_groups/${NAME}.txt"
if [[ -z "${NAME}" ]]; then echo "NG: group name empty (index=$IDX)"; exit 1; fi
if [[ ! -f "$GROUP_FILE" ]]; then echo "NG: $GROUP_FILE not found"; exit 1; fi

TARGETS="$(tr '\n' ' ' < "$GROUP_FILE")"
echo "IDX=$IDX"
echo "NAME=$NAME"
echo "GROUP_FILE=$GROUP_FILE"
echo "TARGETS(first few)=$(echo $TARGETS | cut -c1-120)"

python "$ROOT/bin/train.py" \
  --model RF \
  --subject_split_strategy subject_time_split \
  --target_subjects $TARGETS \
  --tag only10_${NAME}


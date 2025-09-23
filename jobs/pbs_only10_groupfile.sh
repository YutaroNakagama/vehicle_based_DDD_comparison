#!/bin/bash
#PBS -N only10_rankfiles
#PBS -J 1-18
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

PROJECT_ROOT="${PBS_O_WORKDIR}"
cd "$PROJECT_ROOT"

#NAMES_FILE="$PROJECT_ROOT/misc/pretrain_groups/group_names.txt"
#NAMES_FILE="$PROJECT_ROOT/misc/pretrain_groups/group_names_all.txt"
# === New behavior ===
# NAMES_FILE now contains *paths* to the rank txt files (absolute or relative to PROJECT_ROOT),
# e.g.:
#   results/ranks/dtw_mean_high.txt
#   results/ranks/dtw_mean_low.txt
#   results/ranks/mmd_std_middle.txt
# You can generate it with: ls results/ranks/*.txt > misc/pretrain_groups/rank_names.txt
NAMES_FILE="${NAMES_FILE:-$PROJECT_ROOT/misc/pretrain_groups/rank_names.txt}"

if [[ ! -f "$NAMES_FILE" ]]; then
  echo "NG: $NAMES_FILE not found"; exit 1
fi

IDX="${PBS_ARRAY_INDEX}"
#NAME="$(sed -n "${IDX}p" "$NAMES_FILE" | tr -d '\r\n' || true)"
#if [[ -z "${NAME:-}" ]]; then
#  echo "NG: group name empty for index=$IDX"; exit 1
GROUP_FILE_REL_OR_ABS="$(sed -n "${IDX}p" "$NAMES_FILE" | tr -d '\r\n' || true)"
if [[ -z "${GROUP_FILE_REL_OR_ABS:-}" ]]; then
  echo "NG: group file path empty for index=$IDX"; exit 1
fi

#GROUP_FILE="$PROJECT_ROOT/misc/pretrain_groups/${NAME}.txt"
#if [[ ! -f "$GROUP_FILE" ]]; then
# Resolve to absolute path if a relative path is given
if [[ "$GROUP_FILE_REL_OR_ABS" = /* ]]; then
  GROUP_FILE="$GROUP_FILE_REL_OR_ABS"
else
  GROUP_FILE="$PROJECT_ROOT/$GROUP_FILE_REL_OR_ABS"
fi

if [[ ! -f "$GROUP_FILE" ]]; then
  echo "NG: $GROUP_FILE not found"; exit 1
fi

TARGETS="$(tr '\n' ' ' < "$GROUP_FILE" | sed 's/[[:space:]]\+/ /g' | sed 's/^ //; s/ $//')"
if [[ -z "$TARGETS" ]]; then
  echo "NG: targets empty in $GROUP_FILE"; exit 1
fi

# Derive a short name from file basename (without .txt) for tagging
BASE="$(basename "$GROUP_FILE" .txt)"

echo "IDX=$IDX"
#echo "NAME=$NAME"
echo "GROUP_FILE=$GROUP_FILE"
echo "BASE=$BASE"
echo "TARGETS(first few)=$(echo "$TARGETS" | cut -c1-160)"

python "$PROJECT_ROOT/bin/train.py" \
  --model RF \
  --subject_split_strategy subject_time_split \
  --target_subjects $TARGETS \
#  --tag "only10_${NAME}" \
  --tag "only10_${BASE}" \
  --time_stratify_labels \
  --time_stratify_tolerance 0.02 \
  --time_stratify_window 0.10 \
  --time_stratify_min_chunk 200


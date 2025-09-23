#!/bin/bash
#PBS -N finetune_groups
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

PROJECT_ROOT="${PBS_O_WORKDIR}"
cd "$PROJECT_ROOT"

# ★ ここを group_names_all.txt に。環境変数で上書きも可
GROUP_NAMES_FILE="${GROUP_NAMES_FILE:-$PROJECT_ROOT/misc/pretrain_groups/group_names_all.txt}"
if [[ ! -f "$GROUP_NAMES_FILE" ]]; then
  echo "NG: $GROUP_NAMES_FILE not found"; exit 1
fi

IDX="${PBS_ARRAY_INDEX}"
NAME="$(sed -n "${IDX}p" "$GROUP_NAMES_FILE" | tr -d '\r\n' || true)"
if [[ -z "${NAME:-}" ]]; then
  echo "NG: group name empty (index=$IDX)"; exit 1
fi

GROUP_FILE="$PROJECT_ROOT/misc/pretrain_groups/${NAME}.txt"
if [[ ! -f "$GROUP_FILE" ]]; then
  echo "NG: $GROUP_FILE not found"; exit 1
fi

TARGETS="$(tr '\n' ' ' < "$GROUP_FILE" | sed 's/[[:space:]]\+/ /g' | sed 's/^ //; s/ $//')"
if [[ -z "$TARGETS" ]]; then
  echo "NG: targets empty in $GROUP_FILE"; exit 1
fi

echo "IDX=$IDX"
echo "NAME=$NAME"
echo "GROUP_FILE=$GROUP_FILE"
echo "TARGETS(first few)=$(echo "$TARGETS" | cut -c1-160)"

PRESET_DIR="$PROJECT_ROOT/model/common"
mkdir -p "$PRESET_DIR"
PRESET="$PRESET_DIR/pretrain_setting_${NAME}.pkl"

SAVE_PRETRAIN_ARGS=()
if [[ ! -f "$PRESET" ]]; then
  echo "pretrain_setting not found. It will be created at: $PRESET"
  SAVE_PRETRAIN_ARGS=(--save_pretrain "$PRESET")
else
  echo "pretrain_setting exists. Reusing: $PRESET"
fi

# Optional: enable "evaluate-only" mode (no fine-tuning on target subjects)
EVAL_ONLY_FLAG=()
if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  EVAL_ONLY_FLAG=(--eval_only_pretrained)
fi

python "$PROJECT_ROOT/bin/train.py" \
  --model RF \
  --subject_split_strategy finetune_target_subjects \
  --target_subjects $TARGETS \
  "${SAVE_PRETRAIN_ARGS[@]}" \
  --finetune_setting "$PRESET" \
  "${EVAL_ONLY_FLAG[@]}" \
  --tag "finetune_${NAME}" \
  --seed 42


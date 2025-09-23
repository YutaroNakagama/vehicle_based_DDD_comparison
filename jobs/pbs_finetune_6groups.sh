#!/bin/bash
#PBS -N finetune_6groups
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

# ★送信元を起点に（spool対策）
PROJECT_ROOT="${PBS_O_WORKDIR}"
cd "$PROJECT_ROOT"

# 入力
GROUP_NAMES_FILE="$PROJECT_ROOT/misc/pretrain_groups/group_names.txt"
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

# 10名をスペース区切りに
TARGETS="$(tr '\n' ' ' < "$GROUP_FILE" | sed 's/[[:space:]]\+/ /g' | sed 's/^ //; s/ $//')"
if [[ -z "$TARGETS" ]]; then
  echo "NG: targets empty in $GROUP_FILE"; exit 1
fi

echo "IDX=$IDX"
echo "NAME=$NAME"
echo "GROUP_FILE=$GROUP_FILE"
echo "TARGETS(first few)=$(echo "$TARGETS" | cut -c1-160)"

# === pretrain設定ファイル ===
# 既存を再利用、無ければ作成（上書きしたくないのでこの方針）
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

# 実行
python "$PROJECT_ROOT/bin/train.py" \
  --model RF \
  --subject_split_strategy finetune_target_subjects \
  --target_subjects $TARGETS \
  "${SAVE_PRETRAIN_ARGS[@]}" \
  --finetune_setting "$PRESET" \
  --tag "finetune_${NAME}" \
  --seed 42


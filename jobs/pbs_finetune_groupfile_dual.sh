#!/bin/bash
#PBS -N finetune_groups_dual
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

# === 設定 ===
GROUP_NAMES_FILE="${GROUP_NAMES_FILE:-$PROJECT_ROOT/misc/pretrain_groups/group_names_all.txt}"
SEED="${SEED:-42}"
MODEL="${MODEL:-RF}"
SPLIT_STRATEGY="finetune_target_subjects"

# 実行順序: eval-only → fine-tune（必要に応じて変更可）
RUN_ORDER=("evalonly" "finetune")

# === 入力確認 ===
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

echo "=== JOB INFO ==="
echo "IDX=$IDX"
echo "NAME=$NAME"
echo "GROUP_FILE=$GROUP_FILE"
echo "TARGETS(first few)=$(echo "$TARGETS" | cut -c1-160)"

# === 事前学習プリセット（features/scaler）の準備 ===
PRESET_DIR="$PROJECT_ROOT/model/common"
mkdir -p "$PRESET_DIR"
PRESET="$PRESET_DIR/pretrain_setting_${NAME}.pkl"

SAVE_PRETRAIN_ARGS=()
if [[ ! -f "$PRESET" ]]; then
  echo "[PRETRAIN] pretrain_setting not found. It will be created at: $PRESET"
  SAVE_PRETRAIN_ARGS=(--save_pretrain "$PRESET")
else
  echo "[PRETRAIN] pretrain_setting exists. Reusing: $PRESET"
fi

# 共通タグ（両実験で揃える）
TAG="finetune_${NAME}"

run_evalonly () {
  echo "----- RUN: EVAL-ONLY (no fine-tune) -----"
  python "$PROJECT_ROOT/bin/train.py" \
    --model "$MODEL" \
    --subject_split_strategy "$SPLIT_STRATEGY" \
    --target_subjects $TARGETS \
    "${SAVE_PRETRAIN_ARGS[@]}" \
    --finetune_setting "$PRESET" \
    --eval_only_pretrained \
    --tag "$TAG" \
    --seed "$SEED"
}

run_finetune () {
  echo "----- RUN: FINE-TUNE (use target-train) -----"
  python "$PROJECT_ROOT/bin/train.py" \
    --model "$MODEL" \
    --subject_split_strategy "$SPLIT_STRATEGY" \
    --target_subjects $TARGETS \
    "${SAVE_PRETRAIN_ARGS[@]}" \
    --finetune_setting "$PRESET" \
    --tag "$TAG" \
    --seed "$SEED"
}

# 実行
for phase in "${RUN_ORDER[@]}"; do
  case "$phase" in
    evalonly)  run_evalonly ;;
    finetune)  run_finetune ;;
    *) echo "Unknown phase: $phase"; exit 1 ;;
  esac
done

echo "=== DONE (NAME=$NAME) ==="


#!/bin/bash
#PBS -N rank_dual
#PBS -J 1-9
#PBS -l select=1:ncpus=8:mem=128gb
#PBS -l walltime=24:00:00
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
RANK_NAMES_FILE="${RANK_NAMES_FILE:-$PROJECT_ROOT/misc/pretrain_groups/rank_names.txt}"
SEED="${SEED:-42}"
MODEL="${MODEL:-RF}"

# 実行順序：必要に応じて変更（only_general=general→targetに当てる、finetune=generalで前処理→targetで微調整、only_target=targetのみ学習）
RUN_ORDER=("only_general" "finetune" "only_target")

# 任意：時間ストラタムを使いたいなら true に（train.py のフラグにそのまま渡す）
TIME_STRATIFY="${TIME_STRATIFY:-false}"
TIME_ARGS=()
if [[ "$TIME_STRATIFY" == "true" ]]; then
  TIME_ARGS=(--time_stratify_labels --time_stratify_tolerance 0.02 --time_stratify_window 0.10 --time_stratify_min_chunk 100)
fi

# ===== inputs =====
if [[ ! -f "$RANK_NAMES_FILE" ]]; then
  echo "NG: $RANK_NAMES_FILE not found"; exit 1
fi

IDX="${PBS_ARRAY_INDEX}"
NAME="$(sed -n "${IDX}p" "$RANK_NAMES_FILE" | tr -d '\r\n' || true)"
if [[ -z "${NAME:-}" ]]; then
  echo "NG: name empty (index=$IDX)"; exit 1
fi

GROUP_FILE="$PROJECT_ROOT/$NAME"   # results/ranks/xxx.txt を想定
if [[ ! -f "$GROUP_FILE" ]]; then
  echo "NG: $GROUP_FILE not found"; exit 1
fi

BASENAME="$(basename "$GROUP_FILE" .txt)"
TARGETS="$(tr '\n' ' ' < "$GROUP_FILE" | sed 's/[[:space:]]\+/ /g' | sed 's/^ //; s/ $//')"
if [[ -z "$TARGETS" ]]; then
  echo "NG: targets empty in $GROUP_FILE"; exit 1
fi

echo "=== JOB INFO ==="
echo "IDX=$IDX"
echo "GROUP_FILE=$GROUP_FILE"
echo "BASENAME=$BASENAME"
echo "TARGETS(first few)=$(echo "$TARGETS" | cut -c1-160)"

# ===== finetune 用の pretrain 設定（features+scaler） =====
PRESET_DIR="$PROJECT_ROOT/model/common"
mkdir -p "$PRESET_DIR"
PRESET="$PRESET_DIR/pretrain_setting_${BASENAME}.pkl"

SAVE_PRETRAIN_ARGS=()
if [[ ! -f "$PRESET" ]]; then
  echo "[PRETRAIN] pretrain_setting not found. It will be created at: $PRESET"
  SAVE_PRETRAIN_ARGS=(--save_pretrain "$PRESET")
else
  echo "[PRETRAIN] pretrain_setting exists. Reusing: $PRESET"
fi

# 共通タグ（ファイル名suffixのグルーピング用）
TAG="rank_${BASENAME}"

run_only_general () {
  echo "----- RUN: ONLY_GENERAL (general model → target 評価のみ) -----"
  python "$PROJECT_ROOT/bin/train.py" \
    --model "$MODEL" \
    --mode only_general \
    --target_subjects $TARGETS \
    --tag "$TAG" \
    --seed "$SEED" \
    "${TIME_ARGS[@]}"
}

run_finetune () {
  echo "----- RUN: FINETUNE (general前処理→targetで微調整) -----"
  python "$PROJECT_ROOT/bin/train.py" \
    --model "$MODEL" \
    --mode finetune \
    --target_subjects $TARGETS \
    "${SAVE_PRETRAIN_ARGS[@]}" \
    --finetune_setting "$PRESET" \
    --tag "$TAG" \
    --seed "$SEED" \
    "${TIME_ARGS[@]}"
}

run_only_target () {
  echo "----- RUN: ONLY_TARGET (targetのみで学習→評価) -----"
  python "$PROJECT_ROOT/bin/train.py" \
    --model "$MODEL" \
    --mode only_target \
    --target_subjects $TARGETS \
    --tag "$TAG" \
    --seed "$SEED" \
    "${TIME_ARGS[@]}"
}

# ===== execute =====
for phase in "${RUN_ORDER[@]}"; do
  case "$phase" in
    only_general) run_only_general ;;
    finetune)     run_finetune ;;
    only_target)  run_only_target ;;
    *) echo "Unknown phase: $phase"; exit 1 ;;
  esac
done

echo "=== DONE (BASENAME=$BASENAME) ==="


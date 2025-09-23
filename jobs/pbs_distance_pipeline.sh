#!/bin/bash
#PBS -N ddd_dist_pipeline
#PBS -l select=1:ncpus=16
#PBS -j oe
#PBS -q SINGLE

set -euo pipefail

# ==== 移動：qsub した場所(=プロジェクトルート)へ ====
if [[ -n "${PBS_O_WORKDIR:-}" && -d "$PBS_O_WORKDIR" ]]; then
  cd "$PBS_O_WORKDIR"
else
  echo "[ERROR] PBS_O_WORKDIR is not set. Submit this job from the project root dir."
  exit 1
fi

echo "[INFO] Working dir: $(pwd)"

# ==== ユーザー設定（プロジェクト直下基準）====
CONDA_ENV="python310"

# 距離計算の入力
SUBJECT_LIST="../../dataset/mdapbe/subject_list.txt"
DATA_ROOT="data/processed/common"
GROUPS_FILE="../misc/target_groups.txt"   # ← ../ を外す

# 相関分析の入力（summary CSV が無ければ自動スキップ）
SUMMARY_CSV="model/common/summary_6groups_only10_vs_finetune_wide.csv"
DISTANCE_NPY="results/distances/wasserstein_matrix.npy"
SUBJECTS_JSON="results/distances/subjects.json"
OUTDIR="model/common/dist_corr"
GROUPS_DIR="misc/pretrain_groups"
GROUP_NAMES_FILE="misc/pretrain_groups/group_names.txt"

# ==== 環境セットアップ ====
export PATH="$HOME/conda/bin:$PATH"
source "$HOME/conda/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1

echo "[INFO] Python: $(which python)"
python -V

# 依存（requirements があれば使用）
if [[ -f "misc/requirements.txt" ]]; then
  echo "[INFO] Installing requirements.txt ..."
  pip install -r misc/requirements.txt
else
  echo "[INFO] Installing minimal deps ..."
  pip install numpy pandas scikit-learn matplotlib seaborn tqdm scipy tslearn
fi

# ==== 1) 距離行列の計算 ====
echo "[STEP 1] Computing distance matrices..."
python bin/analyze.py comp-dist \
  --subject_list "$SUBJECT_LIST" \
  --data_root "$DATA_ROOT" \
  --groups_file "$GROUPS_FILE"

# ==== 2) 相関分析（SUMMARY_CSV があれば実行）====
if [[ -f "$SUMMARY_CSV" ]]; then
  echo "[STEP 2] Running correlation analysis..."
  # フォールバック（念のため）
  [[ -f "$SUBJECTS_JSON" ]] || SUBJECTS_JSON="results/distances/subjects.json"
  [[ -f "$DISTANCE_NPY"  ]] || DISTANCE_NPY="results/distances/wasserstein_matrix.npy"

  python bin/analyze.py corr \
    --summary_csv "$SUMMARY_CSV" \
    --distance "$DISTANCE_NPY" \
    --subjects_json "$SUBJECTS_JSON" \
    --groups_dir "$GROUPS_DIR" \
    --group_names_file "$GROUP_NAMES_FILE" \
    --outdir "$OUTDIR"
else
  echo "[SKIP] SUMMARY_CSV not found: $SUMMARY_CSV"
fi

echo "[DONE] All steps finished."


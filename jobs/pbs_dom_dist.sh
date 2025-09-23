#!/bin/bash
#PBS -N domain_def
#PBS -l select=1:ncpus=16
#PBS -j oe
#PBS -q SINGLE

set -euo pipefail

# === 作業ディレクトリを送信元に固定 ===
if [[ -n "${PBS_O_WORKDIR:-}" && -d "$PBS_O_WORKDIR" ]]; then
  cd "$PBS_O_WORKDIR"
else
  echo "[ERROR] PBS_O_WORKDIR is not set. Submit this job from the project root dir."
  exit 1
fi

# === Conda ===
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

# === スレッド系 & ヘッドレス描画 ===
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1

# === 依存 ===
pip install -r misc/requirements.txt || pip install numpy pandas scikit-learn matplotlib seaborn tqdm scipy tslearn

# === 距離計算（新CLI）===
python bin/analyze.py comp-dist \
  --subject_list ../../dataset/mdapbe/subject_list.txt \
  --data_root data/processed/common \
  --groups_file ../misc/target_groups.txt


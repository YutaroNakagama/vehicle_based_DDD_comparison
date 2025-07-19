#!/bin/bash
#PBS -N preprocess_mp
#PBS -l select=1:ncpus=16
#PBS -j oe
#PBS -q SINGLE

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

export N_PROC=14  # ← 15に下げると99.99%安全（まだkillされるなら）

export PYTHONNOUSERSITE=1

# 隠れ並列の抑制
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1   # ←追加！

cd $PBS_O_WORKDIR

python bin/preprocess.py --model common --multi_process


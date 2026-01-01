#!/bin/bash
#PBS -N data_preprocess_mp
#PBS -l select=1:ncpus=20
#PBS -j oe
#PBS -q SINGLE
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/
#PBS -M yutaro.nakagama@bosch.com
#PBS -m abe

export PATH=~/conda/bin:$PATH
export PYTHONPATH=$PBS_O_WORKDIR:$PYTHONPATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

export N_PROC=14  

export PYTHONNOUSERSITE=1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1   

cd $PBS_O_WORKDIR

python scripts/python/preprocess.py --model ${MODEL:-common} --multi_process

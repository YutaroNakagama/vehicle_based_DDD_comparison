#!/bin/bash
#PBS -N train_group
#PBS -J 1-9         
#PBS -l select=1:ncpus=8:mem=128gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -q SINGLE
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/jobs/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/jobs/log/

# Condaを有効化する（パスは環境によって違う場合あり）
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310   # ←ここはあなたの仮想環境名

cd $PBS_O_WORKDIR

GROUP=$(sed -n "${PBS_ARRAY_INDEX}p" ../misc/target_groups.txt)

python bin/train.py \
    --model RF \
    --subject_split_strategy random \
    --target_subjects $GROUP


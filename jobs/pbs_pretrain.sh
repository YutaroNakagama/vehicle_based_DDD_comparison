#!/bin/bash
#PBS -N pretrain_sin
#PBS -J 1-9         
#PBS -l select=1:ncpus=8:mem=128gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -q SINGLE
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/jobs/log/
#PBS -e /home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/jobs/log/

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

cd $PBS_O_WORKDIR

GROUP_LINE=$(sed -n "${PBS_ARRAY_INDEX}p" ../misc/target_groups.txt)
ALL_SUBJECTS=$(tr '\n' ' ' < ../misc/subject_list.txt)

GENERAL_SUBJECTS=$(for s in $ALL_SUBJECTS; do
    if [[ ! " $GROUP_LINE " =~ " $s " ]]; then echo -n "$s "; fi
done)

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python bin/train.py \
    --model RF \
    --subject_split_strategy finetune_target_subjects \
    --general_subjects $GENERAL_SUBJECTS \
    --target_subjects $GROUP_LINE \
    --save_pretrain pretrain_setting_group${PBS_ARRAY_INDEX}.pkl \
    --tag pretrain_group${PBS_ARRAY_INDEX} \
    --seed 42


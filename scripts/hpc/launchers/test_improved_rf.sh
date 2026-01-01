#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -N test_rf_smote

cd $PBS_O_WORKDIR
source ~/.bashrc
conda activate python310

export PYTHONPATH="${PBS_O_WORKDIR}:${PYTHONPATH}"
export N_TRIALS=5
export KSS_SIMPLIFIED=1
export TRAIN_RATIO=0.4
export VAL_RATIO=0.3
export TEST_RATIO=0.3

echo "=== Test Improved RF with SMOTE ==="
echo "Job ID: $PBS_JOBID"
echo "Date: $(date)"

python scripts/python/train.py \
  --model RF \
  --mode pooled \
  --seed 42 \
  --tag test_smote \
  --use_oversampling \
  --oversample_method smote

echo "Training completed: $(date)"

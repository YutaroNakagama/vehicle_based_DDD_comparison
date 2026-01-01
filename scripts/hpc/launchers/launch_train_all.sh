#!/bin/bash
# Launch training jobs for multiple models

#MODELS=("SvmA" "SvmW" "Lstm" "RF")
MODELS=("RF")

for m in "${MODELS[@]}"; do
  echo "Submitting training job for MODEL=$m"
  qsub -v MODEL=$m scripts/hpc/jobs/train/pbs_train.sh
done


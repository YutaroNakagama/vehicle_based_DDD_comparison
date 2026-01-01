#!/bin/bash
# Launch preprocessing jobs for all models

MODELS=("common" "SvmA" "SvmW" "Lstm")

for m in "${MODELS[@]}"; do
  echo "Submitting preprocess job for MODEL=$m"
  qsub -v MODEL=$m scripts/hpc/jobs/preprocess/pbs_preprocess.sh
done


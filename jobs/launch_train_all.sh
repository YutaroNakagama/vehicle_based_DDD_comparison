#!/bin/bash
# Launch training jobs for multiple models

MODELS=("SvmA" "SvmW" "Lstm" "RF")

for m in "${MODELS[@]}"; do
  echo "Submitting training job for MODEL=$m"
  qsub -v MODEL=$m jobs/pbs_train.sh
done


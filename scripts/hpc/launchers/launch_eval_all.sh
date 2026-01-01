#!/bin/bash
# Launch evaluation jobs for multiple models

#MODELS=("SvmA" "SvmW" "Lstm" "RF")
MODELS=("RF")

for m in "${MODELS[@]}"; do
  echo "Submitting evaluation job for MODEL=$m (MODE=pooled)"
  qsub -v MODEL=$m,MODE=pooled scripts/hpc/jobs/evaluate/pbs_evaluate.sh
done


#!/bin/bash
set -euo pipefail
BASE="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
LOGDIR="$BASE/scripts/hpc/logs/train"
OUT="/tmp/exp3_verification_$(date +%s).txt"
:> "$OUT"

echo "Exp3 verification run: $(date)" | tee -a "$OUT"
for MODEL in Lstm SvmA SvmW; do
  echo "\n=== $MODEL ===" | tee -a "$OUT"
  model_dir="$BASE/models/$MODEL"
  train_dir="$BASE/results/outputs/training/$MODEL"
  eval_dir="$BASE/results/outputs/evaluation/$MODEL"

  # all model dirs (unique, exclude latest and spcc duplicates)
  for jid in $(ls "$model_dir" 2>/dev/null | grep -v spcc | grep -v latest | sort -n); do
    # ensure training output exists (considered completed)
    if [ ! -d "$train_dir/$jid" ]; then
      continue
    fi
    logf="$LOGDIR/${jid}.spcc-adm1.OU"
    mdir="$model_dir/$jid"
    has_model=0
    has_scaler=0
    has_eval=0
    trace=0
    wt=0
    eval_fail=0
    saveerr=0

    # search for model artifacts recursively
    if [ "$MODEL" = "Lstm" ]; then
      if find "$mdir" -type f -iname "*.keras" -o -iname "*.h5" | grep -q .; then has_model=1; fi
    else
      if find "$mdir" -type f -iname "*.pkl" -o -iname "*.joblib" | grep -q .; then has_model=1; fi
    fi
    # scaler
    if find "$mdir" -type f -iname "scaler*.pkl" | grep -q .; then has_scaler=1; fi
    # eval
    if [ -d "$eval_dir/$jid" ]; then has_eval=1; fi
    # logs check
    if [ -f "$logf" ]; then
      if grep -q "Traceback (most recent call last)" "$logf" 2>/dev/null; then trace=1; fi
      if grep -q "job killed: walltime" "$logf" 2>/dev/null; then wt=1; fi
      if grep -q "Evaluation failed" "$logf" 2>/dev/null; then eval_fail=1; fi
      if grep -qi "Model object is None\|skipping save" "$logf" 2>/dev/null; then saveerr=1; fi
    else
      echo "$jid MISSING_LOG" >> "$OUT"
      continue
    fi

    echo "$jid model=$has_model scaler=$has_scaler eval=$has_eval traceback=$trace walltime=$wt eval_fail=$eval_fail saveerr=$saveerr" >> "$OUT"
  done
  echo "(Wrote report to $OUT)"
done

# Summarize SvmW eval failures (detailed)
echo "\n=== SvmW eval failures details ===" | tee -a "$OUT"
grep -R "Evaluation failed" "$LOGDIR" | sed -n '1,200p' | tee -a "$OUT"

echo "\nVerification complete: $OUT" | tee -a "$OUT"

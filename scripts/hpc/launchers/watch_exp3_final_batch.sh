#!/bin/bash
# ============================================================
# Watch progress of exp3 final batch (PBS 15101-15124)
# Run with:  bash scripts/hpc/launchers/watch_exp3_final_batch.sh
# Or in watch mode:  watch -n 60 bash scripts/hpc/launchers/watch_exp3_final_batch.sh
# ============================================================
set -uo pipefail
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

echo "====================== $(date) ======================"
echo
echo "## Active PBS jobs"
qstat -u "$USER" 2>/dev/null | awk 'NR>5' | wc -l | xargs printf "  total active : %s\n"
qstat -u "$USER" 2>/dev/null | awk 'NR>5 {st[$10]++} END {for (s in st) printf "  %s : %d\n", s, st[s]}'
echo
echo "## Final batch status (15101-15124)"
qstat -u "$USER" 2>/dev/null | awk 'NR>5 && $1 ~ /^151[0-2][0-9]/ {printf "  %s  %-22s %-8s %s\n", $1, $4, $3, $10}'
echo

echo "## Canonical tag completion (seeds 42/123)"
gen_expected() {
  local M=$1
  for D in mmd dtw wasserstein; do for DOM in in_domain out_domain; do for S in 42 123; do
    echo "prior_${M}_baseline_knn_${D}_${DOM}_domain_train_split2_s${S}"
    for R in 0.1 0.5; do
      echo "prior_${M}_imbalv3_knn_${D}_${DOM}_domain_train_split2_subjectwise_ratio${R}_s${S}"
      echo "prior_${M}_smote_plain_knn_${D}_${DOM}_domain_train_split2_ratio${R}_s${S}"
      echo "prior_${M}_undersample_rus_knn_${D}_${DOM}_domain_train_split2_ratio${R}_s${S}"
    done
  done; done; done
}
for M in SvmW SvmA Lstm; do
  for ET in within cross; do
    DONE=$(find results/outputs/evaluation/$M -name "eval_results_*_${ET}.json" 2>/dev/null \
      | grep "domain_train_split2" | grep -E "_s(42|123)_${ET}\.json$" \
      | sed -E "s|.*/eval_results_${M}_domain_train_||" | sed -E "s|_${ET}\.json$||" | sort -u)
    DONE_CNT=$(echo "$DONE" | grep -c .)
    MISS=$(comm -23 <(gen_expected "$M" | sort -u) <(echo "$DONE") | grep -c .)
    printf "  %-4s [%s]: %d/84  missing=%d\n" "$M" "$ET" "$DONE_CNT" "$MISS"
  done
done

echo
echo "## Recent PBS error logs (last 30 min)"
find . -maxdepth 1 -name "*.e151[0-2][0-9]" -mmin -30 2>/dev/null | while read f; do
  printf "  %s  size=%s\n" "$f" "$(stat -c %s "$f")"
done

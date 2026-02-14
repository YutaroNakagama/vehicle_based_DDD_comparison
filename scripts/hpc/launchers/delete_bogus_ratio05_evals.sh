#!/bin/bash
# ============================================================
# Delete 34 bogus ratio=0.5 eval files
# ============================================================
# These files were created by batch re-evaluation (pbs_reeval_split2_v2.sh)
# which evaluated ratio=0.1 trained models with a ratio=0.5 tag,
# producing incorrect duplicate results.
# ============================================================
set -euo pipefail

EVAL_DIR="/home/s2240011/git/ddd/vehicle_based_DDD_comparison/results/outputs/evaluation/RF"

FILES=(
  "14740692/14740692[1]/eval_results_RF_source_only_smote_plain_knn_wasserstein_out_domain_source_only_split2_ratio0.5_s123.csv"
  "14740692/14740692[1]/eval_results_RF_source_only_smote_plain_knn_wasserstein_out_domain_source_only_split2_ratio0.5_s123.json"
  "14740737/14740737[1]/eval_results_RF_mixed_undersample_rus_knn_mmd_out_domain_mixed_split2_ratio0.5_s42.csv"
  "14740737/14740737[1]/eval_results_RF_mixed_undersample_rus_knn_mmd_out_domain_mixed_split2_ratio0.5_s42.json"
  "14740743/14740743[1]/eval_results_RF_mixed_smote_plain_knn_mmd_out_domain_mixed_split2_ratio0.5_s123.csv"
  "14740743/14740743[1]/eval_results_RF_mixed_smote_plain_knn_mmd_out_domain_mixed_split2_ratio0.5_s123.json"
  "14740744/14740744[1]/eval_results_RF_mixed_imbalv3_knn_mmd_out_domain_mixed_split2_subjectwise_ratio0.5_s123.csv"
  "14740744/14740744[1]/eval_results_RF_mixed_imbalv3_knn_mmd_out_domain_mixed_split2_subjectwise_ratio0.5_s123.json"
  "14740761/14740761[1]/eval_results_RF_mixed_undersample_rus_knn_mmd_in_domain_mixed_split2_ratio0.5_s123.csv"
  "14740761/14740761[1]/eval_results_RF_mixed_undersample_rus_knn_mmd_in_domain_mixed_split2_ratio0.5_s123.json"
  "14740769/14740769[1]/eval_results_RF_mixed_undersample_rus_knn_dtw_out_domain_mixed_split2_ratio0.5_s42.csv"
  "14740769/14740769[1]/eval_results_RF_mixed_undersample_rus_knn_dtw_out_domain_mixed_split2_ratio0.5_s42.json"
  "14740776/14740776[1]/eval_results_RF_mixed_imbalv3_knn_dtw_out_domain_mixed_split2_subjectwise_ratio0.5_s123.csv"
  "14740776/14740776[1]/eval_results_RF_mixed_imbalv3_knn_dtw_out_domain_mixed_split2_subjectwise_ratio0.5_s123.json"
  "14740791/14740791[1]/eval_results_RF_mixed_smote_plain_knn_dtw_in_domain_mixed_split2_ratio0.5_s123.csv"
  "14740791/14740791[1]/eval_results_RF_mixed_smote_plain_knn_dtw_in_domain_mixed_split2_ratio0.5_s123.json"
  "14740792/14740792[1]/eval_results_RF_mixed_imbalv3_knn_dtw_in_domain_mixed_split2_subjectwise_ratio0.5_s123.csv"
  "14740792/14740792[1]/eval_results_RF_mixed_imbalv3_knn_dtw_in_domain_mixed_split2_subjectwise_ratio0.5_s123.json"
  "14740793/14740793[1]/eval_results_RF_mixed_undersample_rus_knn_dtw_in_domain_mixed_split2_ratio0.5_s123.csv"
  "14740793/14740793[1]/eval_results_RF_mixed_undersample_rus_knn_dtw_in_domain_mixed_split2_ratio0.5_s123.json"
  "14740800/14740800[1]/eval_results_RF_mixed_imbalv3_knn_wasserstein_out_domain_mixed_split2_subjectwise_ratio0.5_s42.csv"
  "14740800/14740800[1]/eval_results_RF_mixed_imbalv3_knn_wasserstein_out_domain_mixed_split2_subjectwise_ratio0.5_s42.json"
  "14740807/14740807[1]/eval_results_RF_mixed_smote_plain_knn_wasserstein_out_domain_mixed_split2_ratio0.5_s123.csv"
  "14740807/14740807[1]/eval_results_RF_mixed_smote_plain_knn_wasserstein_out_domain_mixed_split2_ratio0.5_s123.json"
  "14740808/14740808[1]/eval_results_RF_mixed_imbalv3_knn_wasserstein_out_domain_mixed_split2_subjectwise_ratio0.5_s123.csv"
  "14740808/14740808[1]/eval_results_RF_mixed_imbalv3_knn_wasserstein_out_domain_mixed_split2_subjectwise_ratio0.5_s123.json"
  "14740809/14740809[1]/eval_results_RF_mixed_undersample_rus_knn_wasserstein_out_domain_mixed_split2_ratio0.5_s123.csv"
  "14740809/14740809[1]/eval_results_RF_mixed_undersample_rus_knn_wasserstein_out_domain_mixed_split2_ratio0.5_s123.json"
  "14740816/14740816[1]/eval_results_RF_mixed_imbalv3_knn_wasserstein_in_domain_mixed_split2_subjectwise_ratio0.5_s42.csv"
  "14740816/14740816[1]/eval_results_RF_mixed_imbalv3_knn_wasserstein_in_domain_mixed_split2_subjectwise_ratio0.5_s42.json"
  "14740823/14740823[1]/eval_results_RF_mixed_smote_plain_knn_wasserstein_in_domain_mixed_split2_ratio0.5_s123.csv"
  "14740823/14740823[1]/eval_results_RF_mixed_smote_plain_knn_wasserstein_in_domain_mixed_split2_ratio0.5_s123.json"
  "14740824/14740824[1]/eval_results_RF_mixed_imbalv3_knn_wasserstein_in_domain_mixed_split2_subjectwise_ratio0.5_s123.csv"
  "14740824/14740824[1]/eval_results_RF_mixed_imbalv3_knn_wasserstein_in_domain_mixed_split2_subjectwise_ratio0.5_s123.json"
)

DELETED=0
MISSING=0

for f in "${FILES[@]}"; do
    FULL="${EVAL_DIR}/${f}"
    if [[ -f "$FULL" ]]; then
        rm "$FULL"
        DELETED=$((DELETED + 1))
    else
        echo "[WARN] Not found: $f"
        MISSING=$((MISSING + 1))
    fi
done

echo "[DONE] Deleted: $DELETED, Not found: $MISSING (total: ${#FILES[@]})"

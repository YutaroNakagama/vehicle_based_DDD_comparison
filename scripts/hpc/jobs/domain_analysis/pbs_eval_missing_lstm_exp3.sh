#!/bin/bash
#PBS -N Ls_eval_miss
#PBS -q GPU-1
#PBS -l select=1:ncpus=4:ngpus=1:mem=8gb
#PBS -l walltime=24:00:00
#PBS -J 0-11

# Auto-generated: evaluate missing Lstm exp3 conditions
# 12 eval commands (6 conditions × 2 eval types)

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"
source .venv/bin/activate
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

COMMANDS=(
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_baseline_knn_wasserstein_in_domain_domain_train_split2_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_in_domain.txt --eval_type within --jobid 14873044.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_baseline_knn_wasserstein_in_domain_domain_train_split2_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_out_domain.txt --eval_type cross --jobid 14873044.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_in_domain_domain_train_split2_subjectwise_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_in_domain.txt --eval_type within --jobid 14873018.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_in_domain_domain_train_split2_subjectwise_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_out_domain.txt --eval_type cross --jobid 14873018.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_out_domain_domain_train_split2_subjectwise_ratio0.1_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_out_domain.txt --eval_type within --jobid 14873009.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_out_domain_domain_train_split2_subjectwise_ratio0.1_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_in_domain.txt --eval_type cross --jobid 14873009.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_wasserstein_in_domain_domain_train_split2_subjectwise_ratio0.1_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_in_domain.txt --eval_type within --jobid 14873048.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_wasserstein_in_domain_domain_train_split2_subjectwise_ratio0.1_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_out_domain.txt --eval_type cross --jobid 14873048.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_wasserstein_out_domain_domain_train_split2_subjectwise_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_out_domain.txt --eval_type within --jobid 14873040.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_wasserstein_out_domain_domain_train_split2_subjectwise_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_in_domain.txt --eval_type cross --jobid 14873040.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_smote_plain_knn_dtw_out_domain_domain_train_split2_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_out_domain.txt --eval_type within --jobid 14873010.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_smote_plain_knn_dtw_out_domain_domain_train_split2_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_in_domain.txt --eval_type cross --jobid 14873010.spcc-adm1"
)

echo "Task ${PBS_ARRAY_INDEX}: ${COMMANDS[$PBS_ARRAY_INDEX]}"
eval "${COMMANDS[$PBS_ARRAY_INDEX]}"
EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE

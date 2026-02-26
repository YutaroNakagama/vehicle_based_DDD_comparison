#!/bin/bash
#PBS -N Ls_reeval
#PBS -l select=1:ncpus=4:mem=8gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -q GPU-1
#PBS -J 0-31
#PBS -o /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/logs/train/

set -euo pipefail

# ===== Environment setup =====
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison

# ===== Evaluation commands array =====
CMDS=(
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_baseline_knn_mmd_in_domain_domain_train_split2_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/mmd_in_domain.txt --eval_type within --jobid 14873001.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_baseline_knn_mmd_in_domain_domain_train_split2_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/mmd_out_domain.txt --eval_type cross --jobid 14873001.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_mmd_in_domain_domain_train_split2_subjectwise_ratio0.1_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/mmd_in_domain.txt --eval_type within --jobid 14873002.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_mmd_in_domain_domain_train_split2_subjectwise_ratio0.1_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/mmd_out_domain.txt --eval_type cross --jobid 14873002.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_smote_plain_knn_mmd_in_domain_domain_train_split2_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/mmd_in_domain.txt --eval_type within --jobid 14873003.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_smote_plain_knn_mmd_in_domain_domain_train_split2_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/mmd_out_domain.txt --eval_type cross --jobid 14873003.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_undersample_rus_knn_mmd_in_domain_domain_train_split2_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/mmd_in_domain.txt --eval_type within --jobid 14873004.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_undersample_rus_knn_mmd_in_domain_domain_train_split2_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/mmd_out_domain.txt --eval_type cross --jobid 14873004.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_out_domain.txt --eval_type within --jobid 14873007.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_in_domain.txt --eval_type cross --jobid 14873007.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_baseline_knn_dtw_out_domain_domain_train_split2_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_out_domain.txt --eval_type within --jobid 14873008.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_baseline_knn_dtw_out_domain_domain_train_split2_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_in_domain.txt --eval_type cross --jobid 14873008.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_out_domain.txt --eval_type within --jobid 14873011.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_in_domain.txt --eval_type cross --jobid 14873011.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_baseline_knn_dtw_in_domain_domain_train_split2_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_in_domain.txt --eval_type within --jobid 14873013.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_baseline_knn_dtw_in_domain_domain_train_split2_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_out_domain.txt --eval_type cross --jobid 14873013.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_in_domain_domain_train_split2_subjectwise_ratio0.1_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_in_domain.txt --eval_type within --jobid 14873015.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_in_domain_domain_train_split2_subjectwise_ratio0.1_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_out_domain.txt --eval_type cross --jobid 14873015.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_smote_plain_knn_dtw_in_domain_domain_train_split2_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_in_domain.txt --eval_type within --jobid 14873017.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_smote_plain_knn_dtw_in_domain_domain_train_split2_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_out_domain.txt --eval_type cross --jobid 14873017.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_undersample_rus_knn_dtw_in_domain_domain_train_split2_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_in_domain.txt --eval_type within --jobid 14873019.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_undersample_rus_knn_dtw_in_domain_domain_train_split2_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_out_domain.txt --eval_type cross --jobid 14873019.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_in_domain_domain_train_split2_subjectwise_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_in_domain.txt --eval_type within --jobid 14873025.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_dtw_in_domain_domain_train_split2_subjectwise_ratio0.5_s123 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/dtw_out_domain.txt --eval_type cross --jobid 14873025.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_baseline_knn_wasserstein_out_domain_domain_train_split2_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_out_domain.txt --eval_type within --jobid 14873027.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_baseline_knn_wasserstein_out_domain_domain_train_split2_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_in_domain.txt --eval_type cross --jobid 14873027.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_wasserstein_out_domain_domain_train_split2_subjectwise_ratio0.1_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_out_domain.txt --eval_type within --jobid 14873029.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_imbalv3_knn_wasserstein_out_domain_domain_train_split2_subjectwise_ratio0.1_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_in_domain.txt --eval_type cross --jobid 14873029.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_smote_plain_knn_wasserstein_out_domain_domain_train_split2_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_out_domain.txt --eval_type within --jobid 14873031.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_smote_plain_knn_wasserstein_out_domain_domain_train_split2_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_in_domain.txt --eval_type cross --jobid 14873031.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_undersample_rus_knn_wasserstein_out_domain_domain_train_split2_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_out_domain.txt --eval_type within --jobid 14873033.spcc-adm1"
    "python scripts/python/evaluation/evaluate.py --model Lstm --tag prior_Lstm_undersample_rus_knn_wasserstein_out_domain_domain_train_split2_ratio0.5_s42 --mode domain_train --target_file results/analysis/exp2_domain_shift/distance/rankings/split2/knn/wasserstein_in_domain.txt --eval_type cross --jobid 14873033.spcc-adm1"
)

echo "=== LSTM RE-EVALUATION (PBS_ARRAY_INDEX=$PBS_ARRAY_INDEX) ==="
echo "Command: ${CMDS[$PBS_ARRAY_INDEX]}"

eval "${CMDS[$PBS_ARRAY_INDEX]}"

echo "=== RE-EVALUATION DONE ==="

#!/bin/bash
# ============================================================
# Imbalance Comparison V2: Launch All Jobs
# Submit all 7 training jobs in parallel, then evaluation jobs
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# scripts/hpc/imbalance_comparison_v2 から vehicle_based_DDD_comparison へ
PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Imbalance Comparison V2 - Launching All Jobs"
echo "============================================================"
echo "Project Root: $PROJECT_ROOT"
echo "Time: $(date)"
echo ""

echo "【実験条件】"
echo "  - N_TRIALS: 50"
echo "  - CV: 5-fold"
echo "  - 目的関数: F2スコア"
echo "  - データ分割: 時間ベース層化分割"
echo ""

echo "【手法一覧】"
echo "  1. Baseline (RF, オーバーサンプリングなし)"
echo "  2. SMOTE (RF + SMOTE単体)"
echo "  3. SMOTE+Tomek (RF + SMOTE + Tomek Links)"
echo "  4. SMOTE+ENN (RF + SMOTE + ENN)"
echo "  5. SMOTE+RUS (RF + SMOTE + RandomUnderSampler)"
echo "  6. BalancedRF (BalancedRandomForest)"
echo "  7. EasyEnsemble (アンサンブル手法)"
echo ""

# Submit training jobs (all in parallel)
echo "=== Submitting Training Jobs ==="

JOB_BASELINE=$(qsub -v PBS_O_WORKDIR="$PROJECT_ROOT" pbs_train_baseline.sh)
echo "  Baseline: $JOB_BASELINE"

JOB_SMOTE=$(qsub -v PBS_O_WORKDIR="$PROJECT_ROOT" pbs_train_smote.sh)
echo "  SMOTE: $JOB_SMOTE"

JOB_SMOTE_TOMEK=$(qsub -v PBS_O_WORKDIR="$PROJECT_ROOT" pbs_train_smote_tomek.sh)
echo "  SMOTE+Tomek: $JOB_SMOTE_TOMEK"

JOB_SMOTE_ENN=$(qsub -v PBS_O_WORKDIR="$PROJECT_ROOT" pbs_train_smote_enn.sh)
echo "  SMOTE+ENN: $JOB_SMOTE_ENN"

JOB_SMOTE_RUS=$(qsub -v PBS_O_WORKDIR="$PROJECT_ROOT" pbs_train_smote_rus.sh)
echo "  SMOTE+RUS: $JOB_SMOTE_RUS"

JOB_BALANCED_RF=$(qsub -v PBS_O_WORKDIR="$PROJECT_ROOT" pbs_train_balanced_rf.sh)
echo "  BalancedRF: $JOB_BALANCED_RF"

JOB_EASY_ENSEMBLE=$(qsub -v PBS_O_WORKDIR="$PROJECT_ROOT" pbs_train_easy_ensemble.sh)
echo "  EasyEnsemble: $JOB_EASY_ENSEMBLE"

echo ""
echo "=== Submitting Evaluation Jobs (with dependencies) ==="

# Extract job IDs (remove .spcc-adm1 suffix)
ID_BASELINE=$(echo $JOB_BASELINE | cut -d'.' -f1)
ID_SMOTE=$(echo $JOB_SMOTE | cut -d'.' -f1)
ID_SMOTE_TOMEK=$(echo $JOB_SMOTE_TOMEK | cut -d'.' -f1)
ID_SMOTE_ENN=$(echo $JOB_SMOTE_ENN | cut -d'.' -f1)
ID_SMOTE_RUS=$(echo $JOB_SMOTE_RUS | cut -d'.' -f1)
ID_BALANCED_RF=$(echo $JOB_BALANCED_RF | cut -d'.' -f1)
ID_EASY_ENSEMBLE=$(echo $JOB_EASY_ENSEMBLE | cut -d'.' -f1)

# Submit evaluation jobs with afterok dependency
JOB_EVAL_BASELINE=$(qsub -W depend=afterok:$JOB_BASELINE \
    -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL=RF,TAG=imbal_v2_baseline,TRAIN_JOBID=$ID_BASELINE \
    pbs_evaluate.sh)
echo "  Eval Baseline: $JOB_EVAL_BASELINE (after $ID_BASELINE)"

JOB_EVAL_SMOTE=$(qsub -W depend=afterok:$JOB_SMOTE \
    -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL=RF,TAG=imbal_v2_smote,TRAIN_JOBID=$ID_SMOTE \
    pbs_evaluate.sh)
echo "  Eval SMOTE: $JOB_EVAL_SMOTE (after $ID_SMOTE)"

JOB_EVAL_SMOTE_TOMEK=$(qsub -W depend=afterok:$JOB_SMOTE_TOMEK \
    -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL=RF,TAG=imbal_v2_smote_tomek,TRAIN_JOBID=$ID_SMOTE_TOMEK \
    pbs_evaluate.sh)
echo "  Eval SMOTE+Tomek: $JOB_EVAL_SMOTE_TOMEK (after $ID_SMOTE_TOMEK)"

JOB_EVAL_SMOTE_ENN=$(qsub -W depend=afterok:$JOB_SMOTE_ENN \
    -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL=RF,TAG=imbal_v2_smote_enn,TRAIN_JOBID=$ID_SMOTE_ENN \
    pbs_evaluate.sh)
echo "  Eval SMOTE+ENN: $JOB_EVAL_SMOTE_ENN (after $ID_SMOTE_ENN)"

JOB_EVAL_SMOTE_RUS=$(qsub -W depend=afterok:$JOB_SMOTE_RUS \
    -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL=RF,TAG=imbal_v2_smote_rus,TRAIN_JOBID=$ID_SMOTE_RUS \
    pbs_evaluate.sh)
echo "  Eval SMOTE+RUS: $JOB_EVAL_SMOTE_RUS (after $ID_SMOTE_RUS)"

JOB_EVAL_BALANCED_RF=$(qsub -W depend=afterok:$JOB_BALANCED_RF \
    -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL=BalancedRF,TAG=imbal_v2_balanced_rf,TRAIN_JOBID=$ID_BALANCED_RF \
    pbs_evaluate.sh)
echo "  Eval BalancedRF: $JOB_EVAL_BALANCED_RF (after $ID_BALANCED_RF)"

JOB_EVAL_EASY_ENSEMBLE=$(qsub -W depend=afterok:$JOB_EASY_ENSEMBLE \
    -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL=EasyEnsemble,TAG=imbal_v2_easy_ensemble,TRAIN_JOBID=$ID_EASY_ENSEMBLE \
    pbs_evaluate.sh)
echo "  Eval EasyEnsemble: $JOB_EVAL_EASY_ENSEMBLE (after $ID_EASY_ENSEMBLE)"

echo ""
echo "============================================================"
echo "All jobs submitted successfully!"
echo "============================================================"
echo ""
echo "【ジョブID一覧】"
echo "Training:"
echo "  Baseline:      $ID_BASELINE"
echo "  SMOTE:         $ID_SMOTE"
echo "  SMOTE+Tomek:   $ID_SMOTE_TOMEK"
echo "  SMOTE+ENN:     $ID_SMOTE_ENN"
echo "  SMOTE+RUS:     $ID_SMOTE_RUS"
echo "  BalancedRF:    $ID_BALANCED_RF"
echo "  EasyEnsemble:  $ID_EASY_ENSEMBLE"
echo ""
echo "【予想完了時間】"
echo "  並列実行: 約15-16時間（最長ジョブ基準）"
echo ""
echo "【確認コマンド】"
echo "  qstat -u \$USER"
echo ""
echo "============================================================"

# Save job IDs for later reference
cat > job_ids_v2.txt << EOF
# Imbalance Comparison V2 Job IDs
# Submitted: $(date)

[Training Jobs]
BASELINE=$ID_BASELINE
SMOTE=$ID_SMOTE
SMOTE_TOMEK=$ID_SMOTE_TOMEK
SMOTE_ENN=$ID_SMOTE_ENN
SMOTE_RUS=$ID_SMOTE_RUS
BALANCED_RF=$ID_BALANCED_RF
EASY_ENSEMBLE=$ID_EASY_ENSEMBLE

[Evaluation Jobs]
EVAL_BASELINE=$(echo $JOB_EVAL_BASELINE | cut -d'.' -f1)
EVAL_SMOTE=$(echo $JOB_EVAL_SMOTE | cut -d'.' -f1)
EVAL_SMOTE_TOMEK=$(echo $JOB_EVAL_SMOTE_TOMEK | cut -d'.' -f1)
EVAL_SMOTE_ENN=$(echo $JOB_EVAL_SMOTE_ENN | cut -d'.' -f1)
EVAL_SMOTE_RUS=$(echo $JOB_EVAL_SMOTE_RUS | cut -d'.' -f1)
EVAL_BALANCED_RF=$(echo $JOB_EVAL_BALANCED_RF | cut -d'.' -f1)
EVAL_EASY_ENSEMBLE=$(echo $JOB_EVAL_EASY_ENSEMBLE | cut -d'.' -f1)
EOF

echo "Job IDs saved to: scripts/hpc/logs/imbalance/job_ids_v2.txt"

#!/bin/bash
# ============================================================
# Launch Threshold Optimization + Ensemble Evaluation Jobs
# ============================================================
set -euo pipefail

cd "$(dirname "$0")"

echo "============================================================"
echo "[IMBALANCE V2] Launching Improvement Evaluations"
echo "============================================================"
echo ""

# Training job IDs from V2
BASELINE_JOBID="14468417"
SMOTE_TOMEK_JOBID="14468418"
SMOTE_ENN_JOBID="14468419"
BALANCED_RF_JOBID="14468420"
SMOTE_RUS_JOBID="14468421"
EASY_ENSEMBLE_JOBID="14468501"

echo "=== Part A: Threshold Optimization Evaluations ==="
echo ""

# Optimal thresholds from simulation results
# SMOTE+ENN: 0.15 -> Recall 100%
# SMOTE+RUS: 0.15 -> Recall 100%
# BalancedRF: already high recall at 0.5
# Baseline RF: 0.10 -> Recall 100%

echo "1. SMOTE+ENN with threshold=0.15"
THR_ENN=$(qsub -v MODEL=RF,TAG=imbal_v2_smote_enn,TRAIN_JOBID=$SMOTE_ENN_JOBID,THRESHOLD=0.15 pbs_eval_threshold.sh)
echo "   Job: $THR_ENN"

echo "2. SMOTE+RUS with threshold=0.15"
THR_RUS=$(qsub -v MODEL=RF,TAG=imbal_v2_smote_rus,TRAIN_JOBID=$SMOTE_RUS_JOBID,THRESHOLD=0.15 pbs_eval_threshold.sh)
echo "   Job: $THR_RUS"

echo "3. Baseline RF with threshold=0.10"
THR_BASE=$(qsub -v MODEL=RF,TAG=imbal_v2_baseline,TRAIN_JOBID=$BASELINE_JOBID,THRESHOLD=0.10 pbs_eval_threshold.sh)
echo "   Job: $THR_BASE"

echo "4. SMOTE+Tomek with threshold=0.10"
THR_TOMEK=$(qsub -v MODEL=RF,TAG=imbal_v2_smote_tomek,TRAIN_JOBID=$SMOTE_TOMEK_JOBID,THRESHOLD=0.10 pbs_eval_threshold.sh)
echo "   Job: $THR_TOMEK"

echo ""
echo "=== Part B: Ensemble Evaluation ==="
echo ""

ENSEMBLE_JOB=$(qsub pbs_eval_ensemble.sh)
echo "Ensemble Job: $ENSEMBLE_JOB"

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "============================================================"
echo ""
echo "Threshold Optimization Jobs:"
echo "  THR_SMOTE_ENN:   $THR_ENN"
echo "  THR_SMOTE_RUS:   $THR_RUS"
echo "  THR_BASELINE:    $THR_BASE"
echo "  THR_SMOTE_TOMEK: $THR_TOMEK"
echo ""
echo "Ensemble Job:"
echo "  ENSEMBLE:        $ENSEMBLE_JOB"
echo ""
echo "Monitor: qstat -u \$USER"
echo "============================================================"

# Save job IDs
cat >> job_ids_v2.txt << EOF

[Improvement Evaluation Jobs - $(date)]
THR_SMOTE_ENN=$THR_ENN
THR_SMOTE_RUS=$THR_RUS
THR_BASELINE=$THR_BASE
THR_SMOTE_TOMEK=$THR_TOMEK
ENSEMBLE=$ENSEMBLE_JOB
EOF

echo "Job IDs saved to job_ids_v2.txt"

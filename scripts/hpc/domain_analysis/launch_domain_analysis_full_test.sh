#!/bin/bash
# launch_domain_analysis_full_test.sh
# Purpose: Run complete domain analysis pipeline in TEST mode
# Includes: distance computation → ranking → train → eval → summary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "   DOMAIN ANALYSIS - FULL TEST PIPELINE   "
echo "=========================================="
echo "[TEST] Mode: N_TRIALS=5, KSS_SIMPLIFIED=1, split=0.4:0.3:0.3"
echo "[TEST] Array jobs: 2 ranks (instead of 9)"
echo ""

# --- 1. Distance computation (3 metrics: MMD, Wasserstein, DTW) ---
echo "[STEP 1/5] Submitting distance computation jobs..."
jid_distance=$(qsub "${SCRIPT_DIR}/pbs_compute_distance_test.sh")
echo "[INFO] Distance computation job: $jid_distance (3 array tasks)"

# --- 2. Ranking (depends on distance completion) ---
echo "[STEP 2/5] Submitting ranking job..."
jid_ranking=$(qsub -W depend=afterok:${jid_distance%%.*} "${SCRIPT_DIR}/pbs_ranking_test.sh")
echo "[INFO] Ranking job: $jid_ranking (depends on distance)"

# --- 3. Train jobs (depends on ranking completion, 2 array jobs × 3 modes) ---
echo "[STEP 3/5] Submitting training jobs..."
jid_train=$(qsub -W depend=afterok:${jid_ranking%%.*} "${SCRIPT_DIR}/pbs_train_rank_test.sh")
echo "[INFO] Training job: $jid_train (2 array × 3 modes = 6 tasks)"

# --- 4. Eval jobs (depends on training completion) ---
echo "[STEP 4/5] Submitting evaluation jobs..."
jid_eval=$(qsub -W depend=afterok:${jid_train%%.*} "${SCRIPT_DIR}/pbs_eval_rank_test.sh")
echo "[INFO] Evaluation job: $jid_eval (depends on train)"

# --- 5. Summary table generation (depends on eval completion) ---
echo "[STEP 5/5] Submitting analysis job..."
jid_summary=$(qsub -W depend=afterok:${jid_eval%%.*} "${SCRIPT_DIR}/pbs_analysis_test.sh")
echo "[INFO] Analysis job: $jid_summary (depends on eval)"

echo ""
echo "=== FULL TEST PIPELINE SUBMITTED ==="
echo "1. Distance:  $jid_distance (3 metrics)"
echo "2. Ranking:   $jid_ranking"
echo "3. Train:     $jid_train (2 ranks × 3 modes)"
echo "4. Eval:      $jid_eval"
echo "5. Summary:   $jid_summary"
echo "======================================"
echo ""
echo "Monitor with: qstat -u $USER"
echo "Check logs in: scripts/hpc/log/"

#!/bin/bash
# launch_light_test.sh
# ==============================================================================
# 軽量テスト版：ロジック検証用
# ==============================================================================
# テスト範囲:
#   - Array Job 1-2のみ (mmd_high, mmd_middle)
#   - source_onlyモードのみ
#   - N_TRIALS=3
#   - 3手法すべてをテスト (mean_distance, centroid_umap, lof)
#
# 目的: パイプライン全体のロジック検証を高速に行う
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RANKING_METHODS=("mean_distance" "centroid_umap" "lof")

echo "============================================================"
echo "=== Light Test: Domain Analysis Pipeline ==="
echo "============================================================"
echo "Testing: 2 groups × 3 methods × 1 mode = 6 tasks"
echo "Methods: ${RANKING_METHODS[*]}"
echo "============================================================"
echo ""

# --- 1. Train jobs (3 methods, array 1-2 each) ---
declare -a train_jobs=()

for method in "${RANKING_METHODS[@]}"; do
    jid_train=$(qsub -v RANKING_METHOD="$method" "${SCRIPT_DIR}/pbs_train_rank_light.sh")
    train_jobs+=("$jid_train")
    echo "[INFO] Submitted train (${method}): $jid_train"
done

# Build dependency string
TRAIN_DEPEND=""
for jid in "${train_jobs[@]}"; do
    if [[ -z "$TRAIN_DEPEND" ]]; then
        TRAIN_DEPEND="${jid%%.*}"
    else
        TRAIN_DEPEND="${TRAIN_DEPEND}:${jid%%.*}"
    fi
done

# --- 2. Eval jobs (after training) ---
declare -a eval_jobs=()

for method in "${RANKING_METHODS[@]}"; do
    jid_eval=$(qsub -v RANKING_METHOD="$method" -W depend=afterok:${TRAIN_DEPEND} "${SCRIPT_DIR}/pbs_eval_rank_light.sh")
    eval_jobs+=("$jid_eval")
    echo "[INFO] Submitted eval (${method}): $jid_eval"
done

echo ""
echo "============================================================"
echo "=== Light Test Submitted ==="
echo "============================================================"
echo ""
echo "Training:"
for i in "${!RANKING_METHODS[@]}"; do
    echo "  ${RANKING_METHODS[$i]}: ${train_jobs[$i]}"
done
echo ""
echo "Evaluation:"
for i in "${!RANKING_METHODS[@]}"; do
    echo "  ${RANKING_METHODS[$i]}: ${eval_jobs[$i]}"
done
echo ""
echo "Expected completion: ~10-20 minutes"
echo "============================================================"
echo ""
echo "Monitor with: qstat -u \$USER"
echo "Check logs:   ls -la scripts/hpc/log/RF_*_light*"
echo "============================================================"

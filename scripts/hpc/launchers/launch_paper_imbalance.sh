#!/bin/bash
# ============================================================
# 論文用クラス不均衡実験ランチャー
# ============================================================
# 実験条件:
#   - シード: 42, 123
#   - ターゲット比率: 0.1, 0.5
#   - 分類モデル: RF (BalancedRFは手法として含む)
#   - 不均衡対策手法: Baseline, Plain SMOTE, Subject-wise SMOTE, RUS, Balanced RF
#   - Optuna trial数: 100
#   - Optuna目的関数: F2 (既に実装済み)
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/imbalance/pbs_imbalance_comparison.sh"

# 論文用設定
SEEDS="42 123"
RATIOS="0.1 0.5"
N_TRIALS=100

# 実験条件（論文用）
EXPERIMENTS=(
    "baseline"           # ベースライン（オーバーサンプリングなし）
    "smote"              # Plain SMOTE
    "smote_subjectwise"  # Subject-wise SMOTE
    "undersample_rus"    # Random Undersampling
    "balanced_rf"        # BalancedRandomForest
)

# Parse arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Resource configurations (キュー状況に基づいて最適化)
# TINY: 最大30分制限のため使用しない
get_resources() {
    local method="$1"
    case "$method" in
        balanced_rf)
            # BalancedRF: 8コア必要、LONGキューを使用（長時間実行の可能性）
            echo "ncpus=8:mem=8gb 08:00:00 LONG"
            ;;
        smote|smote_subjectwise)
            # SMOTE系: 4コア、SINGLEキューを使用
            echo "ncpus=4:mem=8gb 08:00:00 SINGLE"
            ;;
        baseline|undersample_rus)
            # 軽量な実験: 4コア、SINGLEキューを使用（LONGは容量小さい）
            echo "ncpus=4:mem=8gb 04:00:00 SINGLE"
            ;;
        *)
            # デフォルト: SINGLEキュー
            echo "ncpus=4:mem=8gb 06:00:00 SINGLE"
            ;;
    esac
}

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/imbalance"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_paper_${TIMESTAMP}.txt"

echo "============================================================"
echo "論文用クラス不均衡実験ランチャー"
echo "============================================================"
echo "実験手法: ${EXPERIMENTS[*]}"
echo "シード: $SEEDS"
echo "比率: $RATIOS"
echo "Optuna trials: $N_TRIALS"
echo "Dry run: $DRY_RUN"
echo "Log: $LOG_FILE"
echo ""
echo "キュー状態:"
qstat -Q | grep -E "Queue|TINY|SINGLE|DEFAULT|SMALL"
echo "============================================================"
echo ""

echo "# Launched at $(date)" > "$LOG_FILE"

JOB_COUNT=0
TOTAL_JOBS=0

# 総ジョブ数を計算
for METHOD in "${EXPERIMENTS[@]}"; do
    for SEED in $SEEDS; do
        if [[ "$METHOD" == "baseline" || "$METHOD" == "balanced_rf" ]]; then
            # Baseline と BalancedRF は比率不要
            TOTAL_JOBS=$((TOTAL_JOBS + 1))
        else
            # SMOTE系とRUSは比率ごとに実行
            for RATIO in $RATIOS; do
                TOTAL_JOBS=$((TOTAL_JOBS + 1))
            done
        fi
    done
done

echo "合計 $TOTAL_JOBS ジョブを投入します"
echo ""

# ジョブ投入
for METHOD in "${EXPERIMENTS[@]}"; do
    for SEED in $SEEDS; do
        if [[ "$METHOD" == "baseline" || "$METHOD" == "balanced_rf" ]]; then
            # Baseline と BalancedRF は比率不要
            RESOURCES=$(get_resources "$METHOD")
            NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
            WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
            QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
            
            JOB_NAME="${METHOD:0:6}_s${SEED}"
            
            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v METHOD=$METHOD,SEED=$SEED,N_TRIALS=$N_TRIALS $JOB_SCRIPT"
            
            if $DRY_RUN; then
                echo "[DRY-RUN] $CMD"
            else
                echo "[$((JOB_COUNT + 1))/$TOTAL_JOBS] Submitting: $METHOD (seed=$SEED)"
                JOBID=$(eval $CMD)
                echo "$JOBID | $METHOD | seed=$SEED | $QUEUE" >> "$LOG_FILE"
                echo "  -> JobID: $JOBID (Queue: $QUEUE)"
            fi
            JOB_COUNT=$((JOB_COUNT + 1))
        else
            # SMOTE系とRUSは比率ごとに実行
            for RATIO in $RATIOS; do
                RESOURCES=$(get_resources "$METHOD")
                NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                
                JOB_NAME="${METHOD:0:6}_r${RATIO}_s${SEED}"
                
                CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v METHOD=$METHOD,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS $JOB_SCRIPT"
                
                if $DRY_RUN; then
                    echo "[DRY-RUN] $CMD"
                else
                    echo "[$((JOB_COUNT + 1))/$TOTAL_JOBS] Submitting: $METHOD (ratio=$RATIO, seed=$SEED)"
                    JOBID=$(eval $CMD)
                    echo "$JOBID | $METHOD | ratio=$RATIO | seed=$SEED | $QUEUE" >> "$LOG_FILE"
                    echo "  -> JobID: $JOBID (Queue: $QUEUE)"
                fi
                JOB_COUNT=$((JOB_COUNT + 1))
            done
        fi
    done
done

echo ""
echo "============================================================"
echo "合計 $JOB_COUNT ジョブを投入しました"
if ! $DRY_RUN; then
    echo "ログ: $LOG_FILE"
    echo ""
    echo "ジョブ状態確認:"
    echo "  qstat -u s2240011"
    echo ""
    echo "特定ジョブのログ確認:"
    echo "  tail -f $LOG_DIR/\${PBS_JOBID}.o*"
fi
echo "============================================================"

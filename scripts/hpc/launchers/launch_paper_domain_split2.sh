#!/bin/bash
# ============================================================
# 論文用ドメインシフト実験ランチャー（2グループ分割版）
# ============================================================
# 実験条件:
#   - シード: 42, 123
#   - ターゲット比率: 0.1, 0.5
#   - 分類モデル: RF (BalancedRFは手法として含む)
#   - 不均衡対策手法: baseline, plain SMOTE, subject-wise SMOTE, RUS, Balanced RF
#   - Optuna trial数: 100
#   - Optuna目的関数: F2 (既に実装済み)
#   - ランキング手法: KNN
#   - 距離指標: mmd, dtw, wasserstein
#   - ドメイングループ: out_domain (HIGH), in_domain (LOW) ※2グループ分割
#   - 訓練モード: 
#       source_only (cross domain): ターゲットの逆ドメインで訓練
#       target_only (single domain): ターゲットドメイン内で訓練
#
# 新しいロジック:
#   - source_only + out_domain → in_domainで訓練、out_domainで評価
#   - source_only + in_domain → out_domainで訓練、in_domainで評価
#   - target_only + out_domain → out_domainで訓練・評価
#   - target_only + in_domain → in_domainで訓練・評価
#
# Total: 3 distances × 2 domains × 2 modes × 2 seeds × 8 conditions = 96 jobs
# (mid_domainを除外したため、192→96ジョブに削減)
# ============================================================

set -uo pipefail  # Remove -e to continue on errors

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

# 論文用設定
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"

# 距離指標とドメイングループ（2グループ分割）
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")  # mid_domainを除外

# 訓練モード
# source_only = cross domain (ターゲットの逆ドメインで訓練)
# target_only = single domain (ターゲットドメイン内で訓練)
MODES=("source_only" "target_only")

# 不均衡対策手法（論文用）
# Format: "CONDITION:description"
CONDITIONS=(
    "baseline:Baseline (no handling)"
    "smote_plain:Plain SMOTE"
    "smote:Subject-wise SMOTE"
    "undersample:Random Undersampling"
    "balanced_rf:Balanced RF"
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

# Resource configurations (メモリ最適化版)
get_resources() {
    local condition="$1"
    case "$condition" in
        balanced_rf)
            # BalancedRF: 8コア必要、メモリ多め
            echo "ncpus=8:mem=12gb 08:00:00 LONG"
            ;;
        smote|smote_plain)
            # SMOTE系: 4コア、中程度のメモリ
            echo "ncpus=4:mem=10gb 08:00:00 SINGLE"
            ;;
        baseline|undersample)
            # 軽量な実験: 4コア、少なめのメモリ
            echo "ncpus=4:mem=8gb 06:00:00 SINGLE"
            ;;
        *)
            # デフォルト
            echo "ncpus=4:mem=8gb 08:00:00 SINGLE"
            ;;
    esac
}

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_paper_domain_split2_${TIMESTAMP}.log"

echo "============================================================"
echo "論文用ドメインシフト実験ランチャー (2グループ分割版)"
echo "============================================================"
echo "分割方式: split2 (in_domain=44名, out_domain=43名)"
echo "距離指標: ${DISTANCES[*]}"
echo "ドメイングループ: ${DOMAINS[*]} (※mid_domainなし)"
echo "訓練モード: ${MODES[*]}"
echo "  - source_only (cross domain): ターゲットの逆ドメインで訓練"
echo "  - target_only (single domain): ターゲットドメイン内で訓練"
echo "不均衡対策手法: 5種類 (baseline, plain SMOTE, subject-wise SMOTE, RUS, Balanced RF)"
echo "シード: ${SEEDS[*]}"
echo "ターゲット比率: ${RATIOS[*]}"
echo "Optuna trials: $N_TRIALS"
echo "Dry run: $DRY_RUN"
echo ""
echo "予想ジョブ数: $((${#DISTANCES[@]} * ${#DOMAINS[@]} * ${#MODES[@]} * ${#SEEDS[@]} * 8)) jobs"
echo "  - 各条件8ジョブ = baseline(1) + smote_plain(2) + smote(2) + undersample(2) + balanced_rf(1)"
echo "  - 3距離 × 2ドメイン × 2モード × 2シード × 8 = 96 jobs"
echo "============================================================"
echo ""

# Verify job script exists
if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[ERROR] Job script not found: $JOB_SCRIPT"
    echo "Creating job script..."
    exit 1
fi

# Start logging
{
    echo "# Launch started at $(date)"
    echo "# Command: $0 $*"
    echo "# User: $(whoami)"
    echo "# Host: $(hostname)"
    echo ""
} > "$LOG_FILE"

JOB_COUNT=0
SKIP_COUNT=0

# Main loop
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # Baseline (no ratio)
                CONDITION="baseline"
                RESOURCES=$(get_resources "$CONDITION")
                NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                
                JOB_NAME="bs_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_s${SEED}"
                
                CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                CMD="$CMD -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                CMD="$CMD $JOB_SCRIPT"
                
                if $DRY_RUN; then
                    echo "[DRY-RUN] baseline | $DISTANCE | $DOMAIN | $MODE | seed=$SEED"
                else
                    JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed to submit: $CMD"; ((SKIP_COUNT++)); continue; }
                    echo "[SUBMIT] baseline | $DISTANCE | $DOMAIN | $MODE | s$SEED → $JOB_ID"
                    echo "baseline:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                    ((JOB_COUNT++))
                    sleep 0.2
                fi
                
                # SMOTE variants and undersample (with ratios)
                for RATIO in "${RATIOS[@]}"; do
                    for COND_SPEC in "smote_plain:Plain SMOTE" "smote:Subject-wise SMOTE" "undersample:RUS"; do
                        CONDITION="${COND_SPEC%%:*}"
                        
                        RESOURCES=$(get_resources "$CONDITION")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                        
                        COND_SHORT="${CONDITION:0:2}"
                        JOB_NAME="${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_r${RATIO}_s${SEED}"
                        
                        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                        CMD="$CMD -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                        CMD="$CMD $JOB_SCRIPT"
                        
                        if $DRY_RUN; then
                            echo "[DRY-RUN] $CONDITION | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s=$SEED"
                        else
                            JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed to submit: $CMD"; ((SKIP_COUNT++)); continue; }
                            echo "[SUBMIT] $CONDITION | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s$SEED → $JOB_ID"
                            echo "$CONDITION:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
                            ((JOB_COUNT++))
                            sleep 0.2
                        fi
                    done
                done
                
                # Balanced RF (no ratio)
                CONDITION="balanced_rf"
                RESOURCES=$(get_resources "$CONDITION")
                NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                
                JOB_NAME="bf_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_s${SEED}"
                
                CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                CMD="$CMD -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                CMD="$CMD $JOB_SCRIPT"
                
                if $DRY_RUN; then
                    echo "[DRY-RUN] balanced_rf | $DISTANCE | $DOMAIN | $MODE | seed=$SEED"
                else
                    JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed to submit: $CMD"; ((SKIP_COUNT++)); continue; }
                    echo "[SUBMIT] balanced_rf | $DISTANCE | $DOMAIN | $MODE | s$SEED → $JOB_ID"
                    echo "balanced_rf:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                    ((JOB_COUNT++))
                    sleep 0.2
                fi
            done
        done
    done
done

# Summary
{
    echo ""
    echo "# Launch completed at $(date)"
    echo "# Total jobs submitted: $JOB_COUNT"
    echo "# Skipped: $SKIP_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
    echo "Expected jobs: $JOB_COUNT"
else
    echo "Successfully submitted: $JOB_COUNT jobs"
    echo "Skipped: $SKIP_COUNT jobs"
    echo "Log file: $LOG_FILE"
fi
echo "============================================================"

#!/bin/bash
# ============================================================
# 論文用ドメインシフト実験ランチャー
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
#   - ドメイングループ: out_domain, in_domain
#   - 訓練モード: source_only (cross domain), target_only (single domain)
#
# Total: 3 distances × 2 domains × 2 modes × 2 seeds × 8 conditions = 192 jobs
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison.sh"

# 論文用設定
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"

# 距離指標とドメイングループ
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")

# 訓練モード（論文での表記に対応）
# source_only = cross domain (ドメイン外の被験者で訓練)
# target_only = single domain (ターゲット被験者のみで訓練)
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

# Resource configurations (キュー状況に基づいて最適化)
get_resources() {
    local condition="$1"
    case "$condition" in
        balanced_rf)
            # BalancedRF: 8コア必要、LONGキュー使用
            echo "ncpus=8:mem=8gb 08:00:00 LONG"
            ;;
        smote|smote_plain)
            # SMOTE系: 4コア、SINGLEキュー使用
            echo "ncpus=4:mem=8gb 08:00:00 SINGLE"
            ;;
        baseline|undersample)
            # 軽量な実験: 4コア、SINGLEキュー使用
            echo "ncpus=4:mem=8gb 06:00:00 SINGLE"
            ;;
        *)
            echo "ncpus=4:mem=8gb 06:00:00 SINGLE"
            ;;
    esac
}

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_paper_domain_${TIMESTAMP}.txt"

echo "============================================================"
echo "論文用ドメインシフト実験ランチャー"
echo "============================================================"
echo "距離指標: ${DISTANCES[*]}"
echo "ドメイングループ: ${DOMAINS[*]}"
echo "訓練モード: ${MODES[*]} (source_only=cross domain, target_only=single domain)"
echo "シード: ${SEEDS[*]}"
echo "比率: ${RATIOS[*]}"
echo "Optuna trials: $N_TRIALS"
echo "ランキング手法: $RANKING"
echo "不均衡対策手法: ${#CONDITIONS[@]}"
for cond in "${CONDITIONS[@]}"; do
    echo "  - ${cond##*:}"
done
echo "Dry run: $DRY_RUN"
echo "Log: $LOG_FILE"
echo ""
echo "キュー状態:"
qstat -Q | grep -E "Queue|SINGLE|LONG|DEFAULT"
echo "============================================================"
echo ""

# 総ジョブ数を計算
TOTAL_JOBS=0
for dist in "${DISTANCES[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        for mode in "${MODES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for cond_entry in "${CONDITIONS[@]}"; do
                    CONDITION="${cond_entry%%:*}"
                    if [[ "$CONDITION" == "baseline" || "$CONDITION" == "balanced_rf" ]]; then
                        # 比率不要
                        TOTAL_JOBS=$((TOTAL_JOBS + 1))
                    else
                        # 比率ごとに実行
                        for ratio in "${RATIOS[@]}"; do
                            TOTAL_JOBS=$((TOTAL_JOBS + 1))
                        done
                    fi
                done
            done
        done
    done
done

echo "合計 $TOTAL_JOBS ジョブを投入します"
echo ""

echo "# Launched at $(date)" > "$LOG_FILE"

JOB_COUNT=0

# ジョブ投入
for dist in "${DISTANCES[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        for mode in "${MODES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for cond_entry in "${CONDITIONS[@]}"; do
                    CONDITION="${cond_entry%%:*}"
                    
                    if [[ "$CONDITION" == "baseline" || "$CONDITION" == "balanced_rf" ]]; then
                        # Baseline と Balanced RF は比率不要
                        RATIO=0
                        
                        RESOURCES=$(get_resources "$CONDITION")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                        
                        # ジョブ名生成（15文字制限）
                        JOB_NAME="d_${CONDITION:0:4}_${dist:0:3}_${domain:0:3}_${mode:0:3}_s${seed}"
                        JOB_NAME="${JOB_NAME:0:15}"
                        
                        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v CONDITION=$CONDITION,MODE=$mode,DISTANCE=$dist,DOMAIN=$domain,RATIO=$RATIO,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true $JOB_SCRIPT"
                        
                        if $DRY_RUN; then
                            echo "[DRY-RUN] $CMD"
                        else
                            echo "[$((JOB_COUNT + 1))/$TOTAL_JOBS] Submitting: $CONDITION $dist $domain $mode s$seed"
                            JOBID=$(eval $CMD)
                            echo "$JOBID | $CONDITION | $dist | $domain | $mode | seed=$seed | $QUEUE" >> "$LOG_FILE"
                            echo "  -> JobID: $JOBID (Queue: $QUEUE)"
                        fi
                        JOB_COUNT=$((JOB_COUNT + 1))
                    else
                        # SMOTE系とRUSは比率ごとに実行
                        for ratio in "${RATIOS[@]}"; do
                            RESOURCES=$(get_resources "$CONDITION")
                            NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                            WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                            QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                            
                            # ジョブ名生成（15文字制限）
                            JOB_NAME="d_${CONDITION:0:4}_r${ratio}_${dist:0:3}_${domain:0:3}_${mode:0:3}_s${seed}"
                            JOB_NAME="${JOB_NAME:0:15}"
                            
                            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v CONDITION=$CONDITION,MODE=$mode,DISTANCE=$dist,DOMAIN=$domain,RATIO=$ratio,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true $JOB_SCRIPT"
                            
                            if $DRY_RUN; then
                                echo "[DRY-RUN] $CMD"
                            else
                                echo "[$((JOB_COUNT + 1))/$TOTAL_JOBS] Submitting: $CONDITION r$ratio $dist $domain $mode s$seed"
                                JOBID=$(eval $CMD)
                                echo "$JOBID | $CONDITION | ratio=$ratio | $dist | $domain | $mode | seed=$seed | $QUEUE" >> "$LOG_FILE"
                                echo "  -> JobID: $JOBID (Queue: $QUEUE)"
                            fi
                            JOB_COUNT=$((JOB_COUNT + 1))
                        done
                    fi
                done
            done
        done
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

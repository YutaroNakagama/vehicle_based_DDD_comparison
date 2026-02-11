#!/bin/bash
# ============================================================
# 論文用先行研究実験ランチャー（ドメイン分割版）
# ============================================================
# 実験条件:
#   - モデル: SvmA, SvmW, Lstm
#   - シード: 42, 123
#   - ターゲット比率: 0.1, 0.5
#   - 不均衡対策手法: baseline, plain SMOTE, subject-wise SMOTE, RUS, balanced RF (※SvmA, Lstmは該当しない場合あり)
#   - Optuna試行回数: 100 (SvmWのみ)
#   - Optuna目的関数: 各先行研究に準ずる
#   - ランキング手法: knn
#   - 距離指標: mmd, dtw, wasserstein
#   - ドメイングループ: out_domain, in_domain (2分割)
#   - 訓練モード: source_only (cross domain), target_only (single domain)
#
# 注意:
#   - SvmAはPSO最適化のため、不均衡対策手法の組み合わせは限定的
#   - LstmはDeep Learningのため、Balanced RFは適用不可
#   - SvmWのみOptunaを使用するため、N_TRIALS=100を設定
#
# Total: 3 models × 3 distances × 2 domains × 2 modes × 2 seeds × 条件数 = 多数
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

# 論文用設定
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"

# 距離指標とドメイングループ（2分割）
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")

# 訓練モード
MODES=("source_only" "target_only")

# モデル
MODELS=("SvmW" "SvmA" "Lstm")

# キュー設定（複数キューに分散投入）
USE_MULTI_QUEUE=true

# 不均衡対策手法（モデルごとに適用可能な手法が異なる）
# SvmW: すべて適用可能
# SvmA: baseline, smote, smote_plain, undersample (Balanced RFは不可)
# Lstm: baseline, smote, smote_plain, undersample (Balanced RFは不可)

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

# Queue counter for load balancing
QUEUE_COUNTER=0

# Resource configurations with multi-queue support
get_resources() {
    local model="$1"
    local condition="$2"
    local queue
    
    # キュー選択（ラウンドロビン方式で分散）
    if $USE_MULTI_QUEUE; then
        local queues=("SINGLE" "LONG" "DEFAULT")
        queue="${queues[$((QUEUE_COUNTER % 3))]}"
        ((QUEUE_COUNTER++))
    else
        queue="SINGLE"
    fi
    
    case "$model" in
        SvmA)
            # PSO最適化で時間がかかる
            echo "ncpus=8:mem=32gb 24:00:00 $queue"
            ;;
        SvmW)
            # Optuna最適化
            echo "ncpus=8:mem=16gb 12:00:00 $queue"
            ;;
        Lstm)
            # Deep Learning (CPU)
            echo "ncpus=8:mem=32gb 16:00:00 $queue"
            ;;
    esac
}

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_prior_research_split2_${TIMESTAMP}.log"

echo "============================================================"
echo "論文用先行研究実験ランチャー (2グループ分割版)"
echo "============================================================"
echo "モデル: ${MODELS[*]}"
echo "分割方式: split2 (in_domain=44名, out_domain=43名)"
echo "距離指標: ${DISTANCES[*]}"
echo "ドメイングループ: ${DOMAINS[*]}"
echo "訓練モード: ${MODES[*]}"
echo "  - source_only (cross domain): ターゲットの逆ドメインで訓練"
echo "  - target_only (single domain): ターゲットドメイン内で訓練"
echo "不均衡対策手法: モデルごとに異なる"
echo "シード: ${SEEDS[*]}"
echo "ターゲット比率: ${RATIOS[*]}"
echo "Optuna trials (SvmWのみ): $N_TRIALS"
echo "複数キュー使用: $USE_MULTI_QUEUE (SINGLE, LONG, DEFAULT に分散)"
echo "Dry run: $DRY_RUN"
echo "============================================================"
echo ""

# Verify job script exists
if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[WARNING] Job script not found: $JOB_SCRIPT"
    echo "[INFO] You need to create this script first."
    exit 1
fi

# Start logging
{
    echo "# Launch started at $(date)"
    echo "# Command: $0 $*"
    echo ""
} > "$LOG_FILE"

JOB_COUNT=0
SKIP_COUNT=0

# Helper function to determine applicable conditions for each model
get_conditions() {
    local model="$1"
    case "$model" in
        SvmW)
            # balanced_rfは別モデル(BalancedRF)であり SvmWには不要
            echo "baseline smote_plain smote undersample"
            ;;
        SvmA|Lstm)
            echo "baseline smote_plain smote undersample"
            ;;
    esac
}

# Main loop
for MODEL in "${MODELS[@]}"; do
    CONDITIONS=$(get_conditions "$MODEL")
    
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for MODE in "${MODES[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    # Baseline (no ratio)
                    if echo "$CONDITIONS" | grep -q "baseline"; then
                        CONDITION="baseline"
                        RESOURCES=$(get_resources "$MODEL" "$CONDITION")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                        
                        JOB_NAME="${MODEL:0:2}_bs_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_s${SEED}"
                        
                        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                        CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                        CMD="$CMD $JOB_SCRIPT"
                        
                        if $DRY_RUN; then
                            echo "[DRY-RUN] $MODEL | baseline | $DISTANCE | $DOMAIN | $MODE | s=$SEED"
                        else
                            JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed: $CMD"; ((SKIP_COUNT++)); continue; }
                            echo "[SUBMIT] $MODEL | baseline | $DISTANCE | $DOMAIN | $MODE | s$SEED → $JOB_ID"
                            echo "$MODEL:baseline:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                            ((JOB_COUNT++))
                            sleep 0.2
                        fi
                    fi
                    
                    # Ratio-based methods
                    for RATIO in "${RATIOS[@]}"; do
                        for COND in "smote_plain" "smote" "undersample"; do
                            # Skip if not applicable for this model
                            if ! echo "$CONDITIONS" | grep -q "$COND"; then
                                continue
                            fi
                            
                            RESOURCES=$(get_resources "$MODEL" "$COND")
                            NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                            WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                            QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                            
                            COND_SHORT="${COND:0:2}"
                            JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_r${RATIO}_s${SEED}"
                            
                            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                            CMD="$CMD -v MODEL=$MODEL,CONDITION=$COND,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                            CMD="$CMD $JOB_SCRIPT"
                            
                            if $DRY_RUN; then
                                echo "[DRY-RUN] $MODEL | $COND | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s=$SEED"
                            else
                                JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed: $CMD"; ((SKIP_COUNT++)); continue; }
                                echo "[SUBMIT] $MODEL | $COND | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s$SEED → $JOB_ID"
                                echo "$MODEL:$COND:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
                                ((JOB_COUNT++))
                                sleep 0.2
                            fi
                        done
                    done
                done
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
echo ""
echo "予想ジョブ数の計算:"
echo "  SvmW: 3距離 × 2ドメイン × 2モード × 2シード × (1 baseline + 2×4 ratio-based) = 216 jobs"
echo "  SvmA: 3距離 × 2ドメイン × 2モード × 2シード × (1 baseline + 2×3 ratio-based) = 168 jobs"
echo "  Lstm: 3距離 × 2ドメイン × 2モード × 2シード × (1 baseline + 2×3 ratio-based) = 168 jobs"
echo "  合計: 552 jobs"
echo "============================================================"

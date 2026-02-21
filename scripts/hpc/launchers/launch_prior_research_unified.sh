#!/bin/bash
# ============================================================
# 論文用先行研究実験ランチャー（統一版 - domain_train）
# ============================================================
# 変更点（split2版との違い）:
#   - source_only/target_only の重複トレーニングを解消
#   - 1ジョブ = 1回の学習 + 2回の評価（within + cross）
#   - 分割比率: train(70%) / val(15%) / test(15%)
#   - ジョブ数が半分に削減（MODEループなし）
#
# 実験条件:
#   - モデル: SvmA, SvmW, Lstm
#   - シード: 42, 123
#   - ターゲット比率: 0.1, 0.5
#   - 不均衡対策手法: baseline, plain SMOTE, subject-wise SMOTE, RUS
#   - Optuna試行回数: 100 (SvmWのみ)
#   - ランキング手法: knn
#   - 距離指標: mmd, dtw, wasserstein
#   - ドメイングループ: out_domain, in_domain (2分割)
#
# Total: 3 models × 3 distances × 2 domains × 2 seeds × 条件数
#   SvmW: 3 × 2 × 2 × (1 + 2×3) = 84 jobs  (cf. split2: 168)
#   SvmA: 3 × 2 × 2 × (1 + 2×3) = 84 jobs  (cf. split2: 168)
#   Lstm: 3 × 2 × 2 × (1 + 2×3) = 84 jobs  (cf. split2: 168)
#   合計: 252 jobs (cf. split2: 504 ← 半減)
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"

# 論文用設定
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"

# 距離指標とドメイングループ（2分割）
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")

# モデル
MODELS=("SvmW" "SvmA" "Lstm")

# キュー設定（複数キューに分散投入）
USE_MULTI_QUEUE=true

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
            echo "ncpus=8:mem=32gb 48:00:00 $queue"
            ;;
        SvmW)
            echo "ncpus=8:mem=16gb 12:00:00 $queue"
            ;;
        Lstm)
            echo "ncpus=8:mem=32gb 16:00:00 $queue"
            ;;
    esac
}

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_prior_research_unified_${TIMESTAMP}.log"

echo "============================================================"
echo "統一版先行研究実験ランチャー (domain_train)"
echo "============================================================"
echo "モデル: ${MODELS[*]}"
echo "分割方式: split2 (in_domain=44名, out_domain=43名)"
echo "距離指標: ${DISTANCES[*]}"
echo "ドメイングループ: ${DOMAINS[*]}"
echo "訓練モード: domain_train (1回学習 + within/cross 2回評価)"
echo "分割比率: train(70%) / val(15%) / test(15%)"
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
            echo "baseline smote_plain smote undersample"
            ;;
        SvmA|Lstm)
            echo "baseline smote_plain smote undersample"
            ;;
    esac
}

# Main loop (NOTE: no MODE loop — domain_train handles both within/cross)
for MODEL in "${MODELS[@]}"; do
    CONDITIONS=$(get_conditions "$MODEL")
    
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # Baseline (no ratio)
                if echo "$CONDITIONS" | grep -q "baseline"; then
                    CONDITION="baseline"
                    RESOURCES=$(get_resources "$MODEL" "$CONDITION")
                    NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                    WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                    QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                    
                    JOB_NAME="${MODEL:0:2}_bs_${DISTANCE:0:1}${DOMAIN:0:1}_dt_s${SEED}"
                    
                    CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                    CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                    CMD="$CMD $JOB_SCRIPT"
                    
                    if $DRY_RUN; then
                        echo "[DRY-RUN] $MODEL | baseline | $DISTANCE | $DOMAIN | domain_train | s=$SEED"
                    else
                        JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed: $CMD"; ((SKIP_COUNT++)); continue; }
                        echo "[SUBMIT] $MODEL | baseline | $DISTANCE | $DOMAIN | domain_train | s$SEED → $JOB_ID"
                        echo "$MODEL:baseline:$DISTANCE:$DOMAIN:domain_train:$SEED:$JOB_ID" >> "$LOG_FILE"
                        ((JOB_COUNT++))
                        sleep 0.2
                    fi
                fi
                
                # Ratio-based methods
                for RATIO in "${RATIOS[@]}"; do
                    for COND in "smote_plain" "smote" "undersample"; do
                        if ! echo "$CONDITIONS" | grep -q "$COND"; then
                            continue
                        fi
                        
                        RESOURCES=$(get_resources "$MODEL" "$COND")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                        
                        COND_SHORT="${COND:0:2}"
                        JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_r${RATIO}_s${SEED}"
                        
                        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                        CMD="$CMD -v MODEL=$MODEL,CONDITION=$COND,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                        CMD="$CMD $JOB_SCRIPT"
                        
                        if $DRY_RUN; then
                            echo "[DRY-RUN] $MODEL | $COND | $DISTANCE | $DOMAIN | domain_train | r=$RATIO | s=$SEED"
                        else
                            JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed: $CMD"; ((SKIP_COUNT++)); continue; }
                            echo "[SUBMIT] $MODEL | $COND | $DISTANCE | $DOMAIN | domain_train | r=$RATIO | s$SEED → $JOB_ID"
                            echo "$MODEL:$COND:$DISTANCE:$DOMAIN:domain_train:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
                            ((JOB_COUNT++))
                            sleep 0.2
                        fi
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
echo "  各モデル: 3距離 × 2ドメイン × 2シード × (1 baseline + 2比率 × 3手法) = 84 jobs"
echo "  SvmW: 84 jobs  (split2版: 168 → 50%削減)"
echo "  SvmA: 84 jobs  (split2版: 168 → 50%削減)"
echo "  Lstm: 84 jobs  (split2版: 168 → 50%削減)"
echo "  合計: 252 jobs (split2版: 504 → 50%削減)"
echo ""
echo "各ジョブの構成:"
echo "  学習: domain_train (ドメイン内 70%で学習, 15%でval)"
echo "  評価①: within-domain (同ドメイン test 15%)"
echo "  評価②: cross-domain  (逆ドメイン test 15%)"
echo "============================================================"

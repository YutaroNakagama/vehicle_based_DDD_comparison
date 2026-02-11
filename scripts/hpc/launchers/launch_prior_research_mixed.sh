#!/bin/bash
# ============================================================
# 先行研究実験ランチャー — mixed domain のみ（2グループ分割版）
# ============================================================
# 既存の source_only / target_only に加えて mixed モードを実行する。
# mixed モードでは全87名で訓練し、各ドメイングループで評価する。
#
# データ量が約2倍になるため、リソースを増量している:
#   SvmA : 30h / 48GB   (source_only: 24h / 32GB)
#   SvmW : 16h / 24GB   (source_only: 12h / 16GB)
#   Lstm : 20h / 48GB   (source_only: 16h / 32GB)
#
# 予想ジョブ数:
#   SvmW : 3距離 × 2ドメイン × 2シード × 9条件 = 108
#   SvmA : 3距離 × 2ドメイン × 2シード × 7条件 =  84
#   Lstm : 3距離 × 2ドメイン × 2シード × 7条件 =  84
#   合計 : 276 jobs
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

# ---- 固定パラメータ ----
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
MODE="mixed"  # このランチャーは mixed 専用

DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODELS=("SvmW" "SvmA" "Lstm")

# キュー設定（SINGLE/LONG/DEFAULT にラウンドロビンで分散）
# 注意: SEMINARキューは6h壁時間制限があるため使用不可
USE_MULTI_QUEUE=true
# FIXED_QUEUE="SEMINAR"  # 6h制限のため使用禁止

# ---- 引数解析 ----
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *)         echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- キューカウンタ ----
QUEUE_COUNTER=0

# ---- リソース定義（mixed 用に増量） ----
get_resources() {
    local model="$1"
    local queue

    if $USE_MULTI_QUEUE; then
        local queues=("SINGLE" "LONG" "DEFAULT")
        queue="${queues[$((QUEUE_COUNTER % 3))]}"
        ((QUEUE_COUNTER++))
    else
        queue="${FIXED_QUEUE:-SEMINAR}"
    fi

    case "$model" in
        SvmA)  echo "ncpus=8:mem=48gb 30:00:00 $queue" ;;
        SvmW)  echo "ncpus=8:mem=24gb 16:00:00 $queue" ;;
        Lstm)  echo "ncpus=8:mem=48gb 20:00:00 $queue" ;;
    esac
}

# ---- 条件リスト ----
get_conditions() {
    local model="$1"
    case "$model" in
        SvmW)      echo "baseline smote_plain smote undersample" ;;
        SvmA|Lstm) echo "baseline smote_plain smote undersample" ;;
    esac
}

# ---- ログ ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_prior_research_mixed_${TIMESTAMP}.log"

echo "============================================================"
echo "先行研究実験ランチャー — mixed domain (2グループ分割版)"
echo "============================================================"
echo "モデル     : ${MODELS[*]}"
echo "分割方式   : split2 (in_domain=44名, out_domain=43名)"
echo "距離指標   : ${DISTANCES[*]}"
echo "ドメイン   : ${DOMAINS[*]}"
echo "訓練モード : $MODE (全87名で訓練、各ドメインで評価)"
echo "シード     : ${SEEDS[*]}"
echo "ターゲット比率 : ${RATIOS[*]}"
echo "Optuna trials  : $N_TRIALS (SvmWのみ)"
echo "複数キュー     : $USE_MULTI_QUEUE"
echo "Dry run        : $DRY_RUN"
echo "============================================================"
echo ""

# ---- スクリプト存在チェック ----
if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[ERROR] Job script not found: $JOB_SCRIPT"
    exit 1
fi

{
    echo "# Launch started at $(date)"
    echo "# Command: $0 $*"
    echo ""
} > "$LOG_FILE"

JOB_COUNT=0
SKIP_COUNT=0

# ---- メインループ ----
for MODEL in "${MODELS[@]}"; do
    CONDITIONS=$(get_conditions "$MODEL")

    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do

                # --- baseline (ratio なし) ---
                if echo "$CONDITIONS" | grep -q "baseline"; then
                    CONDITION="baseline"
                    RESOURCES=$(get_resources "$MODEL" "$CONDITION")
                    NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                    WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                    QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)

                    JOB_NAME="${MODEL:0:2}_bs_${DISTANCE:0:1}${DOMAIN:0:1}_m_s${SEED}"

                    CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                    CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                    CMD="$CMD $JOB_SCRIPT"

                    if $DRY_RUN; then
                        echo "[DRY-RUN] $MODEL | baseline | $DISTANCE | $DOMAIN | $MODE | s=$SEED"
                        ((JOB_COUNT++))
                    else
                        JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed: $CMD"; ((SKIP_COUNT++)); continue; }
                        echo "[SUBMIT] $MODEL | baseline | $DISTANCE | $DOMAIN | $MODE | s$SEED → $JOB_ID"
                        echo "$MODEL:baseline:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                        ((JOB_COUNT++))
                        sleep 0.2
                    fi
                fi

                # --- ratio ベース手法 ---
                for RATIO in "${RATIOS[@]}"; do
                    for COND in "smote_plain" "smote" "undersample" "balanced_rf"; do
                        if ! echo "$CONDITIONS" | grep -q "$COND"; then
                            continue
                        fi

                        RESOURCES=$(get_resources "$MODEL" "$COND")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)

                        COND_SHORT="${COND:0:2}"
                        JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_m_r${RATIO}_s${SEED}"

                        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                        CMD="$CMD -v MODEL=$MODEL,CONDITION=$COND,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                        CMD="$CMD $JOB_SCRIPT"

                        if $DRY_RUN; then
                            echo "[DRY-RUN] $MODEL | $COND | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s=$SEED"
                            ((JOB_COUNT++))
                        else
                            JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed: $CMD"; ((SKIP_COUNT++)); continue; }
                            echo "[SUBMIT] $MODEL | $COND | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s$SEED → $JOB_ID"
                            echo "$MODEL:$COND:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
                            ((JOB_COUNT++))
                            sleep 0.2
                        fi
                    done
                done

            done  # SEED
        done  # DOMAIN
    done  # DISTANCE
done  # MODEL

# ---- サマリー ----
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
echo "予想ジョブ数:"
echo "  SvmW : 3距離 × 2ドメイン × 2シード × (1 baseline + 2×4 ratio-based) = 108 jobs"
echo "  SvmA : 3距離 × 2ドメイン × 2シード × (1 baseline + 2×3 ratio-based) =  84 jobs"
echo "  Lstm : 3距離 × 2ドメイン × 2シード × (1 baseline + 2×3 ratio-based) =  84 jobs"
echo "  合計 : 276 jobs"
echo "============================================================"

#!/bin/bash
# 不均衡対策単体実験 - seed 42, 123 並列実行

cd "$(dirname "$0")/../.." || exit 1

# venv をアクティベート
source .venv-linux/bin/activate

# PYTHONPATHを設定
export PYTHONPATH="${PWD}:${PYTHONPATH}"

LOG_DIR="scripts/local/logs/imbalance"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL=RF
export N_TRIALS_OVERRIDE=50

echo "=========================================="
echo "不均衡対策単体実験 開始: $(date)"
echo "Seeds: 42, 123"
echo "=========================================="

# 共通オプション
# --subject_wise_split: subject_time_split戦略を使用
# --time_stratify_labels: 時間層化ラベル分割を有効化
COMMON_OPTS="--model $MODEL --mode pooled --subject_wise_split --time_stratify_labels"

# 実験タイプの定義
declare -A EXP_CONFIGS=(
    ["baseline"]=""
    ["smote_ratio0.1"]="--use_oversampling --oversample_method smote --target_ratio 0.1"
    ["smote_ratio0.5"]="--use_oversampling --oversample_method smote --target_ratio 0.5"
    ["subjectwise_smote_ratio0.1"]="--use_oversampling --oversample_method smote --target_ratio 0.1 --subject_wise_oversampling"
    ["subjectwise_smote_ratio0.5"]="--use_oversampling --oversample_method smote --target_ratio 0.5 --subject_wise_oversampling"
)

SEEDS=(42 123)
PIDS=()
EXPERIMENTS=()

# 全実験を並列起動
for seed in "${SEEDS[@]}"; do
    for exp_name in baseline smote_ratio0.1 smote_ratio0.5 subjectwise_smote_ratio0.1 subjectwise_smote_ratio0.5; do
        exp_opts="${EXP_CONFIGS[$exp_name]}"
        tag="${exp_name}_s${seed}"
        log_file="${LOG_DIR}/${tag}_${TIMESTAMP}.log"
        
        echo "[$(date +%H:%M:%S)] Starting: ${tag}"
        
        python scripts/python/train/train.py \
            $COMMON_OPTS \
            --seed "$seed" \
            --tag "$tag" \
            $exp_opts \
            > "$log_file" 2>&1 &
        
        PIDS+=($!)
        EXPERIMENTS+=("$tag")
    done
done

echo ""
echo "=========================================="
echo "全 ${#PIDS[@]} 実験を並列起動しました"
echo "PIDs: ${PIDS[*]}"
echo "=========================================="

# 完了待ち
FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    exp=${EXPERIMENTS[$i]}
    wait $pid
    status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] ✓ Completed: $exp"
    else
        echo "[$(date +%H:%M:%S)] ✗ Failed: $exp (exit code: $status)"
        ((FAILED++))
    fi
done

echo ""
echo "=========================================="
echo "全実験完了: $(date)"
echo "成功: $((${#PIDS[@]} - FAILED)) / ${#PIDS[@]}"
if [ $FAILED -gt 0 ]; then
    echo "失敗: $FAILED"
fi
echo "=========================================="

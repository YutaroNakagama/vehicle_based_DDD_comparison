#!/bin/bash
# DEFAULT/SMALLキューに集中投入（無制限キュー活用）

set -e

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

LOG_DIR="scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/submit_unlimited_queues_${TIMESTAMP}.log"
SUBMITTED_FILE="$LOG_DIR/submitted_unlimited_${TIMESTAMP}.txt"
touch "$SUBMITTED_FILE"

echo "============================================================" | tee -a "$LOG_FILE"
echo "無制限キュー（DEFAULT/SMALL）集中投入" | tee -a "$LOG_FILE"
echo "開始: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# 設定
MODELS="SvmW SvmA Lstm"
DISTANCES="mmd dtw wasserstein"
DOMAINS="out_domain in_domain"
MODES="source_only target_only"
SEEDS="42 123"
RATIOS="0.1 0.5"
RANKING="knn"
N_TRIALS=100

JOB_SCRIPT="$WORKSPACE_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

# 無制限キューのみ使用
QUEUES=("DEFAULT" "SMALL")
queue_index=0

total_submitted=0
total_failed=0
total_skipped=0

# リソース取得
get_resources() {
    local model=$1
    case $model in
        SvmW) echo "ncpus=8:mem=16gb 12:00:00" ;;
        SvmA) echo "ncpus=8:mem=32gb 24:00:00" ;;
        Lstm) echo "ncpus=8:mem=32gb 16:00:00" ;;
    esac
}

# ジョブ投入
submit_job() {
    local model=$1 condition=$2 distance=$3 domain=$4 mode=$5 seed=$6 ratio=$7
    
    # 重複チェック（全ログファイル）
    local job_id="${model}_${condition}_${distance}_${domain}_${mode}_${seed}"
    [ -n "$ratio" ] && job_id="${job_id}_${ratio}"
    
    if grep -rq "^${job_id}$" "$LOG_DIR"/submitted_*.txt 2>/dev/null; then
        ((total_skipped++))
        return 2
    fi
    
    # キュー選択（DEFAULT/SMALL交互）
    local queue="${QUEUES[$queue_index]}"
    queue_index=$(( (queue_index + 1) % 2 ))
    
    # ジョブ名
    local model_abbr="${model:0:2}"
    local cond_abbr="${condition:0:2}"
    local dist_abbr="${distance:0:1}"
    local domain_abbr="${domain:0:1}"
    local mode_abbr="${mode:0:1}"
    
    local job_name="${model_abbr}_${cond_abbr}_${dist_abbr}${domain_abbr}_${mode_abbr}"
    [ -n "$ratio" ] && job_name="${job_name}_r${ratio}" || job_name="${job_name}"
    job_name="${job_name}_s${seed}"
    
    # リソース
    local resources=$(get_resources "$model")
    local ncpus_mem=$(echo "$resources" | awk '{print $1}')
    local walltime=$(echo "$resources" | awk '{print $2}')
    
    # 環境変数
    local env_vars="MODEL=$model,CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    [ -n "$ratio" ] && env_vars="${env_vars},RATIO=$ratio"
    
    # 投入
    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $env_vars $JOB_SCRIPT"
    
    if output=$($cmd 2>&1); then
        echo "$job_id" >> "$SUBMITTED_FILE"
        echo "[$(date +%H:%M:%S)] [$queue] ✓ $job_name → $output" | tee -a "$LOG_FILE"
        ((total_submitted++))
        return 0
    else
        if [[ "$output" == *"would exceed"* ]]; then
            echo "[上限到達] 一時停止（60秒後に再試行）" | tee -a "$LOG_FILE"
            sleep 60
            # リトライ
            if output=$($cmd 2>&1); then
                echo "$job_id" >> "$SUBMITTED_FILE"
                echo "[$(date +%H:%M:%S)] [$queue] ✓ $job_name → $output (retry)" | tee -a "$LOG_FILE"
                ((total_submitted++))
                return 0
            fi
        fi
        echo "[$(date +%H:%M:%S)] [$queue] ✗ $job_name" >> "$LOG_FILE"
        ((total_failed++))
        return 1
    fi
}

# メインループ
echo "[$(date +%H:%M:%S)] 投入開始..." | tee -a "$LOG_FILE"

for model in $MODELS; do
    [ "$model" = "SvmW" ] && CONDITIONS="baseline smote_plain smote undersample balanced_rf" || CONDITIONS="baseline smote_plain smote undersample"
    
    for condition in $CONDITIONS; do
        for distance in $DISTANCES; do
            for domain in $DOMAINS; do
                for mode in $MODES; do
                    for seed in $SEEDS; do
                        if [ "$condition" = "baseline" ]; then
                            submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" ""
                        else
                            for ratio in $RATIOS; do
                                submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" "$ratio"
                            done
                        fi
                    done
                done
            done
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "投入完了" | tee -a "$LOG_FILE"
echo "成功: $total_submitted | 失敗: $total_failed | スキップ: $total_skipped" | tee -a "$LOG_FILE"
echo "終了: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "DEFAULT: $(grep -c '\[DEFAULT\] ✓' "$LOG_FILE" || echo 0) ジョブ" | tee -a "$LOG_FILE"
echo "SMALL: $(grep -c '\[SMALL\] ✓' "$LOG_FILE" || echo 0) ジョブ" | tee -a "$LOG_FILE"
echo ""
echo "ログ: $LOG_FILE"

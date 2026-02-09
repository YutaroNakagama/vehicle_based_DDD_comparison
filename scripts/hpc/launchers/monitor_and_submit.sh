#!/bin/bash
# ジョブキューを監視して空きができたら自動投入

set -e

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

LOG_DIR="scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/monitor_submit_${TIMESTAMP}.log"
SUBMITTED_FILE="$LOG_DIR/submitted_jobs_${TIMESTAMP}.txt"
touch "$SUBMITTED_FILE"

# ユーザージョブ上限（安全のため45に設定）
MAX_USER_JOBS=45
CHECK_INTERVAL=60  # 1分ごとにチェック
MIN_SLOTS=5  # 最低5スロット空いたら投入開始

# キューのリスト（より多くのキューに分散投入）
QUEUES=("SINGLE" "LONG" "DEFAULT" "SMALL" "LARGE" "XLARGE")
QUEUE_INDEX=0

echo "============================================================" | tee -a "$LOG_FILE"
echo "先行研究実験 監視自動投入スクリプト" | tee -a "$LOG_FILE"
echo "開始時刻: $(date)" | tee -a "$LOG_FILE"
echo "ユーザー上限: $MAX_USER_JOBS ジョブ" | tee -a "$LOG_FILE"
echo "チェック間隔: ${CHECK_INTERVAL}秒" | tee -a "$LOG_FILE"
echo "最小空きスロット: $MIN_SLOTS" | tee -a "$LOG_FILE"
echo "使用キュー: ${QUEUES[*]} (ラウンドロビン)" | tee -a "$LOG_FILE"
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

total_submitted=0
total_failed=0

# リソース取得関数
get_resources() {
    local model=$1
    
    case $model in
        SvmW)
            echo "ncpus=8:mem=16gb 12:00:00"
            ;;
        SvmA)
            echo "ncpus=8:mem=32gb 24:00:00"
            ;;
        Lstm)
            echo "ncpus=8:mem=32gb 16:00:00"
            ;;
    esac
}

# ジョブ投入関数
submit_job() {
    local model=$1
    local condition=$2
    local distance=$3
    local domain=$4
    local mode=$5
    local seed=$6
    local ratio=$7
    
    # ジョブIDを生成（重複チェック用）
    local job_id="${model}_${condition}_${distance}_${domain}_${mode}_${seed}"
    if [ -n "$ratio" ]; then
        job_id="${job_id}_${ratio}"
    fi
    
    # 既に投入済みかチェック
    if grep -q "^${job_id}$" "$SUBMITTED_FILE" 2>/dev/null; then
        return 2  # 既に投入済み
    fi
    
    # キューを選択（ラウンドロビン）
    local queue="${QUEUES[$QUEUE_INDEX]}"
    QUEUE_INDEX=$(( (QUEUE_INDEX + 1) % ${#QUEUES[@]} ))
    
    # ジョブ名を生成
    local model_abbr="${model:0:2}"
    local cond_abbr="${condition:0:2}"
    local dist_abbr="${distance:0:1}"
    local domain_abbr="${domain:0:1}"
    local mode_abbr="${mode:0:1}"
    
    if [ -n "$ratio" ]; then
        local job_name="${model_abbr}_${cond_abbr}_${dist_abbr}${domain_abbr}_${mode_abbr}_r${ratio}_s${seed}"
    else
        local job_name="${model_abbr}_${cond_abbr}_${dist_abbr}${domain_abbr}_${mode_abbr}_s${seed}"
    fi
    
    # リソース設定
    local resources=$(get_resources "$model")
    local ncpus_mem=$(echo "$resources" | awk '{print $1}')
    local walltime=$(echo "$resources" | awk '{print $2}')
    
    # 環境変数を準備
    local env_vars="MODEL=$model,CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    if [ -n "$ratio" ]; then
        env_vars="${env_vars},RATIO=$ratio"
    fi
    
    # qsubコマンド実行
    local qsub_cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $env_vars $JOB_SCRIPT"
    
    if output=$($qsub_cmd 2>&1); then
        echo "$job_id" >> "$SUBMITTED_FILE"
        echo "[$(date +%H:%M:%S)] [$queue] OK: $job_name → $output" | tee -a "$LOG_FILE"
        ((total_submitted++))
        return 0
    else
        echo "[$(date +%H:%M:%S)] [$queue] NG: $job_name ($output)" >> "$LOG_FILE"
        ((total_failed++))
        return 1
    fi
}

# 現在のユーザージョブ数を取得
get_current_jobs() {
    qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l || echo "0"
}

# メインループ
echo "[$(date +%H:%M:%S)] ジョブ投入を開始します..." | tee -a "$LOG_FILE"

for model in $MODELS; do
    # balanced_rfはSvmWのみ
    if [ "$model" = "SvmW" ]; then
        CONDITIONS="baseline smote_plain smote undersample balanced_rf"
    else
        CONDITIONS="baseline smote_plain smote undersample"
    fi
    
    for condition in $CONDITIONS; do
        for distance in $DISTANCES; do
            for domain in $DOMAINS; do
                for mode in $MODES; do
                    for seed in $SEEDS; do
                        # 投入前にスロット数をチェック
                        while true; do
                            current_jobs=$(get_current_jobs)
                            available_slots=$((MAX_USER_JOBS - current_jobs))
                            
                            if [ $available_slots -ge $MIN_SLOTS ]; then
                                break
                            fi
                            
                            echo "[$(date +%H:%M:%S)] 待機中... (現在: $current_jobs/$MAX_USER_JOBS, 空き: $available_slots)" | tee -a "$LOG_FILE"
                            sleep $CHECK_INTERVAL
                        done
                        
                        if [ "$condition" = "baseline" ]; then
                            # baselineは比率なし
                            submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" ""
                            sleep 0.2
                        else
                            # その他の条件は比率あり
                            for ratio in $RATIOS; do
                                # 各投入前にスロット数をチェック
                                current_jobs=$(get_current_jobs)
                                available_slots=$((MAX_USER_JOBS - current_jobs))
                                
                                while [ $available_slots -lt $MIN_SLOTS ]; do
                                    echo "[$(date +%H:%M:%S)] 待機中... (現在: $current_jobs/$MAX_USER_JOBS, 空き: $available_slots)" | tee -a "$LOG_FILE"
                                    sleep $CHECK_INTERVAL
                                    current_jobs=$(get_current_jobs)
                                    available_slots=$((MAX_USER_JOBS - current_jobs))
                                done
                                
                                submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" "$ratio"
                                sleep 0.2
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
echo "すべてのジョブ投入処理が完了しました" | tee -a "$LOG_FILE"
echo "投入成功: $total_submitted ジョブ" | tee -a "$LOG_FILE"
echo "投入失敗: $total_failed ジョブ" | tee -a "$LOG_FILE"
echo "終了時刻: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "ログファイル: $LOG_FILE"
echo "投入済みリスト: $SUBMITTED_FILE"

# キューごとの投入数を表示
echo "" | tee -a "$LOG_FILE"
echo "キュー別投入数:" | tee -a "$LOG_FILE"
for queue in "${QUEUES[@]}"; do
    count=$(grep -c "\[$queue\] OK:" "$LOG_FILE" 2>/dev/null || echo "0")
    echo "  $queue: $count ジョブ" | tee -a "$LOG_FILE"
done

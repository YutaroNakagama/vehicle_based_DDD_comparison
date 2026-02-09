#!/bin/bash
# 先行研究実験の自動投入スクリプト
# キューの空きを監視して順次投入

set -e

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

LOG_DIR="scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/auto_submit_prior_${TIMESTAMP}.log"

# ジョブ上限を設定 (安全のため48に設定)
MAX_JOBS=48
CHECK_INTERVAL=300  # 5分ごとにチェック

echo "============================================================" | tee -a "$LOG_FILE"
echo "先行研究実験 自動投入スクリプト" | tee -a "$LOG_FILE"
echo "開始時刻: $(date)" | tee -a "$LOG_FILE"
echo "ジョブ上限: $MAX_JOBS" | tee -a "$LOG_FILE"
echo "チェック間隔: ${CHECK_INTERVAL}秒" | tee -a "$LOG_FILE"
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

# 投入済みジョブを記録
SUBMITTED_JOBS_FILE="$LOG_DIR/submitted_prior_jobs.txt"
touch "$SUBMITTED_JOBS_FILE"

total_submitted=0
total_skipped=0
total_errors=0

# ジョブ投入関数
submit_job() {
    local model=$1
    local condition=$2
    local distance=$3
    local domain=$4
    local mode=$5
    local seed=$6
    local ratio=$7
    
    # ジョブIDを生成 (重複チェック用)
    local job_id="${model}_${condition}_${distance}_${domain}_${mode}_${seed}"
    if [ -n "$ratio" ]; then
        job_id="${job_id}_${ratio}"
    fi
    
    # 既に投入済みかチェック
    if grep -q "^${job_id}$" "$SUBMITTED_JOBS_FILE" 2>/dev/null; then
        return 1  # 既に投入済み
    fi
    
    # ジョブ名を生成
    local model_abbr="${model:0:2}"
    local cond_abbr="${condition:0:2}"
    local dist_abbr="${distance:0:1}"
    local domain_abbr="${domain:0:1}"
    local mode_abbr="${mode:0:1}"
    local job_name="${model_abbr}_${cond_abbr}_${dist_abbr}${domain_abbr}_${mode_abbr}_s${seed}"
    
    # リソース設定
    local walltime mem
    case $model in
        SvmW)
            walltime="12:00:00"
            mem="16gb"
            ;;
        SvmA)
            walltime="24:00:00"
            mem="32gb"
            ;;
        Lstm)
            walltime="16:00:00"
            mem="32gb"
            ;;
    esac
    
    # 環境変数を準備
    local env_vars="MODEL=$model,CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    if [ -n "$ratio" ]; then
        env_vars="${env_vars},RATIO=$ratio"
    fi
    
    # qsubコマンド実行
    local qsub_cmd="qsub -N $job_name -l select=1:ncpus=8:mem=$mem -l walltime=$walltime -q SINGLE -v $env_vars $JOB_SCRIPT"
    
    if $qsub_cmd >> "$LOG_FILE" 2>&1; then
        echo "$job_id" >> "$SUBMITTED_JOBS_FILE"
        echo "[$(date +%H:%M:%S)] 投入成功: $job_id" | tee -a "$LOG_FILE"
        ((total_submitted++))
        return 0
    else
        echo "[$(date +%H:%M:%S)] 投入失敗: $job_id" | tee -a "$LOG_FILE"
        ((total_errors++))
        return 1
    fi
}

# メインループ
while true; do
    # 現在のジョブ数を確認
    current_jobs=$(qstat -u s2240011 2>/dev/null | grep -E "R|Q" | wc -l || echo "0")
    available_slots=$((MAX_JOBS - current_jobs))
    
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date +%H:%M:%S)] 現在のジョブ数: $current_jobs / $MAX_JOBS (空き: $available_slots)" | tee -a "$LOG_FILE"
    
    if [ $available_slots -le 0 ]; then
        echo "[$(date +%H:%M:%S)] キューが満杯です。${CHECK_INTERVAL}秒後に再チェックします..." | tee -a "$LOG_FILE"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # 投入可能な数だけジョブを投入
    submitted_in_round=0
    
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
                            if [ "$condition" = "baseline" ]; then
                                # baselineは比率なし
                                if submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" ""; then
                                    ((submitted_in_round++))
                                fi
                            else
                                # その他の条件は比率あり
                                for ratio in $RATIOS; do
                                    if submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" "$ratio"; then
                                        ((submitted_in_round++))
                                    fi
                                    
                                    # 投入上限に達したらbreak
                                    if [ $submitted_in_round -ge $available_slots ]; then
                                        break 5
                                    fi
                                done
                            fi
                            
                            # 投入上限に達したらbreak
                            if [ $submitted_in_round -ge $available_slots ]; then
                                break 4
                            fi
                        done
                    done
                done
            done
        done
    done
    
    echo "[$(date +%H:%M:%S)] このラウンドで投入: $submitted_in_round ジョブ" | tee -a "$LOG_FILE"
    
    # すべてのジョブを投入完了したかチェック
    total_expected=552
    total_done=$((total_submitted + total_skipped))
    
    if [ $total_done -ge $total_expected ]; then
        echo "" | tee -a "$LOG_FILE"
        echo "============================================================" | tee -a "$LOG_FILE"
        echo "すべてのジョブ投入が完了しました！" | tee -a "$LOG_FILE"
        echo "投入成功: $total_submitted ジョブ" | tee -a "$LOG_FILE"
        echo "スキップ: $total_skipped ジョブ" | tee -a "$LOG_FILE"
        echo "エラー: $total_errors ジョブ" | tee -a "$LOG_FILE"
        echo "終了時刻: $(date)" | tee -a "$LOG_FILE"
        echo "============================================================" | tee -a "$LOG_FILE"
        break
    fi
    
    # 次のチェックまで待機
    if [ $submitted_in_round -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] これ以上投入できるジョブがありません。${CHECK_INTERVAL}秒後に再チェックします..." | tee -a "$LOG_FILE"
        sleep $CHECK_INTERVAL
    else
        # ジョブを投入したら少し待つ
        sleep 10
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "ログファイル: $LOG_FILE"
echo "投入済みジョブリスト: $SUBMITTED_JOBS_FILE"

#!/bin/bash
# ユーザー上限を無視して大量投入（DEFAULTとSMALLは無制限）

WORKSPACE_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$WORKSPACE_ROOT"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="scripts/hpc/logs/train/massive_submit_${TIMESTAMP}.log"
SUBMITTED_FILE="scripts/hpc/logs/train/submitted_jobs_${TIMESTAMP}.txt"
touch "$SUBMITTED_FILE"

echo "============================================================"
echo "大量投入スクリプト (DEFAULT/SMALL優先)"
echo "開始: $(date)"
echo "============================================================"

MODELS="SvmW SvmA Lstm"
DISTANCES="mmd dtw wasserstein"
DOMAINS="out_domain in_domain"
MODES="source_only target_only"
SEEDS="42 123"
RATIOS="0.1 0.5"
RANKING="knn"
N_TRIALS=100

JOB_SCRIPT="$WORKSPACE_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

# DEFAULT/SMALL/LARGEを優先的に使用
QUEUES=("DEFAULT" "SMALL" "LARGE" "XLARGE" "LONG" "SINGLE")
queue_index=0

total_submitted=0
total_skipped=0

get_resources() {
    case $1 in
        SvmW) echo "ncpus=8:mem=16gb 12:00:00" ;;
        SvmA) echo "ncpus=8:mem=32gb 24:00:00" ;;
        Lstm) echo "ncpus=8:mem=32gb 16:00:00" ;;
    esac
}

submit_job() {
    local job_id="${1}_${2}_${3}_${4}_${5}_${6}"
    [ -n "$7" ] && job_id="${job_id}_${7}"
    
    # 既投入チェック
    if grep -q "^${job_id}$" scripts/hpc/logs/train/submitted_jobs_*.txt 2>/dev/null; then
        ((total_skipped++))
        return 2
    fi
    
    local queue="${QUEUES[$queue_index]}"
    queue_index=$(( (queue_index + 1) % ${#QUEUES[@]} ))
    
    local model_abbr="${1:0:2}"
    local cond_abbr="${2:0:2}"
    local dist_abbr="${3:0:1}"
    local domain_abbr="${4:0:1}"
    local mode_abbr="${5:0:1}"
    
    local job_name="${model_abbr}_${cond_abbr}_${dist_abbr}${domain_abbr}_${mode_abbr}"
    [ -n "$7" ] && job_name="${job_name}_r${7}" || true
    job_name="${job_name}_s${6}"
    
    local resources=$(get_resources "$1")
    local ncpus_mem=$(echo "$resources" | awk '{print $1}')
    local walltime=$(echo "$resources" | awk '{print $2}')
    
    local env_vars="MODEL=$1,CONDITION=$2,MODE=$5,DISTANCE=$3,DOMAIN=$4,SEED=$6,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    [ -n "$7" ] && env_vars="${env_vars},RATIO=$7" || true
    
    if output=$(qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $env_vars $JOB_SCRIPT 2>&1); then
        echo "$job_id" >> "$SUBMITTED_FILE"
        echo "[$queue] ✓ $job_name → $output"
        ((total_submitted++))
        return 0
    else
        # エラーを無視して継続
        echo "[$queue] × $job_name"
        return 1
    fi
}

for model in $MODELS; do
    [ "$model" = "SvmW" ] && CONDITIONS="baseline smote_plain smote undersample balanced_rf" || CONDITIONS="baseline smote_plain smote undersample"
    
    for condition in $CONDITIONS; do
        for distance in $DISTANCES; do
            for domain in $DOMAINS; do
                for mode in $MODES; do
                    for seed in $SEEDS; do
                        if [ "$condition" = "baseline" ]; then
                            submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" "" || true
                            sleep 0.05
                        else
                            for ratio in $RATIOS; do
                                submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" "$ratio" || true
                                sleep 0.05
                            done
                        fi
                    done
                done
            done
        done
    done
done

echo ""
echo "============================================================"
echo "投入完了"
echo "成功: $total_submitted, スキップ: $total_skipped"
echo "終了: $(date)"
echo "============================================================"

#!/bin/bash
# ============================================================
# Exp2 Mixed-domain ジョブ投入ランチャー
# ============================================================
# 全被験者（87名）で訓練し、in_domain / out_domain で評価する
# Mixed-domain ケースの全96ジョブを投入する
#
# ジョブ数: 3 (距離) × 2 (ドメイン) × 2 (シード) × 8 (条件) = 96
#
# Usage:
#   bash scripts/hpc/launchers/launch_exp2_mixed.sh
#   bash scripts/hpc/launchers/launch_exp2_mixed.sh --dry-run
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

# Mode is always mixed for this launcher
MODE="mixed"

# Parameters
DISTANCES="mmd dtw wasserstein"
DOMAINS="in_domain out_domain"
SEEDS="42 123"
RATIOS="0.1 0.5"
RANKING="knn"
N_TRIALS=100

# Queues (round-robin across available queues)
QUEUES=("SEMINAR" "SMALL" "SEMINAR" "SMALL")
QUEUE_INDEX=0

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_exp2_mixed_${TIMESTAMP}.log"

JOB_COUNT=0
FAIL_COUNT=0

echo "============================================================"
echo "  Exp2 Mixed-domain ジョブ投入"
echo "  $(date)"
echo "============================================================"
echo "  MODE: $MODE (train=ALL 87 subjects, eval=target domain)"
echo "  DRY_RUN: $DRY_RUN"
echo ""

{
    echo "# Exp2 Mixed-domain launch: $(date)"
    echo "# MODE=$MODE"
    echo ""
} > "$LOG_FILE"

# Resource settings per condition
get_resources() {
    local condition="$1"
    case "$condition" in
        balanced_rf) echo "ncpus=8:mem=12gb 08:00:00 LONG" ;;
        smote|smote_plain) echo "ncpus=4:mem=10gb 08:00:00 SINGLE" ;;
        baseline|undersample) echo "ncpus=4:mem=8gb 06:00:00 SINGLE" ;;
    esac
}

submit_job() {
    local condition="$1"
    local distance="$2"
    local domain="$3"
    local seed="$4"
    local ratio="${5:-}"

    # Short job name
    local c_short="${condition:0:2}"
    local d_short="${distance:0:1}"
    local dom_short="${domain:0:1}"
    local r_tag=""
    [[ -n "$ratio" ]] && r_tag="_r${ratio}"

    local job_name="mx_${c_short}_${d_short}${dom_short}${r_tag}_s${seed}"

    # Resources
    local res=$(get_resources "$condition")
    local ncpus_mem=$(echo "$res" | awk '{print $1}')
    local walltime=$(echo "$res" | awk '{print $2}')

    # Queue (round-robin)
    local queue="${QUEUES[$QUEUE_INDEX]}"
    QUEUE_INDEX=$(( (QUEUE_INDEX + 1) % ${#QUEUES[@]} ))

    # Environment variables
    local env_vars="CONDITION=$condition,MODE=$MODE,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    [[ -n "$ratio" ]] && env_vars="${env_vars},RATIO=$ratio"

    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $env_vars $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY] [$queue] $condition | $distance | $domain | mixed | r=${ratio:-N/A} | s$seed"
        ((JOB_COUNT++))
        return 0
    fi

    if job_id=$(eval "$cmd" 2>&1); then
        echo "[OK] [$queue] $job_name → $job_id"
        echo "$queue:$condition:$distance:$domain:mixed:${ratio:-}:$seed:$job_id" >> "$LOG_FILE"
        ((JOB_COUNT++))
    else
        echo "[FAIL] [$queue] $job_name — $job_id"
        ((FAIL_COUNT++))
    fi
    sleep 0.1
}

# --- Submit all jobs ---
for distance in $DISTANCES; do
    for domain in $DOMAINS; do
        for seed in $SEEDS; do

            # 1. baseline (no ratio)
            submit_job baseline "$distance" "$domain" "$seed"

            # 2-3. smote_plain × 2 ratios
            for ratio in $RATIOS; do
                submit_job smote_plain "$distance" "$domain" "$seed" "$ratio"
            done

            # 4-5. smote (subject-wise) × 2 ratios
            for ratio in $RATIOS; do
                submit_job smote "$distance" "$domain" "$seed" "$ratio"
            done

            # 6-7. undersample × 2 ratios
            for ratio in $RATIOS; do
                submit_job undersample "$distance" "$domain" "$seed" "$ratio"
            done

            # 8. balanced_rf (no ratio)
            submit_job balanced_rf "$distance" "$domain" "$seed"

        done
    done
done

{
    echo ""
    echo "# Completed: $(date)"
    echo "# Submitted: $JOB_COUNT"
    echo "# Failed: $FAIL_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Submitted: $JOB_COUNT"
echo "  Failed:    $FAIL_COUNT"
echo "  Log: $LOG_FILE"
echo "============================================================"

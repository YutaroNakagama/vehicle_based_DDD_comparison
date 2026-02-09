#!/bin/bash
# ============================================================
# 実験2 残りジョブ再投入ランチャー (2026-02-07)
# ============================================================
# launch_remaining_exp2.sh で投入できなかった 112 ジョブを投入する。
# キュー上限に達した場合は途中で停止し、再実行で残りを投入する。
#
# 投入済み (40件):
#   - baseline pooled ×2
#   - smote_plain: mmd全16 + dtw全16 + wasserstein/in_domain/source_only×4
#     + wasserstein/in_domain/target_only×2 (r=0.1/s42, r=0.5/s42)
#
# 残り (112件):
#   - smote_plain: wasserstein 残り10 + pooled 2 = 12
#   - smote:       全48 + pooled 2 = 50
#   - undersample: 全48 + pooled 2 = 50
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
DOMAIN_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
POOLED_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/imbalance/pbs_imbalance_comparison.sh"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("in_domain" "out_domain")
MODES=("source_only" "target_only")

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_remaining_exp2_retry_${TIMESTAMP}.log"

JOB_COUNT=0
FAIL_COUNT=0
QUEUE_FULL=false

# --- Helper: submit domain-split job ---
submit_domain() {
    if $QUEUE_FULL; then return 1; fi
    local condition="$1" mode="$2" distance="$3" domain="$4" seed="$5" ratio="${6:-}"

    local ncpus_mem walltime queue
    case "$condition" in
        smote|smote_plain) ncpus_mem="ncpus=4:mem=10gb"; walltime="08:00:00"; queue="SINGLE" ;;
        undersample)       ncpus_mem="ncpus=4:mem=8gb";  walltime="06:00:00"; queue="SINGLE" ;;
        baseline)          ncpus_mem="ncpus=4:mem=8gb";  walltime="06:00:00"; queue="SINGLE" ;;
        balanced_rf)       ncpus_mem="ncpus=8:mem=12gb"; walltime="08:00:00"; queue="LONG"   ;;
        *)                 ncpus_mem="ncpus=4:mem=8gb";  walltime="08:00:00"; queue="SINGLE" ;;
    esac

    local cond_short="${condition:0:2}"
    local job_name="${cond_short}_${distance:0:1}${domain:0:1}_${mode:0:1}"
    [[ -n "$ratio" ]] && job_name="${job_name}_r${ratio}_s${seed}" || job_name="${job_name}_s${seed}"

    local vars="CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    [[ -n "$ratio" ]] && vars="$vars,RATIO=$ratio"

    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $vars $DOMAIN_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] $condition | $distance | $domain | $mode | r=${ratio:-N/A} | s$seed"
        ((JOB_COUNT++))
    else
        local job_id
        if job_id=$(eval "$cmd" 2>&1); then
            echo "[SUBMIT] $condition | $distance | $domain | $mode | r=${ratio:-N/A} | s$seed → $job_id"
            echo "$condition:$distance:$domain:$mode:${ratio:-N/A}:$seed:$job_id" >> "$LOG_FILE"
            ((JOB_COUNT++))
        else
            echo "[FAIL]   Queue full after $JOB_COUNT submitted. Stopping."
            QUEUE_FULL=true
            ((FAIL_COUNT++))
            return 1
        fi
        sleep 0.3
    fi
}

# --- Helper: submit pooled job ---
submit_pooled() {
    if $QUEUE_FULL; then return 1; fi
    local method="$1" seed="$2" ratio="${3:-0.5}"

    local ncpus_mem walltime queue
    case "$method" in
        smote|smote_plain) ncpus_mem="ncpus=4:mem=10gb"; walltime="08:00:00"; queue="SINGLE" ;;
        undersample*|baseline) ncpus_mem="ncpus=4:mem=8gb"; walltime="06:00:00"; queue="SINGLE" ;;
        *)                 ncpus_mem="ncpus=4:mem=8gb";  walltime="08:00:00"; queue="SINGLE" ;;
    esac

    local pbs_method="$method"
    case "$method" in
        smote)       pbs_method="smote_subjectwise" ;;
        undersample) pbs_method="undersample_rus"   ;;
    esac

    local job_name="pl_${method:0:2}_s${seed}"
    local vars="METHOD=$pbs_method,SEED=$seed,RATIO=$ratio,N_TRIALS=$N_TRIALS,RUN_EVAL=true"
    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $vars $POOLED_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] pooled | $method | s$seed"
        ((JOB_COUNT++))
    else
        local job_id
        if job_id=$(eval "$cmd" 2>&1); then
            echo "[SUBMIT] pooled | $method | s$seed → $job_id"
            echo "pooled:$method:$seed:$job_id" >> "$LOG_FILE"
            ((JOB_COUNT++))
        else
            echo "[FAIL]   Queue full after $JOB_COUNT submitted. Stopping."
            QUEUE_FULL=true
            ((FAIL_COUNT++))
            return 1
        fi
        sleep 0.3
    fi
}

# ============================================================
echo "============================================================"
echo "  実験2 残りジョブ再投入 ($(date))"
echo "============================================================"
echo "  Dry run : $DRY_RUN"
echo "  Expected: 112 jobs"
echo ""

{
    echo "# Retry launch started at $(date)"
    echo ""
} > "$LOG_FILE"

# --- smote_plain: wasserstein remaining 10 + pooled 2 ---
echo "--- smote_plain 残り (10 domain + 2 pooled) ---"
# wasserstein/in_domain/target_only  r=0.1,s123 and r=0.5,s123
for RATIO in "${RATIOS[@]}"; do
    submit_domain "smote_plain" "target_only" "wasserstein" "in_domain" "123" "$RATIO"
done
# wasserstein/out_domain (all 8)
for MODE in "${MODES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for RATIO in "${RATIOS[@]}"; do
            submit_domain "smote_plain" "$MODE" "wasserstein" "out_domain" "$SEED" "$RATIO"
        done
    done
done
# smote_plain pooled
for SEED in "${SEEDS[@]}"; do
    submit_pooled "smote_plain" "$SEED"
done

# --- smote: all 48 domain + 2 pooled ---
echo ""
echo "--- smote 全 50 ジョブ ---"
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                for RATIO in "${RATIOS[@]}"; do
                    submit_domain "smote" "$MODE" "$DISTANCE" "$DOMAIN" "$SEED" "$RATIO"
                done
            done
        done
    done
done
for SEED in "${SEEDS[@]}"; do
    submit_pooled "smote" "$SEED"
done

# --- undersample: all 48 domain + 2 pooled ---
echo ""
echo "--- undersample 全 50 ジョブ ---"
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                for RATIO in "${RATIOS[@]}"; do
                    submit_domain "undersample" "$MODE" "$DISTANCE" "$DOMAIN" "$SEED" "$RATIO"
                done
            done
        done
    done
done
for SEED in "${SEEDS[@]}"; do
    submit_pooled "undersample" "$SEED"
done

# --- Summary ---
{
    echo ""
    echo "# Completed at $(date)"
    echo "# Submitted: $JOB_COUNT"
    echo "# Queue full stops: $FAIL_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "  Dry run: would submit $JOB_COUNT jobs"
elif $QUEUE_FULL; then
    echo "  Submitted: $JOB_COUNT jobs (queue full, stopped early)"
    echo "  Remaining: $((112 - JOB_COUNT)) jobs — re-run this script later"
else
    echo "  Submitted: $JOB_COUNT jobs (all done)"
fi
echo "  Log: $LOG_FILE"
echo "============================================================"

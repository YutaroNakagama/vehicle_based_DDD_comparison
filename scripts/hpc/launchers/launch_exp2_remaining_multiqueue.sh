#!/bin/bash
# ============================================================
# 実験2 未投入ジョブ マルチキュー分散投入 (2026-02-07)
# ============================================================
# 完了済み eval + キュー内ジョブを差し引いた、真に未投入の 62 ジョブを
# TINY / DEFAULT / SINGLE / SEMINAR に分散して投入する。
#
# 内訳:
#   smote_plain : 33 jobs (RF, 4cpu/10gb, 8h)
#   undersample : 17 jobs (RF, 4cpu/8gb,  6h)
#   smote (sw)  : 12 jobs (RF, 4cpu/10gb, 8h)
#   合計        : 62 jobs
#
# キュー分散方針:
#   TINY    : 15 jobs  (空いている)
#   DEFAULT : 16 jobs  (余裕あり)
#   SINGLE  : 16 jobs  (メインキュー)
#   SEMINAR : 15 jobs  (空いている)
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
DOMAIN_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_exp2_remaining_multiqueue_${TIMESTAMP}.log"

# Queue rotation
QUEUES=("TINY" "DEFAULT" "SINGLE" "SEMINAR")
QUEUE_IDX=0
JOB_COUNT=0
FAIL_COUNT=0

submit_domain() {
    local condition="$1" mode="$2" distance="$3" domain="$4" seed="$5" ratio="$6"
    local queue="${QUEUES[$QUEUE_IDX]}"
    QUEUE_IDX=$(( (QUEUE_IDX + 1) % ${#QUEUES[@]} ))

    local ncpus_mem walltime
    case "$condition" in
        smote|smote_plain) ncpus_mem="ncpus=4:mem=10gb"; walltime="08:00:00" ;;
        undersample)       ncpus_mem="ncpus=4:mem=8gb";  walltime="06:00:00" ;;
    esac

    local cond_short="${condition:0:2}"
    local job_name="e2_${cond_short}_${distance:0:1}${domain:0:1}_${mode:0:1}_r${ratio}_s${seed}"

    local vars="CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=$ratio"
    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $vars $DOMAIN_SCRIPT"

    if $DRY_RUN; then
        printf "[DRY-RUN] %-7s %-12s %-12s %-11s %-12s r%-3s s%-3s → %s\n" \
            "$queue" "$condition" "$distance" "$domain" "$mode" "$ratio" "$seed" "$job_name"
        ((JOB_COUNT++))
    else
        local job_id
        if job_id=$(eval "$cmd" 2>&1); then
            printf "[SUBMIT] %-7s %-12s %-12s %-11s %-12s r%-3s s%-3s → %s\n" \
                "$queue" "$condition" "$distance" "$domain" "$mode" "$ratio" "$seed" "$job_id"
            echo "$condition:$distance:$domain:$mode:$ratio:$seed:$queue:$job_id" >> "$LOG_FILE"
            ((JOB_COUNT++))
        else
            printf "[FAIL]   %-7s %-12s %-12s %-11s %-12s r%-3s s%-3s → %s\n" \
                "$queue" "$condition" "$distance" "$domain" "$mode" "$ratio" "$seed" "$job_id"
            ((FAIL_COUNT++))
        fi
        sleep 0.3
    fi
}

echo "============================================================"
echo "  実験2 未投入ジョブ マルチキュー分散投入"
echo "  $(date)"
echo "============================================================"
echo "  Dry run : $DRY_RUN"
echo "  Expected: 62 jobs"
echo "  Queues  : ${QUEUES[*]}"
echo ""

{ echo "# Launch started at $(date)"; echo ""; } > "$LOG_FILE"

# ============================================================
# smote_plain: 33 missing jobs
# ============================================================
echo "--- smote_plain (33 jobs) ---"

# dtw - all 16 combos except dtw/in_domain/source_only/0.5/123 (in queue)
submit_domain "smote_plain" "source_only" "dtw" "in_domain"  "42"  "0.1"
submit_domain "smote_plain" "source_only" "dtw" "in_domain"  "42"  "0.5"
# dtw/in_domain/source_only/123/0.5 → in queue, skip
submit_domain "smote_plain" "target_only" "dtw" "in_domain"  "42"  "0.1"
submit_domain "smote_plain" "target_only" "dtw" "in_domain"  "42"  "0.5"
submit_domain "smote_plain" "target_only" "dtw" "in_domain"  "123" "0.1"
submit_domain "smote_plain" "target_only" "dtw" "in_domain"  "123" "0.5"
submit_domain "smote_plain" "source_only" "dtw" "out_domain" "42"  "0.1"
submit_domain "smote_plain" "source_only" "dtw" "out_domain" "42"  "0.5"
submit_domain "smote_plain" "source_only" "dtw" "out_domain" "123" "0.1"
submit_domain "smote_plain" "source_only" "dtw" "out_domain" "123" "0.5"
submit_domain "smote_plain" "target_only" "dtw" "out_domain" "42"  "0.1"
submit_domain "smote_plain" "target_only" "dtw" "out_domain" "42"  "0.5"
submit_domain "smote_plain" "target_only" "dtw" "out_domain" "123" "0.1"
submit_domain "smote_plain" "target_only" "dtw" "out_domain" "123" "0.5"

# mmd - 6 missing
submit_domain "smote_plain" "target_only" "mmd" "in_domain"  "42"  "0.1"
submit_domain "smote_plain" "target_only" "mmd" "in_domain"  "42"  "0.5"
submit_domain "smote_plain" "target_only" "mmd" "in_domain"  "123" "0.1"
submit_domain "smote_plain" "target_only" "mmd" "in_domain"  "123" "0.5"
submit_domain "smote_plain" "target_only" "mmd" "out_domain" "42"  "0.5"
submit_domain "smote_plain" "target_only" "mmd" "out_domain" "123" "0.5"

# wasserstein - 13 missing (minus 1 in queue: was/out/source/123/0.5)
submit_domain "smote_plain" "source_only" "wasserstein" "in_domain"  "42"  "0.1"
submit_domain "smote_plain" "source_only" "wasserstein" "in_domain"  "42"  "0.5"
submit_domain "smote_plain" "source_only" "wasserstein" "in_domain"  "123" "0.1"
submit_domain "smote_plain" "source_only" "wasserstein" "in_domain"  "123" "0.5"
submit_domain "smote_plain" "target_only" "wasserstein" "in_domain"  "42"  "0.1"
submit_domain "smote_plain" "target_only" "wasserstein" "in_domain"  "42"  "0.5"
submit_domain "smote_plain" "target_only" "wasserstein" "in_domain"  "123" "0.1"
submit_domain "smote_plain" "target_only" "wasserstein" "in_domain"  "123" "0.5"
submit_domain "smote_plain" "source_only" "wasserstein" "out_domain" "42"  "0.1"
submit_domain "smote_plain" "source_only" "wasserstein" "out_domain" "42"  "0.5"
# wasserstein/out_domain/source_only/123/0.5 → in queue, skip
submit_domain "smote_plain" "target_only" "wasserstein" "out_domain" "42"  "0.5"
submit_domain "smote_plain" "target_only" "wasserstein" "out_domain" "123" "0.1"
submit_domain "smote_plain" "target_only" "wasserstein" "out_domain" "123" "0.5"

# ============================================================
# undersample: 17 missing jobs
# ============================================================
echo ""
echo "--- undersample (17 jobs) ---"

# dtw - 5 missing
submit_domain "undersample" "target_only" "dtw" "in_domain"  "123" "0.1"
submit_domain "undersample" "target_only" "dtw" "in_domain"  "123" "0.5"
submit_domain "undersample" "source_only" "dtw" "out_domain" "42"  "0.1"
submit_domain "undersample" "source_only" "dtw" "out_domain" "42"  "0.5"
submit_domain "undersample" "source_only" "dtw" "out_domain" "123" "0.1"

# mmd - 12 missing
submit_domain "undersample" "source_only" "mmd" "in_domain"  "42"  "0.1"
submit_domain "undersample" "source_only" "mmd" "in_domain"  "42"  "0.5"
submit_domain "undersample" "source_only" "mmd" "in_domain"  "123" "0.1"
submit_domain "undersample" "source_only" "mmd" "in_domain"  "123" "0.5"
submit_domain "undersample" "target_only" "mmd" "in_domain"  "42"  "0.1"
submit_domain "undersample" "target_only" "mmd" "in_domain"  "42"  "0.5"
submit_domain "undersample" "target_only" "mmd" "in_domain"  "123" "0.1"
submit_domain "undersample" "target_only" "mmd" "in_domain"  "123" "0.5"
submit_domain "undersample" "source_only" "mmd" "out_domain" "42"  "0.1"
submit_domain "undersample" "source_only" "mmd" "out_domain" "42"  "0.5"
submit_domain "undersample" "source_only" "mmd" "out_domain" "123" "0.1"
submit_domain "undersample" "source_only" "mmd" "out_domain" "123" "0.5"

# ============================================================
# smote (sw_smote): 12 missing jobs (all ratio=0.5)
# ============================================================
echo ""
echo "--- smote/sw_smote (12 jobs) ---"

# dtw - 4 missing (minus 1 in queue: dtw/out/source/0.5/42)
submit_domain "smote" "source_only" "dtw" "in_domain"  "42"  "0.5"
submit_domain "smote" "target_only" "dtw" "in_domain"  "42"  "0.5"
submit_domain "smote" "source_only" "dtw" "out_domain" "123" "0.5"
submit_domain "smote" "target_only" "dtw" "out_domain" "123" "0.5"

# mmd - 2 missing
submit_domain "smote" "target_only" "mmd" "in_domain"  "42"  "0.5"
submit_domain "smote" "source_only" "mmd" "out_domain" "123" "0.5"

# wasserstein - 6 missing
submit_domain "smote" "source_only" "wasserstein" "in_domain"  "42"  "0.5"
submit_domain "smote" "source_only" "wasserstein" "in_domain"  "123" "0.5"
submit_domain "smote" "source_only" "wasserstein" "out_domain" "42"  "0.5"
submit_domain "smote" "source_only" "wasserstein" "out_domain" "123" "0.5"
submit_domain "smote" "target_only" "wasserstein" "out_domain" "42"  "0.5"
submit_domain "smote" "target_only" "wasserstein" "out_domain" "123" "0.5"

# ============================================================
# Summary
# ============================================================
{
    echo ""
    echo "# Completed at $(date)"
    echo "# Submitted: $JOB_COUNT"
    echo "# Failed: $FAIL_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "  DRY RUN: would submit $JOB_COUNT jobs"
else
    echo "  Submitted: $JOB_COUNT jobs"
    echo "  Failed: $FAIL_COUNT"
fi
echo "  Log: $LOG_FILE"
echo "============================================================"

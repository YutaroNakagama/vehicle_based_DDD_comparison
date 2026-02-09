#!/bin/bash
# ============================================================
# Unified re-run launcher for split2 domain experiments (v3)
# ============================================================
# Fixes applied in commit 87bc95b:
#   Bug #1: eval_pipeline missing --target_file → random split (all modes)
#   Bug #2: source_group subjects read from Dir B instead of Dir A
#   Bug #3: eval_pipeline didn't handle "mixed" mode
#
# Re-run strategy:
#   Phase 1: target_only  → re-eval only   (training was correct)
#   Phase 2: source_only  → re-train + eval (training used wrong Dir B)
#   Phase 3: mixed        → re-train + eval (training used wrong Dir B)
#
# Queue limits (KAGAYAKI):
#   SINGLE: max_queued=40/user, max_run=10/user
#   LONG:   max_queued=15/user, max_run=2/user
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
REEVAL_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_reeval_split2.sh"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")

# Per-queue per-user limits (leave 1 slot buffer)
SINGLE_LIMIT=39   # actual max_queued=40
LONG_LIMIT=14     # actual max_queued=15

WAIT_INTERVAL=60   # seconds between queue checks

DRY_RUN=false
PHASE=""  # empty = all phases; "1","2","3" for specific phase

# Workaround: /var/tmp and /tmp may be full on head node
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --phase)    PHASE="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/rerun_all_split2_${TIMESTAMP}.log"

# ============================================================
# Helper Functions
# ============================================================

# Count jobs in a specific queue for this user (Q + R + H)
get_queue_count() {
    local queue_name="$1"
    qstat -u s2240011 2>/dev/null | awk -v q="$queue_name" '/s2240011/ && $3==q{n++} END{print n+0}'
}

# Wait until the target queue has a slot available
wait_for_queue_slot() {
    local queue_name="$1"
    local limit
    if [[ "$queue_name" == "LONG" ]]; then
        limit=$LONG_LIMIT
    else
        limit=$SINGLE_LIMIT
    fi

    if $DRY_RUN; then return 0; fi

    local count
    count=$(get_queue_count "$queue_name")
    while (( count >= limit )); do
        echo "[WAIT] $queue_name at $count/$limit — waiting ${WAIT_INTERVAL}s... ($(date +%H:%M:%S))"
        sleep "$WAIT_INTERVAL"
        count=$(get_queue_count "$queue_name")
    done
}

submit_job() {
    local cmd="$1"
    local label="$2"
    local log_entry="$3"
    local queue_name="$4"
    local max_retries=120  # 120 × 60s = 2 hours max wait per job

    if $DRY_RUN; then
        echo "[DRY-RUN] $label"
        return 0
    fi

    local attempt=0
    while (( attempt < max_retries )); do
        wait_for_queue_slot "$queue_name"

        local job_id
        job_id=$(eval "$cmd" 2>&1)
        local rc=$?
        if (( rc == 0 )); then
            echo "[SUBMIT] $label → $job_id"
            echo "OK:$log_entry:$job_id" >> "$LOG_FILE"
            sleep 0.3
            return 0
        fi

        # Queue limit or temp space error — wait and retry
        if [[ "$job_id" == *"exceed"*"limit"* ]] || [[ "$job_id" == *"No space"* ]] || [[ "$job_id" == *"mkstemp"* ]]; then
            ((attempt++))
            if (( attempt % 10 == 1 )); then
                echo "[RETRY $attempt/$max_retries] $label — $job_id — waiting ${WAIT_INTERVAL}s..."
            fi
            sleep "$WAIT_INTERVAL"
        else
            # Unexpected error — log and fail
            echo "[ERROR] $label → $job_id (rc=$rc)"
            echo "FAIL:$log_entry:$job_id" >> "$LOG_FILE"
            return 1
        fi
    done

    echo "[GIVE UP] $label — max retries reached"
    echo "GIVEUP:$log_entry" >> "$LOG_FILE"
    return 1
}

get_train_resources() {
    local condition="$1"
    case "$condition" in
        balanced_rf)       echo "ncpus=8:mem=12gb 08:00:00 LONG" ;;
        smote|smote_plain) echo "ncpus=4:mem=10gb 08:00:00 SINGLE" ;;
        *)                 echo "ncpus=4:mem=8gb 06:00:00 SINGLE" ;;
    esac
}

TOTAL_SUBMITTED=0
TOTAL_FAILED=0

# ============================================================
# Phase 1: target_only → evaluation only (all SINGLE queue)
# ============================================================
run_phase1() {
    echo ""
    echo "============================================================"
    echo "  Phase 1: target_only RE-EVAL (training correct, eval fix)"
    echo "============================================================"
    local count=0

    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # Baseline
                local jname="re_bs_${DIST:0:2}${DOM:0:1}_t_s${SEED}"
                local cmd="qsub -N $jname -l select=1:ncpus=2:mem=8gb -l walltime=00:30:00 -q SINGLE"
                cmd="$cmd -v CONDITION=baseline,MODE=target_only,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,RANKING=$RANKING"
                cmd="$cmd $REEVAL_SCRIPT"
                if submit_job "$cmd" "baseline | $DIST | $DOM | target_only | s$SEED" "reeval:baseline:$DIST:$DOM:target_only:$SEED" "SINGLE"; then
                    ((count++))
                else
                    ((TOTAL_FAILED++))
                fi

                # Ratio-based
                for RATIO in "${RATIOS[@]}"; do
                    for COND in smote_plain smote undersample; do
                        local rr="${RATIO/0./}"
                        jname="re_${COND:0:2}_${DIST:0:2}${DOM:0:1}_r${rr}_s${SEED}"
                        cmd="qsub -N $jname -l select=1:ncpus=2:mem=8gb -l walltime=00:30:00 -q SINGLE"
                        cmd="$cmd -v CONDITION=$COND,MODE=target_only,DISTANCE=$DIST,DOMAIN=$DOM,RATIO=$RATIO,SEED=$SEED,RANKING=$RANKING"
                        cmd="$cmd $REEVAL_SCRIPT"
                        if submit_job "$cmd" "$COND | $DIST | $DOM | target_only | r=$RATIO | s$SEED" "reeval:$COND:$DIST:$DOM:target_only:$RATIO:$SEED" "SINGLE"; then
                            ((count++))
                        else
                            ((TOTAL_FAILED++))
                        fi
                    done
                done

                # Balanced RF
                jname="re_bf_${DIST:0:2}${DOM:0:1}_t_s${SEED}"
                cmd="qsub -N $jname -l select=1:ncpus=2:mem=8gb -l walltime=00:30:00 -q SINGLE"
                cmd="$cmd -v CONDITION=balanced_rf,MODE=target_only,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,RANKING=$RANKING"
                cmd="$cmd $REEVAL_SCRIPT"
                if submit_job "$cmd" "balanced_rf | $DIST | $DOM | target_only | s$SEED" "reeval:balanced_rf:$DIST:$DOM:target_only:$SEED" "SINGLE"; then
                    ((count++))
                else
                    ((TOTAL_FAILED++))
                fi
            done
        done
    done

    echo "  Phase 1 complete: $count jobs submitted"
    TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + count))
}

# ============================================================
# Phase 2/3: source_only or mixed → full train + eval
# ============================================================
run_retrain_phase() {
    local mode="$1"
    local phase_num="$2"
    local prefix
    [[ "$mode" == "source_only" ]] && prefix="fs" || prefix="fm"

    echo ""
    echo "============================================================"
    echo "  Phase $phase_num: $mode RE-TRAIN + EVAL (Dir B → Dir A fix)"
    echo "============================================================"
    local count=0

    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # Baseline
                local res=$(get_train_resources "baseline")
                local ncpus_mem=$(echo "$res" | cut -d' ' -f1)
                local walltime=$(echo "$res" | cut -d' ' -f2)
                local queue=$(echo "$res" | cut -d' ' -f3)
                local jname="${prefix}_bs_${DIST:0:2}${DOM:0:1}_s${SEED}"

                local cmd="qsub -N $jname -l select=1:$ncpus_mem -l walltime=$walltime -q $queue"
                cmd="$cmd -v CONDITION=baseline,MODE=$mode,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                cmd="$cmd $TRAIN_SCRIPT"
                if submit_job "$cmd" "baseline | $DIST | $DOM | $mode | s$SEED" "train:baseline:$DIST:$DOM:$mode:$SEED" "$queue"; then
                    ((count++))
                else
                    ((TOTAL_FAILED++))
                fi

                # Ratio-based
                for RATIO in "${RATIOS[@]}"; do
                    for COND in smote_plain smote undersample; do
                        res=$(get_train_resources "$COND")
                        ncpus_mem=$(echo "$res" | cut -d' ' -f1)
                        walltime=$(echo "$res" | cut -d' ' -f2)
                        queue=$(echo "$res" | cut -d' ' -f3)
                        local rr="${RATIO/0./}"
                        jname="${prefix}_${COND:0:2}_${DIST:0:2}${DOM:0:1}_r${rr}_s${SEED}"

                        cmd="qsub -N $jname -l select=1:$ncpus_mem -l walltime=$walltime -q $queue"
                        cmd="$cmd -v CONDITION=$COND,MODE=$mode,DISTANCE=$DIST,DOMAIN=$DOM,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                        cmd="$cmd $TRAIN_SCRIPT"
                        if submit_job "$cmd" "$COND | $DIST | $DOM | $mode | r=$RATIO | s$SEED" "train:$COND:$DIST:$DOM:$mode:$RATIO:$SEED" "$queue"; then
                            ((count++))
                        else
                            ((TOTAL_FAILED++))
                        fi
                    done
                done

                # Balanced RF
                res=$(get_train_resources "balanced_rf")
                ncpus_mem=$(echo "$res" | cut -d' ' -f1)
                walltime=$(echo "$res" | cut -d' ' -f2)
                queue=$(echo "$res" | cut -d' ' -f3)
                jname="${prefix}_bf_${DIST:0:2}${DOM:0:1}_s${SEED}"

                cmd="qsub -N $jname -l select=1:$ncpus_mem -l walltime=$walltime -q $queue"
                cmd="$cmd -v CONDITION=balanced_rf,MODE=$mode,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                cmd="$cmd $TRAIN_SCRIPT"
                if submit_job "$cmd" "balanced_rf | $DIST | $DOM | $mode | s$SEED" "train:balanced_rf:$DIST:$DOM:$mode:$SEED" "$queue"; then
                    ((count++))
                else
                    ((TOTAL_FAILED++))
                fi
            done
        done
    done

    echo "  Phase $phase_num complete: $count jobs submitted"
    TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + count))
}

# ============================================================
# Main
# ============================================================
echo "============================================================"
echo "  Split2 Bug-Fix Re-run Launcher (v3)"
echo "============================================================"
echo "  Dry run : $DRY_RUN"
echo "  Phase   : ${PHASE:-all}"
echo "  Limits  : SINGLE=$SINGLE_LIMIT, LONG=$LONG_LIMIT"
echo "  Poll    : ${WAIT_INTERVAL}s"
echo "  TMPDIR  : $TMPDIR"
echo "  Log     : $LOG_FILE"
echo "============================================================"

echo "  SINGLE queue: $(get_queue_count SINGLE) jobs"
echo "  LONG queue  : $(get_queue_count LONG) jobs"
echo "============================================================"
echo ""

if [[ -z "$PHASE" || "$PHASE" == "1" ]]; then
    run_phase1
fi

if [[ -z "$PHASE" || "$PHASE" == "2" ]]; then
    run_retrain_phase "source_only" 2
fi

if [[ -z "$PHASE" || "$PHASE" == "3" ]]; then
    run_retrain_phase "mixed" 3
fi

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Submitted: $TOTAL_SUBMITTED"
echo "  Failed   : $TOTAL_FAILED"
echo "  Log      : $LOG_FILE"
echo "============================================================"

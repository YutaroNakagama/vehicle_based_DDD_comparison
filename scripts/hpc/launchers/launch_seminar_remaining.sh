#!/bin/bash
# ============================================================
# Submit remaining Phase 2 + all Phase 3 via SEMINAR queue
# ============================================================
# SEMINAR queue: no per-user max_queued limit, priority=100
# Walltime max: 84h (our jobs need 6-8h)
# ============================================================
set -uo pipefail

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
QUEUE="SEMINAR"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/seminar_submit_${TIMESTAMP}.log"

# Already-submitted Phase 2 jobs (from previous launcher log)
DONE_FILE="/tmp/phase2_done.txt"

is_already_done() {
    local key="$1"
    grep -qF "$key" "$DONE_FILE" 2>/dev/null
}

get_resources() {
    local condition="$1"
    case "$condition" in
        balanced_rf)       echo "ncpus=8:mem=12gb 08:00:00" ;;
        smote|smote_plain) echo "ncpus=4:mem=10gb 08:00:00" ;;
        *)                 echo "ncpus=4:mem=8gb 06:00:00" ;;
    esac
}

echo "============================================================"
echo "  SEMINAR Queue Submission — Phase 2 (remaining) + Phase 3"
echo "============================================================"
echo "  Dry run: $DRY_RUN"
echo "  Queue  : $QUEUE"
echo "  Log    : $LOG_FILE"
echo "============================================================"

SUBMITTED=0
SKIPPED=0
FAILED=0

submit_one() {
    local mode="$1" cond="$2" dist="$3" dom="$4" ratio="$5" seed="$6"
    local prefix
    [[ "$mode" == "source_only" ]] && prefix="fs" || prefix="fm"

    local res=$(get_resources "$cond")
    local ncpus_mem=$(echo "$res" | cut -d' ' -f1)
    local walltime=$(echo "$res" | cut -d' ' -f2)

    local rr="${ratio/0./}"
    local jname
    if [[ "$ratio" == "NA" ]]; then
        jname="${prefix}_${cond:0:2}_${dist:0:2}${dom:0:1}_s${seed}"
    else
        jname="${prefix}_${cond:0:2}_${dist:0:2}${dom:0:1}_r${rr}_s${seed}"
    fi

    local cmd="qsub -N $jname -l select=1:$ncpus_mem -l walltime=$walltime -q $QUEUE"
    if [[ "$ratio" == "NA" ]]; then
        cmd="$cmd -v CONDITION=$cond,MODE=$mode,DISTANCE=$dist,DOMAIN=$dom,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    else
        cmd="$cmd -v CONDITION=$cond,MODE=$mode,DISTANCE=$dist,DOMAIN=$dom,RATIO=$ratio,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    fi
    cmd="$cmd $TRAIN_SCRIPT"

    local label="$cond | $dist | $dom | $mode | r=$ratio | s$seed"

    if $DRY_RUN; then
        echo "[DRY-RUN] $label"
        ((SUBMITTED++))
        return
    fi

    local job_id
    job_id=$(eval "$cmd" 2>&1)
    local rc=$?
    if (( rc == 0 )); then
        echo "[SUBMIT] $label → $job_id"
        echo "OK:$mode:$cond:$dist:$dom:$ratio:$seed:$job_id" >> "$LOG_FILE"
        ((SUBMITTED++))
        sleep 0.2
    else
        echo "[ERROR] $label → $job_id (rc=$rc)"
        echo "FAIL:$mode:$cond:$dist:$dom:$ratio:$seed:$job_id" >> "$LOG_FILE"
        ((FAILED++))
    fi
}

# ============================================================
# Phase 2 remaining: source_only
# ============================================================
echo ""
echo "--- Phase 2: source_only (remaining) ---"
P2_COUNT=0

for DIST in "${DISTANCES[@]}"; do
    for DOM in "${DOMAINS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            # Baseline
            KEY="baseline|${DIST}|${DOM}|NA|${SEED}"
            if is_already_done "$KEY"; then
                ((SKIPPED++))
            else
                submit_one source_only baseline "$DIST" "$DOM" NA "$SEED"
                ((P2_COUNT++))
            fi

            # Ratio-based
            for RATIO in "${RATIOS[@]}"; do
                for COND in smote_plain smote undersample; do
                    KEY="${COND}|${DIST}|${DOM}|${RATIO}|${SEED}"
                    if is_already_done "$KEY"; then
                        ((SKIPPED++))
                    else
                        submit_one source_only "$COND" "$DIST" "$DOM" "$RATIO" "$SEED"
                        ((P2_COUNT++))
                    fi
                done
            done

            # Balanced RF
            KEY="balanced_rf|${DIST}|${DOM}|NA|${SEED}"
            if is_already_done "$KEY"; then
                ((SKIPPED++))
            else
                submit_one source_only balanced_rf "$DIST" "$DOM" NA "$SEED"
                ((P2_COUNT++))
            fi
        done
    done
done
echo "  Phase 2 new: $P2_COUNT, skipped: $SKIPPED"

# ============================================================
# Phase 3: mixed (all 96)
# ============================================================
echo ""
echo "--- Phase 3: mixed (all 96) ---"
P3_COUNT=0

for DIST in "${DISTANCES[@]}"; do
    for DOM in "${DOMAINS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            submit_one mixed baseline "$DIST" "$DOM" NA "$SEED"
            ((P3_COUNT++))

            for RATIO in "${RATIOS[@]}"; do
                for COND in smote_plain smote undersample; do
                    submit_one mixed "$COND" "$DIST" "$DOM" "$RATIO" "$SEED"
                    ((P3_COUNT++))
                done
            done

            submit_one mixed balanced_rf "$DIST" "$DOM" NA "$SEED"
            ((P3_COUNT++))
        done
    done
done
echo "  Phase 3 new: $P3_COUNT"

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Submitted: $SUBMITTED"
echo "  Skipped : $SKIPPED (already in queue)"
echo "  Failed  : $FAILED"
echo "  Log     : $LOG_FILE"
echo "============================================================"

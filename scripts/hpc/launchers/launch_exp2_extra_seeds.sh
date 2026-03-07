#!/bin/bash
# ============================================================
# Experiment 2: Additional Seeds Launcher
# ============================================================
# Add 3 extra seeds (0, 7, 2024) to the existing 2-seed (42, 123)
# Exp2 split2 domain experiments.
#
# All modes need full train + eval (no reeval-only phase needed
# since these are entirely new seeds).
#
# Conditions:
#   baseline      (no ratio)   → 3 modes × 3 dists × 2 doms × 3 seeds =  54
#   smote_plain   (r0.1, r0.5) → 3 × 3 × 2 × 2 × 3                   = 108
#   smote         (r0.1, r0.5) → 3 × 3 × 2 × 2 × 3                   = 108
#   undersample   (r0.1, r0.5) → 3 × 3 × 2 × 2 × 3                   = 108
#   balanced_rf   (no ratio)   → 3 × 3 × 2 × 3                        =  54
#   Total: 432 jobs
#
# Queue limits (KAGAYAKI):
#   SINGLE: max_queued=40/user, max_run=10/user
#   DEFAULT: max_queued=10/user, max_run=5/user
#   SMALL: max_queued=10/user, max_run=5/user
#   LONG:  max_queued=15/user, max_run=2/user
#
# Usage:
#   bash scripts/hpc/launchers/launch_exp2_extra_seeds.sh --dry-run
#   nohup bash scripts/hpc/launchers/launch_exp2_extra_seeds.sh &
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

# New seeds only (existing: 42, 123)
SEEDS=(0 7 2024)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODES=("target_only" "source_only" "mixed")

# Per-queue per-user limits (leave buffer)
SINGLE_LIMIT=38
DEFAULT_LIMIT=9
SMALL_LIMIT=9
LONG_LIMIT=14

WAIT_INTERVAL=60

DRY_RUN=false

# Workaround: /var/tmp and /tmp may be full on head node
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/exp2_extra_seeds_${TIMESTAMP}.log"

# ============================================================
# Helper Functions
# ============================================================
get_queue_count() {
    local queue_name="$1"
    qstat -u s2240011 2>/dev/null | awk -v q="$queue_name" '/s2240011/ && $3==q{n++} END{print n+0}'
}

get_queue_limit() {
    local queue_name="$1"
    case "$queue_name" in
        SINGLE)  echo $SINGLE_LIMIT ;;
        DEFAULT) echo $DEFAULT_LIMIT ;;
        SMALL)   echo $SMALL_LIMIT ;;
        LONG)    echo $LONG_LIMIT ;;
        *)       echo 5 ;;
    esac
}

wait_for_any_queue_slot() {
    # Try SINGLE first, then DEFAULT, then SMALL, then LONG
    # Returns the queue name with available slots
    if $DRY_RUN; then echo "SINGLE"; return 0; fi

    while true; do
        for q in SINGLE DEFAULT SMALL; do
            local count limit
            count=$(get_queue_count "$q")
            limit=$(get_queue_limit "$q")
            if (( count < limit )); then
                echo "$q"
                return 0
            fi
        done
        sleep "$WAIT_INTERVAL"
    done
}

wait_for_long_slot() {
    if $DRY_RUN; then return 0; fi
    local count
    count=$(get_queue_count "LONG")
    while (( count >= LONG_LIMIT )); do
        sleep "$WAIT_INTERVAL"
        count=$(get_queue_count "LONG")
    done
}

submit_job() {
    local cmd="$1"
    local label="$2"
    local log_entry="$3"
    local max_retries=120

    if $DRY_RUN; then
        echo "[DRY-RUN] $label"
        return 0
    fi

    local attempt=0
    while (( attempt < max_retries )); do
        local job_id
        job_id=$(eval "$cmd" 2>&1)
        local rc=$?
        if (( rc == 0 )); then
            echo "[SUBMIT] $label → $job_id"
            echo "OK:$log_entry:$job_id" >> "$LOG_FILE"
            sleep 0.3
            return 0
        fi

        if [[ "$job_id" == *"exceed"*"limit"* ]] || [[ "$job_id" == *"No space"* ]] || [[ "$job_id" == *"mkstemp"* ]]; then
            ((attempt++))
            if (( attempt % 10 == 1 )); then
                echo "[RETRY $attempt/$max_retries] $label — $job_id — waiting ${WAIT_INTERVAL}s..."
            fi
            sleep "$WAIT_INTERVAL"
        else
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
        balanced_rf)       echo "ncpus=8:mem=12gb 10:00:00 LONG" ;;
        smote|smote_plain) echo "ncpus=4:mem=10gb 10:00:00" ;;
        *)                 echo "ncpus=4:mem=8gb 10:00:00" ;;
    esac
}

TOTAL_SUBMITTED=0
TOTAL_FAILED=0

# ============================================================
# Main submission loop
# ============================================================
echo "============================================================"
echo "  Experiment 2: Extra Seeds Launcher"
echo "============================================================"
echo "  Seeds   : ${SEEDS[*]}"
echo "  Dry run : $DRY_RUN"
echo "  Log     : $LOG_FILE"
echo "============================================================"
echo "  Current queue load:"
for q in SINGLE DEFAULT SMALL LONG; do
    echo "    $q: $(get_queue_count $q) / $(get_queue_limit $q)"
done
echo "============================================================"
echo ""

for MODE in "${MODES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Mode: $MODE"
    echo "============================================================"

    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do

                # --- Baseline (no ratio) ---
                local_res=$(get_train_resources "baseline")
                ncpus_mem=$(echo "$local_res" | cut -d' ' -f1)
                walltime=$(echo "$local_res" | cut -d' ' -f2)

                queue=$(wait_for_any_queue_slot)
                jname="e2_bs_${MODE:0:1}${DIST:0:2}${DOM:0:1}_s${SEED}"

                cmd="qsub -N $jname -l select=1:$ncpus_mem -l walltime=$walltime -q $queue"
                cmd="$cmd -v CONDITION=baseline,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                cmd="$cmd $TRAIN_SCRIPT"
                if submit_job "$cmd" "baseline | $MODE | $DIST | $DOM | s$SEED" "train:baseline:$MODE:$DIST:$DOM:$SEED"; then
                    ((TOTAL_SUBMITTED++))
                else
                    ((TOTAL_FAILED++))
                fi

                # --- Ratio-based conditions ---
                for RATIO in "${RATIOS[@]}"; do
                    for COND in smote_plain smote undersample; do
                        local_res=$(get_train_resources "$COND")
                        ncpus_mem=$(echo "$local_res" | cut -d' ' -f1)
                        walltime=$(echo "$local_res" | cut -d' ' -f2)

                        queue=$(wait_for_any_queue_slot)
                        rr="${RATIO/0./}"
                        jname="e2_${COND:0:2}r${rr}_${MODE:0:1}${DIST:0:2}${DOM:0:1}_s${SEED}"

                        cmd="qsub -N $jname -l select=1:$ncpus_mem -l walltime=$walltime -q $queue"
                        cmd="$cmd -v CONDITION=$COND,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                        cmd="$cmd $TRAIN_SCRIPT"
                        if submit_job "$cmd" "$COND r$RATIO | $MODE | $DIST | $DOM | s$SEED" "train:$COND:$MODE:$DIST:$DOM:r$RATIO:$SEED"; then
                            ((TOTAL_SUBMITTED++))
                        else
                            ((TOTAL_FAILED++))
                        fi
                    done
                done

                # --- Balanced RF (no ratio, LONG queue) ---
                local_res=$(get_train_resources "balanced_rf")
                ncpus_mem=$(echo "$local_res" | cut -d' ' -f1)
                walltime=$(echo "$local_res" | cut -d' ' -f2)

                wait_for_long_slot
                jname="e2_bf_${MODE:0:1}${DIST:0:2}${DOM:0:1}_s${SEED}"

                cmd="qsub -N $jname -l select=1:$ncpus_mem -l walltime=$walltime -q LONG"
                cmd="$cmd -v CONDITION=balanced_rf,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                cmd="$cmd $TRAIN_SCRIPT"
                if submit_job "$cmd" "balanced_rf | $MODE | $DIST | $DOM | s$SEED" "train:balanced_rf:$MODE:$DIST:$DOM:$SEED"; then
                    ((TOTAL_SUBMITTED++))
                else
                    ((TOTAL_FAILED++))
                fi

            done
        done
    done
done

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Submitted: $TOTAL_SUBMITTED"
echo "  Failed   : $TOTAL_FAILED"
echo "  Log      : $LOG_FILE"
echo "============================================================"

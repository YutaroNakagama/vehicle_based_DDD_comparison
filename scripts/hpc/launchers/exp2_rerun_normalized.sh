#!/bin/bash
# ============================================================
# Exp2 FULL Re-run Daemon — Max Parallelism (All 4 Queues)
# ============================================================
# Re-submits all 1,296 Exp2 jobs after normalized grouping fix.
# Uses all 4 CPU queues (SINGLE/DEFAULT/SMALL/LONG) for max throughput.
#
# Job count:
#   10-seed (baseline/smote_plain/smote/undersample):
#     3 dist × 2 dom × 3 mode × 10 seed × 7 cond = 1,260
#   2-seed (balanced_rf):
#     3 dist × 2 dom × 3 mode × 2 seed × 1 cond  = 36
#   Total: 1,296 jobs
#
# Queue limits (from qmgr):
#   SINGLE:  max_queued=40
#   DEFAULT: max_queued=40
#   SMALL:   max_queued=30
#   LONG:    max_queued=15
#   Total:   125 concurrent slots
#
# Usage:
#   # Dry run first:
#   bash scripts/hpc/launchers/exp2_rerun_normalized.sh --dry-run
#
#   # Actual submission (run in background):
#   nohup bash scripts/hpc/launchers/exp2_rerun_normalized.sh \
#     > scripts/hpc/logs/domain/exp2_rerun_output.log 2>&1 &
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

# === Seeds ===
SEEDS_RF=(0 1 7 13 42 123 256 512 1337 2024)
SEEDS_BRF=(42 123)

# === Experiment params ===
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODES=("source_only" "target_only" "mixed")

# === Queue config ===
declare -A Q_MAX=([SINGLE]=40 [DEFAULT]=40 [SMALL]=30 [LONG]=15)
QUEUES=("SINGLE" "DEFAULT" "SMALL" "LONG")
POLL_INTERVAL=90  # seconds between queue checks

# === Resource profiles ===
get_resources() {
    local cond="$1"
    case "$cond" in
        balanced_rf) echo "ncpus=8:mem=12gb 20:00:00" ;;
        smote|smote_plain) echo "ncpus=4:mem=10gb 20:00:00" ;;
        *) echo "ncpus=4:mem=8gb 10:00:00" ;;
    esac
}

# === Parse args ===
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# === Logging ===
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/exp2_rerun_normalized_${TIMESTAMP}.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# ============================================================
# Build complete job list
# ============================================================
declare -a ALL_JOBS=()

for MODE in "${MODES[@]}"; do
    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            # RF conditions (10 seeds)
            for SEED in "${SEEDS_RF[@]}"; do
                # Baseline (no ratio)
                ALL_JOBS+=("baseline||$MODE|$DIST|$DOM|$SEED")
                # Ratio conditions
                for RATIO in "${RATIOS[@]}"; do
                    for COND in "smote_plain" "smote" "undersample"; do
                        ALL_JOBS+=("$COND|$RATIO|$MODE|$DIST|$DOM|$SEED")
                    done
                done
            done
            # BalancedRF (2 seeds)
            for SEED in "${SEEDS_BRF[@]}"; do
                ALL_JOBS+=("balanced_rf||$MODE|$DIST|$DOM|$SEED")
            done
        done
    done
done

TOTAL_JOBS=${#ALL_JOBS[@]}
log "============================================================"
log "  Exp2 FULL Re-run (Normalized Grouping) — $(date)"
log "  Total jobs: $TOTAL_JOBS"
log "  Queue limits: SINGLE=${Q_MAX[SINGLE]} DEFAULT=${Q_MAX[DEFAULT]} SMALL=${Q_MAX[SMALL]} LONG=${Q_MAX[LONG]}"
log "  Max concurrent: $((Q_MAX[SINGLE]+Q_MAX[DEFAULT]+Q_MAX[SMALL]+Q_MAX[LONG]))"
log "  DRY_RUN: $DRY_RUN"
log "============================================================"

# ============================================================
# Track submitted jobs (keyed by spec)
# ============================================================
declare -A SUBMITTED=()

# ============================================================
# Queue helpers
# ============================================================
get_queue_count() {
    qstat -u s2240011 2>/dev/null | awk -v q="$1" '/s2240011/ && $3==q{n++} END{print n+0}'
}

# ============================================================
# Submit one job
# ============================================================
submit_job() {
    local cond="$1" ratio="$2" mode="$3" dist="$4" dom="$5" seed="$6" queue="$7"

    local res
    res=$(get_resources "$cond")
    local ncpus_mem walltime
    ncpus_mem=$(echo "$res" | cut -d' ' -f1)
    walltime=$(echo "$res" | cut -d' ' -f2)

    # Short job name
    local cond_short="${cond:0:2}"
    local job_name
    if [[ -n "$ratio" ]]; then
        job_name="${cond_short}_${dist:0:1}${dom:0:1}_${mode:0:1}_r${ratio}_s${seed}"
    else
        job_name="${cond_short}_${dist:0:1}${dom:0:1}_${mode:0:1}_s${seed}"
    fi

    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue"
    cmd="$cmd -v CONDITION=$cond,MODE=$mode,DISTANCE=$dist,DOMAIN=$dom,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    if [[ -n "$ratio" ]]; then
        cmd="$cmd,RATIO=$ratio"
    fi
    cmd="$cmd $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "  [DRY] $queue | $cond | $dist | $dom | $mode | r=$ratio | s=$seed"
        return 0
    fi

    local jobid
    jobid=$(eval "$cmd" 2>&1) || { log "ERROR: $cmd"; return 1; }
    log "OK:$cond|$ratio|$mode|$dist|$dom|$seed|$queue|$jobid"
    return 0
}

# ============================================================
# Main daemon loop
# ============================================================
SUBMITTED_COUNT=0
JOB_INDEX=0

if $DRY_RUN; then
    log "=== DRY RUN: listing all $TOTAL_JOBS jobs ==="
    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r cond ratio mode dist dom seed <<< "$job_spec"
        submit_job "$cond" "$ratio" "$mode" "$dist" "$dom" "$seed" "SINGLE"
        ((SUBMITTED_COUNT++))
    done
    log "=== DRY RUN complete: $SUBMITTED_COUNT jobs would be submitted ==="
    exit 0
fi

# Verify job script exists
if [[ ! -f "$JOB_SCRIPT" ]]; then
    log "ERROR: Job script not found: $JOB_SCRIPT"
    exit 1
fi

# Verify normalized group files exist
for dist in "${DISTANCES[@]}"; do
    for dom in "${DOMAINS[@]}"; do
        target="results/analysis/exp2_domain_shift/distance/rankings/split2/knn/${dist}_${dom}.txt"
        if [[ ! -f "$target" ]]; then
            log "ERROR: Missing group file: $target"
            log "Run regenerate_split2_normalized.py first!"
            exit 1
        fi
    done
done
log "All group files verified."

# Main submit loop
while [[ $JOB_INDEX -lt $TOTAL_JOBS ]]; do
    # Calculate available slots per queue
    TOTAL_AVAILABLE=0
    declare -A Q_AVAIL
    for q in "${QUEUES[@]}"; do
        cur=$(get_queue_count "$q")
        avail=$(( ${Q_MAX[$q]} - cur ))
        [[ $avail -lt 0 ]] && avail=0
        Q_AVAIL[$q]=$avail
        TOTAL_AVAILABLE=$((TOTAL_AVAILABLE + avail))
    done

    if [[ $TOTAL_AVAILABLE -le 0 ]]; then
        remaining=$((TOTAL_JOBS - JOB_INDEX))
        log "All queues full. $remaining remaining. Sleeping ${POLL_INTERVAL}s..."
        sleep "$POLL_INTERVAL"
        continue
    fi

    log "Slots available: SINGLE=${Q_AVAIL[SINGLE]} DEFAULT=${Q_AVAIL[DEFAULT]} SMALL=${Q_AVAIL[SMALL]} LONG=${Q_AVAIL[LONG]} (total=$TOTAL_AVAILABLE)"

    # Submit jobs round-robin across queues with available slots
    BATCH_SUBMITTED=0
    while [[ $JOB_INDEX -lt $TOTAL_JOBS ]] && [[ $TOTAL_AVAILABLE -gt 0 ]]; do
        # Find a queue with slots (round-robin)
        local_queue=""
        for _ in "${QUEUES[@]}"; do
            for q in "${QUEUES[@]}"; do
                if [[ ${Q_AVAIL[$q]} -gt 0 ]]; then
                    local_queue="$q"
                    break
                fi
            done
            [[ -n "$local_queue" ]] && break
        done
        [[ -z "$local_queue" ]] && break

        # Get next job
        job_spec="${ALL_JOBS[$JOB_INDEX]}"
        IFS='|' read -r cond ratio mode dist dom seed <<< "$job_spec"

        # Choose queue: balanced_rf → LONG preferred, else round-robin
        if [[ "$cond" == "balanced_rf" ]] && [[ ${Q_AVAIL[LONG]} -gt 0 ]]; then
            local_queue="LONG"
        fi

        if submit_job "$cond" "$ratio" "$mode" "$dist" "$dom" "$seed" "$local_queue"; then
            SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
            BATCH_SUBMITTED=$((BATCH_SUBMITTED + 1))
            Q_AVAIL[$local_queue]=$(( ${Q_AVAIL[$local_queue]} - 1 ))
            TOTAL_AVAILABLE=$((TOTAL_AVAILABLE - 1))
        fi

        JOB_INDEX=$((JOB_INDEX + 1))
        sleep 0.15  # Avoid overwhelming PBS
    done

    log "Batch submitted: $BATCH_SUBMITTED | Total: $SUBMITTED_COUNT/$TOTAL_JOBS | Index: $JOB_INDEX"

    # If more jobs remain, wait for slots
    if [[ $JOB_INDEX -lt $TOTAL_JOBS ]]; then
        sleep "$POLL_INTERVAL"
    fi
done

log "============================================================"
log "  ALL $SUBMITTED_COUNT jobs submitted."
log "  Log: $LOG_FILE"
log "============================================================"

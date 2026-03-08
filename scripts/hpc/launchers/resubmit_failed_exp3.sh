#!/bin/bash
# ============================================================
# Failed 254 jobs resubmit launcher (2026-02-07)
# ============================================================
# Cause: results/analysis/domain/ → exp2_domain_shift/ after rename
#       Old path not found, 'Target file not found' failure
# Fix: PBS script updated to exp2_domain_shift path
#
# Queue distribution strategy:
#   DEFAULT  : max_queued=40/user → current17 → 23slots
#   SINGLE   : max_queued=40/user → current20 → 20slots
#   SMALL    : max_queued=30/user → current 3 → 27slots
#   LONG     : max_queued=15/user → current15 →  0slots (full)
#   SEMINAR  : no limit          → all remaining
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
PARAMS_FILE="/tmp/failed_exp3_params.txt"

if [[ ! -f "$PARAMS_FILE" ]]; then
    echo "[ERROR] Params file not found: $PARAMS_FILE"
    exit 1
fi

TOTAL_PARAMS=$(wc -l < "$PARAMS_FILE")
N_TRIALS=100
RANKING="knn"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/resubmit_exp3_${TIMESTAMP}.log"

# Queue allocation: fill available slots, rest to SEMINAR
# Order: SMALL(27), DEFAULT(23), SINGLE(20), SEMINAR(rest)
QUEUES_ORDERED=(SMALL DEFAULT SINGLE SEMINAR)
declare -A Q_SLOTS
Q_SLOTS[SMALL]=27
Q_SLOTS[DEFAULT]=23
Q_SLOTS[SINGLE]=20
Q_SLOTS[SEMINAR]=999999  # unlimited

declare -A Q_COUNT
for q in "${QUEUES_ORDERED[@]}"; do Q_COUNT[$q]=0; done

# Resource settings per model
get_resources() {
    local model="$1"
    case "$model" in
        SvmW) echo "ncpus=4:mem=8gb 08:00:00" ;;
        SvmA) echo "ncpus=4:mem=8gb 10:00:00" ;;
        Lstm) echo "ncpus=4:mem=16gb 12:00:00" ;;
    esac
}

# Pick next available queue
pick_queue() {
    for q in "${QUEUES_ORDERED[@]}"; do
        if [[ ${Q_COUNT[$q]} -lt ${Q_SLOTS[$q]} ]]; then
            echo "$q"
            return
        fi
    done
    echo "SEMINAR"  # fallback
}

JOB_COUNT=0
FAIL_COUNT=0

echo "============================================================"
echo "  Exp3 failed job resubmit (target file not found fix)"
echo "  $(date)"
echo "============================================================"
echo "  Dry run : $DRY_RUN"
echo "  Params  : $TOTAL_PARAMS"
echo "  Queues  : ${QUEUES_ORDERED[*]}"
echo "  Slots   : SMALL=${Q_SLOTS[SMALL]} DEFAULT=${Q_SLOTS[DEFAULT]} SINGLE=${Q_SLOTS[SINGLE]} SEMINAR=unlimited"
echo ""

{
    echo "# Exp3 resubmission: $(date)"
    echo "# Params file: $PARAMS_FILE"
    echo "# Total: $TOTAL_PARAMS"
    echo ""
} > "$LOG_FILE"

while IFS=',' read -r MODEL CONDITION MODE DISTANCE DOMAIN RATIO SEED; do
    [[ -z "$MODEL" ]] && continue

    local_queue=$(pick_queue)

    res=$(get_resources "$MODEL")
    ncpus_mem="${res% *}"
    walltime="${res#* }"

    # Short job name
    m_short="${MODEL:0:2}"
    c_short="${CONDITION:0:2}"
    dist_short="${DISTANCE:0:1}"
    dom_short="${DOMAIN:0:1}"
    mode_short="${MODE:0:1}"
    r_short=$(echo "$RATIO" | tr -d '.')
    job_name="r${m_short}_${c_short}_${dist_short}${dom_short}_${mode_short}_r${r_short}_s${SEED}"

    env_vars="MODEL=$MODEL,CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,RATIO=$RATIO,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"

    cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $local_queue -v $env_vars $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] [$local_queue] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s$SEED"
        ((JOB_COUNT++))
        ((Q_COUNT[$local_queue]++))
    else
        if job_id=$(eval "$cmd" 2>&1); then
            echo "[OK] [$local_queue] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s$SEED → $job_id"
            echo "$local_queue:$MODEL:$CONDITION:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED:$job_id" >> "$LOG_FILE"
            ((JOB_COUNT++))
            ((Q_COUNT[$local_queue]++))
        else
            echo "[FAIL] [$local_queue] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s$SEED — $job_id"
            ((FAIL_COUNT++))
        fi
        sleep 0.05
    fi
done < "$PARAMS_FILE"

echo ""
echo "============================================================"
echo "  Results"
echo "============================================================"
echo "  Submitted: $JOB_COUNT / $TOTAL_PARAMS"
echo "  Failed:    $FAIL_COUNT"
echo ""
echo "  Per-queue:"
for q in "${QUEUES_ORDERED[@]}"; do
    echo "    $q: ${Q_COUNT[$q]}"
done
echo ""
echo "  Log: $LOG_FILE"
echo "============================================================"

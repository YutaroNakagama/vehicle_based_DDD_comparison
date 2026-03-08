#!/bin/bash
# ============================================================
# Auto-submission script for missing Experiment 3 mixed jobs
# ============================================================
# Submit only jobs without evaluation results.
# /tmp/missing_mixed_exp3.txt job list processing.
#
# missing count: SvmW=51, SvmA=64, Lstm=24 (Total139job(s))
#
# Usage:
#   bash scripts/hpc/launchers/submit_missing_mixed_exp3.sh --dry-run
#   bash scripts/hpc/launchers/submit_missing_mixed_exp3.sh
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
MISSING_LIST="/tmp/missing_mixed_exp3.txt"
MODE="mixed"
N_TRIALS=100
RANKING="knn"

# ---- Argument parsing ----
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ---- Queue settings (CPU queues, round-robin) ----
CPU_QUEUES=("SINGLE" "DEFAULT" "SMALL" "LONG")
QUEUE_COUNTER=0

# ---- Resource definitions (for mixed mode, SMOTE-family has increased walltime) ----
get_resources() {
    local model="$1"
    local cond="$2"
    local is_smote=false
    [[ "$cond" == "smote_plain" || "$cond" == "smote" ]] && is_smote=true

    case "$model" in
        SvmW)
            if $is_smote; then
                echo "ncpus=8:mem=24gb 24:00:00"
            else
                echo "ncpus=8:mem=24gb 16:00:00"
            fi
            ;;
        SvmA)
            if $is_smote; then
                echo "ncpus=8:mem=48gb 48:00:00"
            else
                echo "ncpus=8:mem=48gb 30:00:00"
            fi
            ;;
        Lstm)
            if $is_smote; then
                echo "ncpus=8:mem=48gb 24:00:00"
            else
                echo "ncpus=8:mem=48gb 20:00:00"
            fi
            ;;
    esac
}

# ---- Job script check ----
if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[ERROR] Job script not found: $JOB_SCRIPT"
    exit 1
fi

if [[ ! -f "$MISSING_LIST" ]]; then
    echo "[ERROR] Missing list not found: $MISSING_LIST"
    echo "  Run the Python missing-check script first."
    exit 1
fi

# ---- Log ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/submit_missing_mixed_exp3_${TIMESTAMP}.log"

echo "============================================================"
echo "  Exp3 missing mixed job(s)submit"
echo "  $(date)"
echo "  Dry run: $DRY_RUN"
echo "============================================================"

{
    echo "# Launch started at $(date)"
    echo "# Dry run: $DRY_RUN"
    echo ""
} > "$LOG_FILE"

JOB_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# ---- Main loop: missing list parsed ----
while IFS='|' read -r MODEL CONDITION DISTANCE DOMAIN SEED RATIO; do
    # Skip comment and blank lines
    [[ "$MODEL" =~ ^#.*$ ]] && continue
    [[ -z "$MODEL" ]] && continue

    # Resource acquisition
    RES=$(get_resources "$MODEL" "$CONDITION")
    NCPUS_MEM=$(echo "$RES" | cut -d' ' -f1)
    WALLTIME=$(echo "$RES" | cut -d' ' -f2)
    # Select queue by round-robin (inline to avoid subshell)
    QUEUE="${CPU_QUEUES[$((QUEUE_COUNTER % ${#CPU_QUEUES[@]}))]}"
    ((QUEUE_COUNTER++)) || true

    # LONG queue has no walltime limit; others have limits
    # SINGLE: 48h, SMALL: 24h, DEFAULT has relaxed limits; LONG: no limit
    # SvmA smote 48h goes to SINGLE, DEFAULT, or LONG
    if [[ "$QUEUE" == "SMALL" && "$WALLTIME" > "24:00:00" ]]; then
        # SMALL queue has 24h limit so changed to different queue
        QUEUE="${CPU_QUEUES[$((QUEUE_COUNTER % ${#CPU_QUEUES[@]}))]}"
        ((QUEUE_COUNTER++)) || true
        if [[ "$QUEUE" == "SMALL" ]]; then
            QUEUE="LONG"
        fi
    fi

    # Generate job name
    COND_SHORT="${CONDITION:0:2}"
    DIST_SHORT="${DISTANCE:0:1}"
    DOM_SHORT="${DOMAIN:0:1}"
    if [[ -n "$RATIO" ]]; then
        JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DIST_SHORT}${DOM_SHORT}_m_r${RATIO}_s${SEED}"
    else
        JOB_NAME="${MODEL:0:2}_bs_${DIST_SHORT}${DOM_SHORT}_m_s${SEED}"
    fi

    # qsub Build command
    VARS="MODEL=$MODEL,CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    if [[ -n "$RATIO" ]]; then
        VARS="$VARS,RATIO=$RATIO"
    fi

    CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v $VARS $JOB_SCRIPT"

    if $DRY_RUN; then
        if [[ -n "$RATIO" ]]; then
            echo "[DRY] [$QUEUE] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | mixed | r=$RATIO | s$SEED | $WALLTIME"
        else
            echo "[DRY] [$QUEUE] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | mixed | s$SEED | $WALLTIME"
        fi
        ((JOB_COUNT++)) || true
    else
        JOB_ID=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            if [[ -n "$RATIO" ]]; then
                echo "[SUBMIT] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | mixed | r=$RATIO | s$SEED | $QUEUE → $JOB_ID"
            else
                echo "[SUBMIT] $MODEL | $CONDITION | $DISTANCE | $DOMAIN | mixed | s$SEED | $QUEUE → $JOB_ID"
            fi
            echo "$MODEL:$CONDITION:$DISTANCE:$DOMAIN:mixed:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
            ((JOB_COUNT++)) || true
            sleep 0.3
        else
            echo "[ERROR] Failed: $JOB_NAME → $JOB_ID"
            ((FAIL_COUNT++)) || true
        fi
    fi

done < <(grep -v '^#' "$MISSING_LIST" | grep -v '^$')

# ---- Summary ----
{
    echo ""
    echo "# Completed at $(date)"
    echo "# Submitted: $JOB_COUNT"
    echo "# Failed: $FAIL_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
if $DRY_RUN; then
    echo "  Dry run — no jobs submitted"
    echo "  Would submit: $JOB_COUNT jobs"
else
    echo "  Submitted: $JOB_COUNT jobs"
    echo "  Failed: $FAIL_COUNT"
    echo "  Log: $LOG_FILE"
fi
echo "============================================================"

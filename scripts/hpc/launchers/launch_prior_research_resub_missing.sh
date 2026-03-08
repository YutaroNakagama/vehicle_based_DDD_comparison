#!/bin/bash
# ============================================================
# Prior research experiment — missing condition resubmission launcher
# ============================================================
# Verification found the following conditions are not yet complete:
#
# 1. SvmW s+t baseline:  24 jobs (all baseline conditions failed with old PBS script)
# 2. SvmA s+t baseline:  24 jobs (same as above)
# 3. SvmW mixed baseline: 12 jobs
# 4. SvmA mixed:          15 jobs (12 baseline + 3 ratio-based)
# 5. Lstm mixed ALL:      84 jobs (SEMINARAll failed due to 6h queue limit)
#
# Total: 159 jobs
#
# Queue strategy: SINGLE/LONG/DEFAULT distributed by round-robin to
# SEMINARNot using queue due to 6h limit
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

# ---- Fixed parameters ----
N_TRIALS=100
RANKING="knn"

# ---- Queue settings ----
QUEUES=("SINGLE" "LONG" "DEFAULT")
QUEUE_COUNTER=0

# ---- Argument parsing ----
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *)         echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Log ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_resub_missing_${TIMESTAMP}.log"

echo "============================================================"
echo "Prior research experiment — resubmit missing conditions"
echo "============================================================"
echo "Queues: ${QUEUES[*]} (round-robin)"
echo "Dry run: $DRY_RUN"
echo "============================================================"
echo ""

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[ERROR] Job script not found: $JOB_SCRIPT"
    exit 1
fi

{
    echo "# Resub launch started at $(date)"
    echo "# Command: $0 $*"
    echo ""
} > "$LOG_FILE"

JOB_COUNT=0
SKIP_COUNT=0

# ---- Job submit function ----
submit_job() {
    local MODEL="$1"
    local CONDITION="$2"
    local MODE="$3"
    local DISTANCE="$4"
    local DOMAIN="$5"
    local SEED="$6"
    local RATIO="${7:-}"
    local WALLTIME="$8"
    local MEM="$9"

    local QUEUE="${QUEUES[$((QUEUE_COUNTER % 3))]}"
    ((QUEUE_COUNTER++))

    # Generate job name
    local MODE_SHORT
    case "$MODE" in
        source_only) MODE_SHORT="s" ;;
        target_only) MODE_SHORT="t" ;;
        mixed)       MODE_SHORT="m" ;;
    esac
    local COND_SHORT="${CONDITION:0:2}"
    local JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE_SHORT}_s${SEED}"

    local CMD="qsub -N $JOB_NAME -l select=1:ncpus=8:mem=${MEM} -l walltime=${WALLTIME} -q $QUEUE"
    CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    if [[ -n "$RATIO" ]]; then
        CMD="$CMD,RATIO=$RATIO"
    fi
    CMD="$CMD $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] $MODEL | $CONDITION | $MODE | $DISTANCE | $DOMAIN | s=$SEED | r=${RATIO:-N/A} | $QUEUE"
        ((JOB_COUNT++))
    else
        local JOB_ID
        JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed: $CMD"; ((SKIP_COUNT++)); return; }
        echo "[SUBMIT] $MODEL | $CONDITION | $MODE | $DISTANCE | $DOMAIN | s$SEED | r=${RATIO:-N/A} → $JOB_ID ($QUEUE)"
        echo "$MODEL:$CONDITION:$MODE:$DISTANCE:$DOMAIN:$SEED:${RATIO:-}:$QUEUE:$JOB_ID" >> "$LOG_FILE"
        ((JOB_COUNT++))
        sleep 0.2
    fi
}

DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
SEEDS=(42 123)

echo "============================================================"
echo "[1/5] SvmW s+t baseline — 24 jobs"
echo "============================================================"
for MODE in "source_only" "target_only"; do
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                submit_job "SvmW" "baseline" "$MODE" "$DISTANCE" "$DOMAIN" "$SEED" "" "12:00:00" "16gb"
            done
        done
    done
done
echo ""

echo "============================================================"
echo "[2/5] SvmA s+t baseline — 24 jobs"
echo "============================================================"
for MODE in "source_only" "target_only"; do
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                submit_job "SvmA" "baseline" "$MODE" "$DISTANCE" "$DOMAIN" "$SEED" "" "24:00:00" "32gb"
            done
        done
    done
done
echo ""

echo "============================================================"
echo "[3/5] SvmW mixed baseline — 12 jobs"
echo "============================================================"
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            submit_job "SvmW" "baseline" "mixed" "$DISTANCE" "$DOMAIN" "$SEED" "" "16:00:00" "24gb"
        done
    done
done
echo ""

echo "============================================================"
echo "[4/5] SvmA mixed — 15 jobs (12 baseline + 3 ratio-based)"
echo "============================================================"
# 12 baseline
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            submit_job "SvmA" "baseline" "mixed" "$DISTANCE" "$DOMAIN" "$SEED" "" "30:00:00" "48gb"
        done
    done
done
# 3 specific ratio-based (wasserstein, in_domain, seed 123)
submit_job "SvmA" "smote_plain" "mixed" "wasserstein" "in_domain" "123" "0.5" "30:00:00" "48gb"
submit_job "SvmA" "smote"       "mixed" "wasserstein" "in_domain" "123" "0.5" "30:00:00" "48gb"
submit_job "SvmA" "undersample" "mixed" "wasserstein" "in_domain" "123" "0.5" "30:00:00" "48gb"
echo ""

echo "============================================================"
echo "[5/5] Lstm mixed ALL — 84 jobs"
echo "============================================================"
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            # baseline
            submit_job "Lstm" "baseline" "mixed" "$DISTANCE" "$DOMAIN" "$SEED" "" "20:00:00" "48gb"
            # ratio-based
            for RATIO in "0.1" "0.5"; do
                for COND in "smote_plain" "smote" "undersample"; do
                    submit_job "Lstm" "$COND" "mixed" "$DISTANCE" "$DOMAIN" "$SEED" "$RATIO" "20:00:00" "48gb"
                done
            done
        done
    done
done
echo ""

# ---- Summary ----
{
    echo ""
    echo "# Launch completed at $(date)"
    echo "# Total jobs submitted: $JOB_COUNT"
    echo "# Skipped: $SKIP_COUNT"
} >> "$LOG_FILE"

echo "============================================================"
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
    echo "Expected jobs: $JOB_COUNT"
else
    echo "Successfully submitted: $JOB_COUNT jobs"
    echo "Skipped: $SKIP_COUNT jobs"
    echo "Log file: $LOG_FILE"
fi
echo "============================================================"

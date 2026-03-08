#!/bin/bash
# ============================================================
# SvmA post-fix: auto-resubmit script
# ============================================================
# Auto-submit remaining unsubmitted SvmA jobs based on queue availability.
# Already submitted jobs are skipped (managed by submitted.txt).
#
# Usage:
#   nohup bash scripts/hpc/launchers/auto_resub_svma_postfix.sh &
#   bash scripts/hpc/launchers/auto_resub_svma_postfix.sh --dry-run
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SPLIT2_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
POOLED_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research.sh"

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

MODEL="SvmA"
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODES=("source_only" "target_only")
CONDITIONS=("baseline" "smote_plain" "smote" "undersample")

NCPUS_MEM="ncpus=8:mem=32gb"
WALLTIME="24:00:00"

SLEEP_INTERVAL=120
MAX_RETRIES=2000
USER="s2240011"

declare -A QUEUE_MAX
QUEUE_MAX[SINGLE]=40
QUEUE_MAX[DEFAULT]=40
QUEUE_MAX[LONG]=15
QUEUE_MAX[SMALL]=30

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/auto_resub_svma_postfix_${TIMESTAMP}.log"
SUBMITTED_FILE="$LOG_DIR/svma_postfix_submitted.txt"

# ---- Build submitted.txt from prior launcher logs (once) ----
if [[ ! -f "$SUBMITTED_FILE" ]]; then
    touch "$SUBMITTED_FILE"
    for PREV_LOG in "$LOG_DIR"/launch_svma_postfix_*.log; do
        [[ -f "$PREV_LOG" ]] || continue
        # pooled:MODEL:SEED:QUEUE:JOBID -> key = pooled:SEED
        grep "^pooled:" "$PREV_LOG" | while IFS=: read -r _ mdl seed rest; do
            echo "pooled:$seed"
        done >> "$SUBMITTED_FILE"
        # split2 (baseline):  split2:baseline:DIST:DOM:MODE:SEED:QUEUE:JOBID
        # split2 (others):    split2:COND:DIST:DOM:MODE:RATIO:SEED:QUEUE:JOBID
        # key format:         split2:COND:DIST:DOM:MODE:RATIO:SEED  (RATIO empty for baseline)
        grep "^split2:" "$PREV_LOG" | while IFS=: read -r _ cond dist dom mode f5 f6 f7 f8; do
            if [[ "$cond" == "baseline" ]]; then
                # f5=SEED, f6=QUEUE, f7=JOBID
                echo "split2:baseline:$dist:$dom:$mode::$f5"
            else
                # f5=RATIO, f6=SEED, f7=QUEUE, f8=JOBID
                echo "split2:$cond:$dist:$dom:$mode:$f5:$f6"
            fi
        done >> "$SUBMITTED_FILE"
    done
    sort -u "$SUBMITTED_FILE" -o "$SUBMITTED_FILE"
    echo "[INFO] Loaded $(wc -l < "$SUBMITTED_FILE") already-submitted keys"
fi

is_submitted() { grep -qxF "$1" "$SUBMITTED_FILE" 2>/dev/null; }
mark_submitted() { echo "$1" >> "$SUBMITTED_FILE"; }

# ---- Build full job list (one line per job) ----
JOB_LIST_FILE=$(mktemp)
trap "rm -f '$JOB_LIST_FILE'" EXIT

# Pooled
for SEED in "${SEEDS[@]}"; do
    echo "pooled:$SEED" >> "$JOB_LIST_FILE"
done

# Split2
for DIST in "${DISTANCES[@]}"; do
    for DOM in "${DOMAINS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                for COND in "${CONDITIONS[@]}"; do
                    if [[ "$COND" == "baseline" ]]; then
                        echo "split2:$COND:$DIST:$DOM:$MODE::$SEED" >> "$JOB_LIST_FILE"
                    else
                        for RATIO in "${RATIOS[@]}"; do
                            echo "split2:$COND:$DIST:$DOM:$MODE:$RATIO:$SEED" >> "$JOB_LIST_FILE"
                        done
                    fi
                done
            done
        done
    done
done

TOTAL_JOBS=$(wc -l < "$JOB_LIST_FILE")
ALREADY=$(wc -l < "$SUBMITTED_FILE")

echo "============================================================"
echo "SvmA post-fix auto-submit script"
echo "  Total: $TOTAL_JOBS  Already submitted: $ALREADY  Dry-run: $DRY_RUN"
echo "============================================================"

# ---- DRY-RUN mode ----
if $DRY_RUN; then
    COUNT=0
    while IFS= read -r JOB; do
        is_submitted "$JOB" && continue
        echo "[DRY-RUN] $JOB"
        ((COUNT++)) || true
    done < "$JOB_LIST_FILE"
    echo ""
    echo "Remaining job count: $COUNT"
    exit 0
fi

# ---- Actual submission loop ----
echo "# Auto resub started at $(date)" > "$LOG_FILE"
TOTAL_SUBMITTED=0

for RETRY in $(seq 1 $MAX_RETRIES); do
    ROUND_SUBMITTED=0

    # Count remaining
    REMAINING=0
    while IFS= read -r JOB; do
        is_submitted "$JOB" || ((REMAINING++)) || true
    done < "$JOB_LIST_FILE"
    [[ $REMAINING -eq 0 ]] && break

    # Get per-queue availability
    declare -A AVAIL
    for q in SINGLE DEFAULT LONG SMALL; do
        current=$(qstat -u "$USER" 2>/dev/null | awk -v q="$q" 'NR>5 && $3==q{n++} END{print n+0}')
        max="${QUEUE_MAX[$q]}"
        avail=$((max - current))
        [[ $avail -lt 0 ]] && avail=0
        AVAIL[$q]=$avail
    done

    TOTAL_AVAIL=$(( ${AVAIL[SINGLE]} + ${AVAIL[DEFAULT]} + ${AVAIL[LONG]} + ${AVAIL[SMALL]} ))
    if [[ $TOTAL_AVAIL -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] All queues full (remain=$REMAINING). Waiting ${SLEEP_INTERVAL}s..."
        sleep $SLEEP_INTERVAL
        continue
    fi

    echo "[$(date +%H:%M:%S)] Available slots: SINGLE=${AVAIL[SINGLE]} DEFAULT=${AVAIL[DEFAULT]} LONG=${AVAIL[LONG]} SMALL=${AVAIL[SMALL]}"

    # Build queue round-robin list
    QUEUES=()
    for q in SINGLE DEFAULT SMALL LONG; do
        [[ ${AVAIL[$q]} -gt 0 ]] && QUEUES+=("$q")
    done
    QI=0

    while IFS= read -r JOB; do
        is_submitted "$JOB" && continue
        [[ ${#QUEUES[@]} -eq 0 ]] && break

        # Pick queue via round-robin
        SELECTED_Q=""
        TRIED=0
        while [[ $TRIED -lt ${#QUEUES[@]} ]]; do
            cand="${QUEUES[$(( QI % ${#QUEUES[@]} ))]}"
            ((QI++))
            ((TRIED++))
            if [[ ${AVAIL[$cand]} -gt 0 ]]; then
                SELECTED_Q="$cand"
                break
            fi
        done
        [[ -z "$SELECTED_Q" ]] && break

        # Parse job spec: TYPE:FIELD1:FIELD2:...
        IFS=: read -r TYPE F1 F2 F3 F4 F5 F6 <<< "$JOB"

        if [[ "$TYPE" == "pooled" ]]; then
            SEED="$F1"
            JOB_NAME="SvmA_pooled_s${SEED}"
            JOB_ID=$(qsub -N "$JOB_NAME" \
                -v "MODEL=$MODEL,SEED=$SEED" \
                -l "select=1:$NCPUS_MEM" -l "walltime=$WALLTIME" -q "$SELECTED_Q" \
                "$POOLED_SCRIPT" 2>&1)
        else
            COND="$F1"; DIST="$F2"; DOM="$F3"; MODE="$F4"; RATIO="$F5"; SEED="$F6"
            CS="${COND:0:2}"
            JOB_NAME="Sa_${CS}_${DIST:0:1}${DOM:0:1}_${MODE:0:1}_s${SEED}"
            ENV_VARS="MODEL=$MODEL,CONDITION=$COND,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
            [[ -n "$RATIO" ]] && ENV_VARS="$ENV_VARS,RATIO=$RATIO"
            JOB_ID=$(qsub -N "$JOB_NAME" \
                -l "select=1:$NCPUS_MEM" -l "walltime=$WALLTIME" -q "$SELECTED_Q" \
                -v "$ENV_VARS" \
                "$SPLIT2_SCRIPT" 2>&1)
        fi

        if [[ $? -eq 0 ]]; then
            echo "[$(date +%H:%M:%S)] [$SELECTED_Q] $JOB → $JOB_ID"
            echo "$JOB:$SELECTED_Q:$JOB_ID" >> "$LOG_FILE"
            mark_submitted "$JOB"
            ((ROUND_SUBMITTED++))
            ((TOTAL_SUBMITTED++))
            ((AVAIL[$SELECTED_Q]--))
            # Remove exhausted queues
            if [[ ${AVAIL[$SELECTED_Q]} -le 0 ]]; then
                NEW_Q=()
                for q in "${QUEUES[@]}"; do
                    [[ ${AVAIL[$q]} -gt 0 ]] && NEW_Q+=("$q")
                done
                QUEUES=("${NEW_Q[@]+"${NEW_Q[@]}"}")
            fi
            sleep 0.2
        else
            ((AVAIL[$SELECTED_Q]--))
            if [[ ${AVAIL[$SELECTED_Q]} -le 0 ]]; then
                NEW_Q=()
                for q in "${QUEUES[@]}"; do
                    [[ ${AVAIL[$q]} -gt 0 ]] && NEW_Q+=("$q")
                done
                QUEUES=("${NEW_Q[@]+"${NEW_Q[@]}"}")
            fi
        fi
    done < "$JOB_LIST_FILE"

    NEW_REMAINING=$((REMAINING - ROUND_SUBMITTED))
    echo "[$(date +%H:%M:%S)] Round $RETRY: submitted=$ROUND_SUBMITTED total=$TOTAL_SUBMITTED remain=$NEW_REMAINING"
    echo "# Round $RETRY: submitted=$ROUND_SUBMITTED total=$TOTAL_SUBMITTED" >> "$LOG_FILE"

    [[ $NEW_REMAINING -le 0 ]] && break

    if [[ $ROUND_SUBMITTED -eq 0 ]]; then
        sleep $SLEEP_INTERVAL
    else
        sleep 30
    fi
done

{
    echo ""
    echo "# Auto resub completed at $(date)"
    echo "# Total submitted: $TOTAL_SUBMITTED"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "SvmA post-fix auto-submission complete: $(date)"
echo "Submitted: $TOTAL_SUBMITTED / $TOTAL_JOBS"
echo "Log: $LOG_FILE"
echo "============================================================"

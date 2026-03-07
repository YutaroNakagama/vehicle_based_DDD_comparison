#!/bin/bash
# ============================================================
# Experiment 2: Bulk Submit Remaining Extra Seeds
# ============================================================
# Submit ALL remaining jobs without waiting for queue slots.
# PBS will queue them and execute when resources are available.
#
# Spreads jobs across SINGLE, DEFAULT, SMALL queues (round-robin).
# balanced_rf uses LONG queue.
#
# Usage:
#   bash scripts/hpc/launchers/launch_exp2_extra_seeds_bulk.sh --dry-run
#   bash scripts/hpc/launchers/launch_exp2_extra_seeds_bulk.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

SEEDS=(0 7 2024)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODES=("target_only" "source_only" "mixed")

# Round-robin queues for non-balanced_rf jobs
QUEUES=("SINGLE" "DEFAULT" "SMALL")
QUEUE_IDX=0

DRY_RUN=false

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/exp2_extra_seeds_bulk_${TIMESTAMP}.log"

# Already-submitted job keys (from previous run)
PREV_LOG="$PROJECT_ROOT/scripts/hpc/logs/domain/exp2_extra_seeds_submit.log"
declare -A SUBMITTED
if [[ -f "$PREV_LOG" ]]; then
    while IFS= read -r line; do
        # Extract key like "train:baseline:target_only:mmd:out_domain:0"
        key=$(echo "$line" | sed 's/^OK://' | sed 's/:[0-9]*\..*$//')
        SUBMITTED["$key"]=1
    done < <(grep "^OK:" "$PREV_LOG" 2>/dev/null)
fi

# Also parse from [SUBMIT] lines as fallback
if [[ -f "$PREV_LOG" ]]; then
    while IFS= read -r line; do
        # Format: [SUBMIT] baseline | target_only | mmd | out_domain | s0 → 148...
        # Parse into key: train:baseline:target_only:mmd:out_domain:0
        cleaned=$(echo "$line" | sed 's/.*\[SUBMIT\] //' | sed 's/ →.*//')
        # Parse condition, mode, dist, dom, seed
        cond=$(echo "$cleaned" | awk -F' \\| ' '{print $1}' | sed 's/ r[0-9.]*//')
        mode=$(echo "$cleaned" | awk -F' \\| ' '{print $2}' | tr -d ' ')
        dist=$(echo "$cleaned" | awk -F' \\| ' '{print $3}' | tr -d ' ')
        dom=$(echo "$cleaned" | awk -F' \\| ' '{print $4}' | tr -d ' ')
        seed_part=$(echo "$cleaned" | awk -F' \\| ' '{print $5}' | tr -d ' ')
        seed=${seed_part#s}
        ratio_part=$(echo "$cleaned" | awk -F' \\| ' '{print $1}' | grep -oP 'r[0-9.]+' || echo "")
        
        if [[ -n "$ratio_part" ]]; then
            key="train:${cond}:${mode}:${dist}:${dom}:${ratio_part}:${seed}"
        else
            key="train:${cond}:${mode}:${dist}:${dom}:${seed}"
        fi
        SUBMITTED["$key"]=1
    done < <(grep "\[SUBMIT\]" "$PREV_LOG" 2>/dev/null)
fi

echo "Already submitted: ${#SUBMITTED[@]} jobs"

get_train_resources() {
    local condition="$1"
    case "$condition" in
        balanced_rf)       echo "ncpus=8:mem=12gb 10:00:00" ;;
        smote|smote_plain) echo "ncpus=4:mem=10gb 10:00:00" ;;
        *)                 echo "ncpus=4:mem=8gb 10:00:00" ;;
    esac
}

next_queue() {
    local q="${QUEUES[$QUEUE_IDX]}"
    QUEUE_IDX=$(( (QUEUE_IDX + 1) % ${#QUEUES[@]} ))
    echo "$q"
}

TOTAL_SUBMITTED=0
TOTAL_SKIPPED=0
TOTAL_FAILED=0

echo "============================================================"
echo "  Experiment 2: Bulk Submit Extra Seeds"
echo "============================================================"
echo "  Seeds   : ${SEEDS[*]}"
echo "  Dry run : $DRY_RUN"
echo "  Log     : $LOG_FILE"
echo "============================================================"

for MODE in "${MODES[@]}"; do
    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do

                # --- Baseline ---
                key="train:baseline:${MODE}:${DIST}:${DOM}:${SEED}"
                if [[ -n "${SUBMITTED[$key]:-}" ]]; then
                    ((TOTAL_SKIPPED++))
                else
                    res=$(get_train_resources "baseline")
                    ncpus_mem=$(echo "$res" | cut -d' ' -f1)
                    walltime=$(echo "$res" | cut -d' ' -f2)
                    queue=$(next_queue)
                    jname="e2_bs_${MODE:0:1}${DIST:0:2}${DOM:0:1}_s${SEED}"

                    cmd="qsub -N $jname -l select=1:$ncpus_mem -l walltime=$walltime -q $queue"
                    cmd="$cmd -v CONDITION=baseline,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                    cmd="$cmd $TRAIN_SCRIPT"

                    if $DRY_RUN; then
                        echo "[DRY-RUN] [$queue] baseline | $MODE | $DIST | $DOM | s$SEED"
                        ((TOTAL_SUBMITTED++))
                    else
                        job_id=$(eval "$cmd" 2>&1)
                        if [[ $? -eq 0 ]]; then
                            echo "[SUBMIT] [$queue] baseline | $MODE | $DIST | $DOM | s$SEED → $job_id"
                            echo "OK:$key:$job_id" >> "$LOG_FILE"
                            ((TOTAL_SUBMITTED++))
                            sleep 0.2
                        else
                            echo "[ERROR] baseline | $MODE | $DIST | $DOM | s$SEED → $job_id"
                            echo "FAIL:$key:$job_id" >> "$LOG_FILE"
                            ((TOTAL_FAILED++))
                        fi
                    fi
                fi

                # --- Ratio-based conditions ---
                for RATIO in "${RATIOS[@]}"; do
                    for COND in smote_plain smote undersample; do
                        key="train:${COND}:${MODE}:${DIST}:${DOM}:r${RATIO}:${SEED}"
                        if [[ -n "${SUBMITTED[$key]:-}" ]]; then
                            ((TOTAL_SKIPPED++))
                        else
                            res=$(get_train_resources "$COND")
                            ncpus_mem=$(echo "$res" | cut -d' ' -f1)
                            walltime=$(echo "$res" | cut -d' ' -f2)
                            queue=$(next_queue)
                            rr="${RATIO/0./}"
                            jname="e2_${COND:0:2}r${rr}_${MODE:0:1}${DIST:0:2}${DOM:0:1}_s${SEED}"

                            cmd="qsub -N $jname -l select=1:$ncpus_mem -l walltime=$walltime -q $queue"
                            cmd="$cmd -v CONDITION=$COND,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                            cmd="$cmd $TRAIN_SCRIPT"

                            if $DRY_RUN; then
                                echo "[DRY-RUN] [$queue] $COND r$RATIO | $MODE | $DIST | $DOM | s$SEED"
                                ((TOTAL_SUBMITTED++))
                            else
                                job_id=$(eval "$cmd" 2>&1)
                                if [[ $? -eq 0 ]]; then
                                    echo "[SUBMIT] [$queue] $COND r$RATIO | $MODE | $DIST | $DOM | s$SEED → $job_id"
                                    echo "OK:$key:$job_id" >> "$LOG_FILE"
                                    ((TOTAL_SUBMITTED++))
                                    sleep 0.2
                                else
                                    echo "[ERROR] $COND r$RATIO | $MODE | $DIST | $DOM | s$SEED → $job_id"
                                    echo "FAIL:$key:$job_id" >> "$LOG_FILE"
                                    ((TOTAL_FAILED++))
                                fi
                            fi
                        fi
                    done
                done

                # --- Balanced RF (LONG queue) ---
                key="train:balanced_rf:${MODE}:${DIST}:${DOM}:${SEED}"
                if [[ -n "${SUBMITTED[$key]:-}" ]]; then
                    ((TOTAL_SKIPPED++))
                else
                    res=$(get_train_resources "balanced_rf")
                    ncpus_mem=$(echo "$res" | cut -d' ' -f1)
                    walltime=$(echo "$res" | cut -d' ' -f2)
                    jname="e2_bf_${MODE:0:1}${DIST:0:2}${DOM:0:1}_s${SEED}"

                    cmd="qsub -N $jname -l select=1:$ncpus_mem -l walltime=$walltime -q LONG"
                    cmd="$cmd -v CONDITION=balanced_rf,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                    cmd="$cmd $TRAIN_SCRIPT"

                    if $DRY_RUN; then
                        echo "[DRY-RUN] [LONG] balanced_rf | $MODE | $DIST | $DOM | s$SEED"
                        ((TOTAL_SUBMITTED++))
                    else
                        job_id=$(eval "$cmd" 2>&1)
                        if [[ $? -eq 0 ]]; then
                            echo "[SUBMIT] [LONG] balanced_rf | $MODE | $DIST | $DOM | s$SEED → $job_id"
                            echo "OK:$key:$job_id" >> "$LOG_FILE"
                            ((TOTAL_SUBMITTED++))
                            sleep 0.2
                        else
                            echo "[ERROR] balanced_rf | $MODE | $DIST | $DOM | s$SEED → $job_id"
                            echo "FAIL:$key:$job_id" >> "$LOG_FILE"
                            ((TOTAL_FAILED++))
                        fi
                    fi
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
echo "  Skipped  : $TOTAL_SKIPPED (already submitted)"
echo "  Failed   : $TOTAL_FAILED"
echo "  Log      : $LOG_FILE"
echo "============================================================"

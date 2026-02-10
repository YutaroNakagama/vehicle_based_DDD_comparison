#!/bin/bash
# =============================================================================
# Continuation Launcher: Submit remaining Lstm/SvmA split2 jobs
# =============================================================================
# Submits jobs that were not submitted in the first launch due to transient
# /var/tmp errors. Picks up where launch_rerun_lstm_svma.sh left off.
#
# First launch submitted: 4 pooled + 34 SvmA split2 = 38 jobs
# Already submitted SvmA split2 for:
#   - mmd/out_domain/source_only  (s42: 7, s123: 7) = 14
#   - mmd/out_domain/target_only  (s42: 7, s123: 7) = 14
#   - mmd/in_domain/source_only   (s42: 6)           =  6  (undersample r=0.5 failed)
# Missing:
#   - SvmA: 134 split2 jobs
#   - Lstm: 168 split2 jobs
# Total remaining: 302 jobs
# =============================================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
SPLIT2_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODES=("source_only" "target_only")

# Use SEMINAR queue (no per-user max_queued limit)
QUEUE="SEMINAR"

LSTM_DEPEND="${1:-}"

get_resources() {
    local model="$1"
    case "$model" in
        SvmA) echo "ncpus=8:mem=32gb 24:00:00 $QUEUE" ;;
        Lstm) echo "ncpus=8:mem=32gb 16:00:00 $QUEUE" ;;
    esac
}

# Read already-submitted job signatures from previous log
PREV_LOG="$PROJECT_ROOT/scripts/hpc/logs/train/launch_rerun_lstm_svma_20260210_000934.log"
declare -A ALREADY_SUBMITTED
if [[ -f "$PREV_LOG" ]]; then
    while IFS= read -r line; do
        # Remove job ID from end to create signature
        SIG=$(echo "$line" | sed 's/:[0-9]*\.spcc-adm1$//')
        ALREADY_SUBMITTED["$SIG"]=1
    done < <(grep "^split2:" "$PREV_LOG")
fi
echo "Already submitted: ${#ALREADY_SUBMITTED[@]} split2 jobs"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_rerun_continue_${TIMESTAMP}.log"

echo "============================================================"
echo "  Continuation: Remaining Split2 Jobs"
echo "============================================================"
if [[ -n "$LSTM_DEPEND" ]]; then
    echo "  Lstm dependency: afterok:$LSTM_DEPEND"
fi
echo "============================================================"

{
    echo "# Continuation launch at $(date)"
    echo "# Lstm dependency: ${LSTM_DEPEND:-none}"
} > "$LOG_FILE"

SUBMIT_COUNT=0
SKIP_COUNT=0
ALREADY_COUNT=0

for MODEL in SvmA Lstm; do
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for MODE in "${MODES[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    
                    DEPEND_OPT=""
                    if [[ "$MODEL" == "Lstm" && -n "$LSTM_DEPEND" ]]; then
                        DEPEND_OPT="-W depend=afterok:${LSTM_DEPEND}"
                    fi
                    
                    # Baseline
                    SIG="split2:${MODEL}:baseline:${DISTANCE}:${DOMAIN}:${MODE}:${SEED}"
                    if [[ -n "${ALREADY_SUBMITTED[$SIG]:-}" ]]; then
                        ((ALREADY_COUNT++))
                    else
                        RESOURCES=$(get_resources "$MODEL")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                        JOB_NAME="${MODEL:0:2}_bs_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_s${SEED}"
                        
                        JOB_ID=$(qsub -N "$JOB_NAME" \
                            -l "select=1:$NCPUS_MEM" -l "walltime=$WALLTIME" -q "$QUEUE" \
                            $DEPEND_OPT \
                            -v "MODEL=$MODEL,CONDITION=baseline,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true" \
                            "$SPLIT2_SCRIPT" 2>&1) || { echo "[ERROR] $JOB_NAME: $JOB_ID"; ((SKIP_COUNT++)); continue; }
                        echo "[SUBMIT] $MODEL | baseline | $DISTANCE | $DOMAIN | $MODE | s$SEED → $JOB_ID"
                        echo "split2:$MODEL:baseline:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                        ((SUBMIT_COUNT++))
                        sleep 0.2
                    fi
                    
                    # Ratio-based
                    for RATIO in "${RATIOS[@]}"; do
                        for COND in smote_plain smote undersample; do
                            SIG="split2:${MODEL}:${COND}:${DISTANCE}:${DOMAIN}:${MODE}:${RATIO}:${SEED}"
                            if [[ -n "${ALREADY_SUBMITTED[$SIG]:-}" ]]; then
                                ((ALREADY_COUNT++))
                            else
                                RESOURCES=$(get_resources "$MODEL")
                                NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                                WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                                QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                                COND_SHORT="${COND:0:2}"
                                JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_r${RATIO}_s${SEED}"
                                
                                JOB_ID=$(qsub -N "$JOB_NAME" \
                                    -l "select=1:$NCPUS_MEM" -l "walltime=$WALLTIME" -q "$QUEUE" \
                                    $DEPEND_OPT \
                                    -v "MODEL=$MODEL,CONDITION=$COND,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true" \
                                    "$SPLIT2_SCRIPT" 2>&1) || { echo "[ERROR] $JOB_NAME: $JOB_ID"; ((SKIP_COUNT++)); continue; }
                                echo "[SUBMIT] $MODEL | $COND | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s$SEED → $JOB_ID"
                                echo "split2:$MODEL:$COND:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
                                ((SUBMIT_COUNT++))
                                sleep 0.2
                            fi
                        done
                    done
                done
            done
        done
    done
done

{
    echo ""
    echo "# Completed at $(date)"
    echo "# Submitted: $SUBMIT_COUNT"
    echo "# Skipped (already): $ALREADY_COUNT"
    echo "# Skipped (error): $SKIP_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Submitted:       $SUBMIT_COUNT"
echo "  Already existed: $ALREADY_COUNT"
echo "  Errors:          $SKIP_COUNT"
echo "  Log: $LOG_FILE"
echo "============================================================"

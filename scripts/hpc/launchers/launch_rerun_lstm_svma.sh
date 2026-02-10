#!/bin/bash
# =============================================================================
# Launcher: Re-run Experiment 3 for Lstm and SvmA (post-fix)
# =============================================================================
# Submits all pooled + split2 jobs for Lstm and SvmA only.
# Handles TMPDIR workaround for /var/tmp full issue.
#
# Usage:
#   ./scripts/hpc/launchers/launch_rerun_lstm_svma.sh --dry-run
#   ./scripts/hpc/launchers/launch_rerun_lstm_svma.sh
#   ./scripts/hpc/launchers/launch_rerun_lstm_svma.sh --lstm-depend 14738988
# =============================================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
POOLED_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research.sh"
SPLIT2_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

# TMPDIR workaround for /var/tmp full issue
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# ===== Configuration =====
MODELS=("SvmA" "Lstm")
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODES=("source_only" "target_only")

# Queue round-robin
USE_MULTI_QUEUE=true
QUEUE_COUNTER=0

# Walltime/resource settings per model
get_resources() {
    local model="$1"
    local queue
    if $USE_MULTI_QUEUE; then
        local queues=("SINGLE" "LONG" "DEFAULT")
        queue="${queues[$((QUEUE_COUNTER % 3))]}"
        ((QUEUE_COUNTER++))
    else
        queue="SINGLE"
    fi
    case "$model" in
        SvmA) echo "ncpus=8:mem=32gb 24:00:00 $queue" ;;
        Lstm) echo "ncpus=8:mem=32gb 16:00:00 $queue" ;;
    esac
}

# Applicable conditions per model (no balanced_rf for SvmA/Lstm)
get_conditions() {
    echo "baseline smote_plain smote undersample"
}

# ===== Parse arguments =====
DRY_RUN=false
LSTM_DEPEND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --lstm-depend) LSTM_DEPEND="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_rerun_lstm_svma_${TIMESTAMP}.log"

echo "============================================================"
echo "  Re-run Exp3: Lstm + SvmA (post-fix)"
echo "============================================================"
echo "  Models:    ${MODELS[*]}"
echo "  Seeds:     ${SEEDS[*]}"
echo "  Distances: ${DISTANCES[*]}"
echo "  Domains:   ${DOMAINS[*]}"
echo "  Modes:     ${MODES[*]}"
echo "  Ratios:    ${RATIOS[*]}"
echo "  TMPDIR:    $TMPDIR"
echo "  Dry run:   $DRY_RUN"
if [[ -n "$LSTM_DEPEND" ]]; then
    echo "  Lstm dependency: afterok:$LSTM_DEPEND (preprocess)"
fi
echo "============================================================"
echo ""

{
    echo "# Launch started at $(date)"
    echo "# Lstm dependency: ${LSTM_DEPEND:-none}"
    echo ""
} > "$LOG_FILE"

POOLED_COUNT=0
SPLIT2_COUNT=0
SKIP_COUNT=0

# ===== 1. Pooled jobs =====
echo "--- Pooled Jobs ---"
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        case "$MODEL" in
            SvmA) WALLTIME="24:00:00" ;;
            Lstm) WALLTIME="12:00:00" ;;
        esac
        JOB_NAME="${MODEL}_pooled_s${SEED}"
        
        DEPEND_OPT=""
        if [[ "$MODEL" == "Lstm" && -n "$LSTM_DEPEND" ]]; then
            DEPEND_OPT="-W depend=afterok:${LSTM_DEPEND}"
        fi

        if $DRY_RUN; then
            echo "[DRY-RUN] qsub -N $JOB_NAME -v MODEL=$MODEL,SEED=$SEED -l walltime=$WALLTIME -q SINGLE $DEPEND_OPT $POOLED_SCRIPT"
            ((POOLED_COUNT++))
        else
            JOB_ID=$(qsub \
                -N "$JOB_NAME" \
                -v "MODEL=$MODEL,SEED=$SEED" \
                -l "walltime=$WALLTIME" \
                -q "SINGLE" \
                $DEPEND_OPT \
                "$POOLED_SCRIPT" 2>&1) || { echo "[ERROR] Failed: $JOB_NAME"; ((SKIP_COUNT++)); continue; }
            echo "[POOLED] $MODEL | s$SEED → $JOB_ID"
            echo "pooled:$MODEL:$SEED:$JOB_ID" >> "$LOG_FILE"
            ((POOLED_COUNT++))
            sleep 0.2
        fi
    done
done
echo ""

# ===== 2. Split2 jobs =====
echo "--- Split2 Jobs ---"
for MODEL in "${MODELS[@]}"; do
    CONDITIONS=$(get_conditions "$MODEL")
    
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for MODE in "${MODES[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    
                    DEPEND_OPT=""
                    if [[ "$MODEL" == "Lstm" && -n "$LSTM_DEPEND" ]]; then
                        DEPEND_OPT="-W depend=afterok:${LSTM_DEPEND}"
                    fi
                    
                    # Baseline (no ratio)
                    RESOURCES=$(get_resources "$MODEL" )
                    NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                    WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                    QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                    
                    JOB_NAME="${MODEL:0:2}_bs_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_s${SEED}"
                    
                    if $DRY_RUN; then
                        echo "[DRY-RUN] $MODEL | baseline | $DISTANCE | $DOMAIN | $MODE | s=$SEED"
                        ((SPLIT2_COUNT++))
                    else
                        JOB_ID=$(qsub -N "$JOB_NAME" \
                            -l "select=1:$NCPUS_MEM" -l "walltime=$WALLTIME" -q "$QUEUE" \
                            $DEPEND_OPT \
                            -v "MODEL=$MODEL,CONDITION=baseline,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true" \
                            "$SPLIT2_SCRIPT" 2>&1) || { echo "[ERROR] $JOB_NAME"; ((SKIP_COUNT++)); continue; }
                        echo "[SPLIT2] $MODEL | baseline | $DISTANCE | $DOMAIN | $MODE | s$SEED → $JOB_ID"
                        echo "split2:$MODEL:baseline:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                        ((SPLIT2_COUNT++))
                        sleep 0.2
                    fi
                    
                    # Ratio-based methods
                    for RATIO in "${RATIOS[@]}"; do
                        for COND in smote_plain smote undersample; do
                            RESOURCES=$(get_resources "$MODEL")
                            NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                            WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                            QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                            
                            COND_SHORT="${COND:0:2}"
                            JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_r${RATIO}_s${SEED}"
                            
                            if $DRY_RUN; then
                                echo "[DRY-RUN] $MODEL | $COND | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s=$SEED"
                                ((SPLIT2_COUNT++))
                            else
                                JOB_ID=$(qsub -N "$JOB_NAME" \
                                    -l "select=1:$NCPUS_MEM" -l "walltime=$WALLTIME" -q "$QUEUE" \
                                    $DEPEND_OPT \
                                    -v "MODEL=$MODEL,CONDITION=$COND,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true" \
                                    "$SPLIT2_SCRIPT" 2>&1) || { echo "[ERROR] $JOB_NAME"; ((SKIP_COUNT++)); continue; }
                                echo "[SPLIT2] $MODEL | $COND | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s$SEED → $JOB_ID"
                                echo "split2:$MODEL:$COND:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
                                ((SPLIT2_COUNT++))
                                sleep 0.2
                            fi
                        done
                    done
                done
            done
        done
    done
done

# Summary
{
    echo ""
    echo "# Launch completed at $(date)"
    echo "# Pooled: $POOLED_COUNT"
    echo "# Split2: $SPLIT2_COUNT"
    echo "# Skipped: $SKIP_COUNT"
    echo "# Total: $((POOLED_COUNT + SPLIT2_COUNT))"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
if $DRY_RUN; then
    echo "  Dry run — no jobs submitted."
fi
echo "  Pooled jobs:  $POOLED_COUNT"
echo "  Split2 jobs:  $SPLIT2_COUNT"
echo "  Skipped:      $SKIP_COUNT"
echo "  Total:        $((POOLED_COUNT + SPLIT2_COUNT))"
echo ""
echo "  Expected:"
echo "    SvmA: 2 pooled + 168 split2 = 170"
echo "    Lstm: 2 pooled + 168 split2 = 170"
echo "    Total: 340 jobs"
echo "  Log: $LOG_FILE"
echo "============================================================"

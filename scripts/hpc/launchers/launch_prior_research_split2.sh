#!/bin/bash
# ============================================================
# Prior research experiment launcher for paper (domain split version)
# ============================================================
# Experiment conditions:
#   - Model: SvmA, SvmW, Lstm
#   - seeds: 42, 123
#   - Target ratio: 0.1, 0.5
#   - Imbalance methods: baseline, plain SMOTE, subject-wise SMOTE, RUS, balanced RF (※SvmA, Lstmmay not be applicable)
#   - Optuna trials: 100 (SvmW only)
#   - Optuna objective: follows each prior study
#   - Ranking method: knn
#   - Distance metrics: mmd, dtw, wasserstein
#   - Domain groups: out_domain, in_domain (2 split)
#   - Training mode: source_only (cross domain), target_only (single domain)
#
# Note:
#   - SvmA has limited imbalance method combinations due to PSO optimization
#   - Lstmis Deep Learning, so Balanced RF is not applicable
#   - SvmWsince only Optuna uses N_TRIALS=100set
#
# Total: 3 models × 3 distances × 2 domains × 2 modes × 2 seeds × conditions count = many
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

# Paper settings
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"

# Distance metrics and domain groups (2-way split)
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")

# Training mode
MODES=("source_only" "target_only")

# Model
MODELS=("SvmW" "SvmA" "Lstm")

# Queue settings (distribute submissions across multiple queues)
USE_MULTI_QUEUE=true

# Imbalance methods (applicable methods differ by model)
# SvmW: all applicable
# SvmA: baseline, smote, smote_plain, undersample (Balanced RFnot possible)
# Lstm: baseline, smote, smote_plain, undersample (Balanced RFnot possible)

# Parse arguments
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Queue counter for load balancing
QUEUE_COUNTER=0

# Resource configurations with multi-queue support
get_resources() {
    local model="$1"
    local condition="$2"
    local queue
    
    # Queue selection (round-robin distribution)
    if $USE_MULTI_QUEUE; then
        local queues=("SINGLE" "LONG" "DEFAULT")
        queue="${queues[$((QUEUE_COUNTER % 3))]}"
        ((QUEUE_COUNTER++))
    else
        queue="SINGLE"
    fi
    
    case "$model" in
        SvmA)
            # PSO optimization takes time (measured: 20-43 hours)
            echo "ncpus=8:mem=32gb 48:00:00 $queue"
            ;;
        SvmW)
            # Optuna optimization
            echo "ncpus=8:mem=16gb 12:00:00 $queue"
            ;;
        Lstm)
            # Deep Learning (CPU)
            echo "ncpus=8:mem=32gb 16:00:00 $queue"
            ;;
    esac
}

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_prior_research_split2_${TIMESTAMP}.log"

echo "============================================================"
echo "Prior research experiment launcher for paper (2-group split version)"
echo "============================================================"
echo "Model: ${MODELS[*]}"
echo "Split mode: split2 (in_domain=44 subjects, out_domain=43 subjects)"
echo "Distance metrics: ${DISTANCES[*]}"
echo "Domain groups: ${DOMAINS[*]}"
echo "Training mode: ${MODES[*]}"
echo "  - source_only (cross domain): train on opposite target domain"
echo "  - target_only (single domain): train within target domain"
echo "Imbalance methods: varies by model"
echo "seeds: ${SEEDS[*]}"
echo "Target ratio: ${RATIOS[*]}"
echo "Optuna trials (SvmWonly): $N_TRIALS"
echo "Multi-queue usage: $USE_MULTI_QUEUE (SINGLE, LONG, DEFAULT distributed to)"
echo "Dry run: $DRY_RUN"
echo "============================================================"
echo ""

# Verify job script exists
if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[WARNING] Job script not found: $JOB_SCRIPT"
    echo "[INFO] You need to create this script first."
    exit 1
fi

# Start logging
{
    echo "# Launch started at $(date)"
    echo "# Command: $0 $*"
    echo ""
} > "$LOG_FILE"

JOB_COUNT=0
SKIP_COUNT=0

# Helper function to determine applicable conditions for each model
get_conditions() {
    local model="$1"
    case "$model" in
        SvmW)
            # balanced_rf is a separate model (BalancedRF); not needed for SvmW
            echo "baseline smote_plain smote undersample"
            ;;
        SvmA|Lstm)
            echo "baseline smote_plain smote undersample"
            ;;
    esac
}

# Main loop
for MODEL in "${MODELS[@]}"; do
    CONDITIONS=$(get_conditions "$MODEL")
    
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for MODE in "${MODES[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    # Baseline (no ratio)
                    if echo "$CONDITIONS" | grep -q "baseline"; then
                        CONDITION="baseline"
                        RESOURCES=$(get_resources "$MODEL" "$CONDITION")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                        
                        JOB_NAME="${MODEL:0:2}_bs_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_s${SEED}"
                        
                        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                        CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                        CMD="$CMD $JOB_SCRIPT"
                        
                        if $DRY_RUN; then
                            echo "[DRY-RUN] $MODEL | baseline | $DISTANCE | $DOMAIN | $MODE | s=$SEED"
                        else
                            JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed: $CMD"; ((SKIP_COUNT++)); continue; }
                            echo "[SUBMIT] $MODEL | baseline | $DISTANCE | $DOMAIN | $MODE | s$SEED → $JOB_ID"
                            echo "$MODEL:baseline:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                            ((JOB_COUNT++))
                            sleep 0.2
                        fi
                    fi
                    
                    # Ratio-based methods
                    for RATIO in "${RATIOS[@]}"; do
                        for COND in "smote_plain" "smote" "undersample"; do
                            # Skip if not applicable for this model
                            if ! echo "$CONDITIONS" | grep -q "$COND"; then
                                continue
                            fi
                            
                            RESOURCES=$(get_resources "$MODEL" "$COND")
                            NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                            WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                            QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                            
                            COND_SHORT="${COND:0:2}"
                            JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_r${RATIO}_s${SEED}"
                            
                            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                            CMD="$CMD -v MODEL=$MODEL,CONDITION=$COND,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                            CMD="$CMD $JOB_SCRIPT"
                            
                            if $DRY_RUN; then
                                echo "[DRY-RUN] $MODEL | $COND | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s=$SEED"
                            else
                                JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed: $CMD"; ((SKIP_COUNT++)); continue; }
                                echo "[SUBMIT] $MODEL | $COND | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s$SEED → $JOB_ID"
                                echo "$MODEL:$COND:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
                                ((JOB_COUNT++))
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
    echo "# Total jobs submitted: $JOB_COUNT"
    echo "# Skipped: $SKIP_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
    echo "Expected jobs: $JOB_COUNT"
else
    echo "Successfully submitted: $JOB_COUNT jobs"
    echo "Skipped: $SKIP_COUNT jobs"
    echo "Log file: $LOG_FILE"
fi
echo ""
echo "Expected job count calculation:"
echo "  SvmW: 3 distances x 2 domains x 2 modes x 2 seeds × (1 baseline + 2×4 ratio-based) = 216 jobs"
echo "  SvmA: 3 distances x 2 domains x 2 modes x 2 seeds × (1 baseline + 2×3 ratio-based) = 168 jobs"
echo "  Lstm: 3 distances x 2 domains x 2 modes x 2 seeds × (1 baseline + 2×3 ratio-based) = 168 jobs"
echo "  Total: 552 jobs"
echo "============================================================"

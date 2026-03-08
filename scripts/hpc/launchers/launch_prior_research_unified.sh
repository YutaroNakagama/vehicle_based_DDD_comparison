#!/bin/bash
# ============================================================
# Unified prior research experiment launcher for paper (domain_train)
# ============================================================
# Changes (differences from split2 version):
#   - source_only/target_only eliminates duplicate training
#   - 1job(s) = 1 training + 2 evaluations (within + cross)
#   - Split ratios: train(70%) / val(15%) / test(15%)
#   - Job count halved (no MODE loop)
#
# Experiment conditions:
#   - Model: SvmA, SvmW, Lstm
#   - seeds: 42, 123
#   - Target ratio: 0.1, 0.5
#   - Imbalance methods: baseline, plain SMOTE, subject-wise SMOTE, RUS
#   - Optuna trials: 100 (SvmW only)
#   - Ranking method: knn
#   - Distance metrics: mmd, dtw, wasserstein
#   - Domain groups: out_domain, in_domain (2 split)
#
# Total: 3 models × 3 distances × 2 domains × 2 seeds × conditions count
#   SvmW: 3 × 2 × 2 × (1 + 2×3) = 84 jobs  (cf. split2: 168)
#   SvmA: 3 × 2 × 2 × (1 + 2×3) = 84 jobs  (cf. split2: 168)
#   Lstm: 3 × 2 × 2 × (1 + 2×3) = 84 jobs  (cf. split2: 168)
#   Total: 252 jobs (cf. split2: 504 ← halved)
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"

# Paper settings
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"

# Distance metrics and domain groups (2-way split)
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")

# Model
MODELS=("SvmW" "SvmA" "Lstm")

# Queue settings (distribute submissions across multiple queues)
USE_MULTI_QUEUE=true

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
            echo "ncpus=8:mem=32gb 48:00:00 $queue"
            ;;
        SvmW)
            echo "ncpus=8:mem=16gb 12:00:00 $queue"
            ;;
        Lstm)
            echo "ncpus=8:mem=32gb 16:00:00 $queue"
            ;;
    esac
}

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_prior_research_unified_${TIMESTAMP}.log"

echo "============================================================"
echo "Unified prior research experiment launcher (domain_train)"
echo "============================================================"
echo "Model: ${MODELS[*]}"
echo "Split mode: split2 (in_domain=44 subjects, out_domain=43 subjects)"
echo "Distance metrics: ${DISTANCES[*]}"
echo "Domain groups: ${DOMAINS[*]}"
echo "Training mode: domain_train (1 training + within/cross 2 evaluations)"
echo "Split ratios: train(70%) / val(15%) / test(15%)"
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
            echo "baseline smote_plain smote undersample"
            ;;
        SvmA|Lstm)
            echo "baseline smote_plain smote undersample"
            ;;
    esac
}

# Main loop (NOTE: no MODE loop — domain_train handles both within/cross)
for MODEL in "${MODELS[@]}"; do
    CONDITIONS=$(get_conditions "$MODEL")
    
    for DISTANCE in "${DISTANCES[@]}"; do
        for DOMAIN in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # Baseline (no ratio)
                if echo "$CONDITIONS" | grep -q "baseline"; then
                    CONDITION="baseline"
                    RESOURCES=$(get_resources "$MODEL" "$CONDITION")
                    NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                    WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                    QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                    
                    JOB_NAME="${MODEL:0:2}_bs_${DISTANCE:0:1}${DOMAIN:0:1}_dt_s${SEED}"
                    
                    CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                    CMD="$CMD -v MODEL=$MODEL,CONDITION=$CONDITION,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                    CMD="$CMD $JOB_SCRIPT"
                    
                    if $DRY_RUN; then
                        echo "[DRY-RUN] $MODEL | baseline | $DISTANCE | $DOMAIN | domain_train | s=$SEED"
                    else
                        JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed: $CMD"; ((SKIP_COUNT++)); continue; }
                        echo "[SUBMIT] $MODEL | baseline | $DISTANCE | $DOMAIN | domain_train | s$SEED → $JOB_ID"
                        echo "$MODEL:baseline:$DISTANCE:$DOMAIN:domain_train:$SEED:$JOB_ID" >> "$LOG_FILE"
                        ((JOB_COUNT++))
                        sleep 0.2
                    fi
                fi
                
                # Ratio-based methods
                for RATIO in "${RATIOS[@]}"; do
                    for COND in "smote_plain" "smote" "undersample"; do
                        if ! echo "$CONDITIONS" | grep -q "$COND"; then
                            continue
                        fi
                        
                        RESOURCES=$(get_resources "$MODEL" "$COND")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                        
                        COND_SHORT="${COND:0:2}"
                        JOB_NAME="${MODEL:0:2}_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_r${RATIO}_s${SEED}"
                        
                        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                        CMD="$CMD -v MODEL=$MODEL,CONDITION=$COND,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                        CMD="$CMD $JOB_SCRIPT"
                        
                        if $DRY_RUN; then
                            echo "[DRY-RUN] $MODEL | $COND | $DISTANCE | $DOMAIN | domain_train | r=$RATIO | s=$SEED"
                        else
                            JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed: $CMD"; ((SKIP_COUNT++)); continue; }
                            echo "[SUBMIT] $MODEL | $COND | $DISTANCE | $DOMAIN | domain_train | r=$RATIO | s$SEED → $JOB_ID"
                            echo "$MODEL:$COND:$DISTANCE:$DOMAIN:domain_train:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
                            ((JOB_COUNT++))
                            sleep 0.2
                        fi
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
echo "  Per model: 3 distances x 2 domains x 2 seeds x (1 baseline + 2 ratios x 3 methods) = 84 jobs"
echo "  SvmW: 84 jobs (split2 version: 168 → 50% reduction)"
echo "  SvmA: 84 jobs (split2 version: 168 → 50% reduction)"
echo "  Lstm: 84 jobs (split2 version: 168 → 50% reduction)"
echo "  Total: 252 jobs (split2 version: 504 → 50% reduction)"
echo ""
echo "Job composition:"
echo "  Training: domain_train (train on 70% within domain, 15% val)"
echo "  Eval 1: within-domain (same domain test 15%)"
echo "  Eval 2: cross-domain (opposite domain test 15%)"
echo "============================================================"

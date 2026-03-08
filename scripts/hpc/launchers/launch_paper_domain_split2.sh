#!/bin/bash
# ============================================================
# Domain shift experiment launcher for paper (2-group split version)
# ============================================================
# Experiment conditions:
#   - seeds: 42, 123
#   - Target ratio: 0.1, 0.5
#   - Models: RF (BalancedRF is included as a method)
#   - Imbalance methods: baseline, plain SMOTE, subject-wise SMOTE, RUS, Balanced RF
#   - Optuna trials: 100
#   - Optuna objective: F2 (already implemented)
#   - Ranking method: KNN
#   - Distance metrics: mmd, dtw, wasserstein
#   - Domain groups: out_domain (HIGH), in_domain (LOW) ※2group split
#   - Training mode: 
#       source_only (cross domain): train on opposite target domain
#       target_only (single domain): train within target domain
#
# New logic:
#   - source_only + out_domain → in_domaintrain with, evaluate on out_domain
#   - source_only + in_domain → out_domaintrain with, evaluate on in_domain
#   - target_only + out_domain → out_domaintrain and evaluate with
#   - target_only + in_domain → in_domaintrain and evaluate with
#
# Total: 3 distances × 2 domains × 2 modes × 2 seeds × 8 conditions = 96 jobs
# (mid_domainexcluded, so reduced from 192 to 96 jobs)
# ============================================================

set -uo pipefail  # Remove -e to continue on errors

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

# Paper settings
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"

# Distance metrics and domain groups（2group split）
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")  # mid_domainexcluded

# Training mode
# source_only = cross domain (train on the opposite target domain)
# target_only = single domain (train within the target domain)
MODES=("source_only" "target_only")

# Imbalance methods (for paper)
# Format: "CONDITION:description"
CONDITIONS=(
    "baseline:Baseline (no handling)"
    "smote_plain:Plain SMOTE"
    "smote:Subject-wise SMOTE"
    "undersample:Random Undersampling"
    "balanced_rf:Balanced RF"
)

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

# Resource configurations (memory-optimized version)
get_resources() {
    local condition="$1"
    case "$condition" in
        balanced_rf)
            # BalancedRF: 8cores required, more memory
            echo "ncpus=8:mem=12gb 08:00:00 LONG"
            ;;
        smote|smote_plain)
            # SMOTE-family: 4 cores, medium memory
            echo "ncpus=4:mem=10gb 08:00:00 SINGLE"
            ;;
        baseline|undersample)
            # lightweight experiment: 4cores, less memory
            echo "ncpus=4:mem=8gb 06:00:00 SINGLE"
            ;;
        *)
            # default
            echo "ncpus=4:mem=8gb 08:00:00 SINGLE"
            ;;
    esac
}

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_paper_domain_split2_${TIMESTAMP}.log"

echo "============================================================"
echo "Domain shift experiment launcher for paper (2-group split version)"
echo "============================================================"
echo "Split mode: split2 (in_domain=44 subjects, out_domain=43 subjects)"
echo "Distance metrics: ${DISTANCES[*]}"
echo "Domain groups: ${DOMAINS[*]} (※mid_domainnone)"
echo "Training mode: ${MODES[*]}"
echo "  - source_only (cross domain): train on opposite target domain"
echo "  - target_only (single domain): train within target domain"
echo "Imbalance methods: 5 types (baseline, plain SMOTE, subject-wise SMOTE, RUS, Balanced RF)"
echo "seeds: ${SEEDS[*]}"
echo "Target ratio: ${RATIOS[*]}"
echo "Optuna trials: $N_TRIALS"
echo "Dry run: $DRY_RUN"
echo ""
echo "Expected job count: $((${#DISTANCES[@]} * ${#DOMAINS[@]} * ${#MODES[@]} * ${#SEEDS[@]} * 8)) jobs"
echo "  - 8 jobs per condition = baseline(1) + smote_plain(2) + smote(2) + undersample(2) + balanced_rf(1)"
echo "  - 3 distances x 2 domains x 2 modes x 2 seeds × 8 = 96 jobs"
echo "============================================================"
echo ""

# Verify job script exists
if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "[ERROR] Job script not found: $JOB_SCRIPT"
    echo "Creating job script..."
    exit 1
fi

# Start logging
{
    echo "# Launch started at $(date)"
    echo "# Command: $0 $*"
    echo "# User: $(whoami)"
    echo "# Host: $(hostname)"
    echo ""
} > "$LOG_FILE"

JOB_COUNT=0
SKIP_COUNT=0

# Main loop
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # Baseline (no ratio)
                CONDITION="baseline"
                RESOURCES=$(get_resources "$CONDITION")
                NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                
                JOB_NAME="bs_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_s${SEED}"
                
                CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                CMD="$CMD -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                CMD="$CMD $JOB_SCRIPT"
                
                if $DRY_RUN; then
                    echo "[DRY-RUN] baseline | $DISTANCE | $DOMAIN | $MODE | seed=$SEED"
                else
                    JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed to submit: $CMD"; ((SKIP_COUNT++)); continue; }
                    echo "[SUBMIT] baseline | $DISTANCE | $DOMAIN | $MODE | s$SEED → $JOB_ID"
                    echo "baseline:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                    ((JOB_COUNT++))
                    sleep 0.2
                fi
                
                # SMOTE variants and undersample (with ratios)
                for RATIO in "${RATIOS[@]}"; do
                    for COND_SPEC in "smote_plain:Plain SMOTE" "smote:Subject-wise SMOTE" "undersample:RUS"; do
                        CONDITION="${COND_SPEC%%:*}"
                        
                        RESOURCES=$(get_resources "$CONDITION")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                        
                        COND_SHORT="${CONDITION:0:2}"
                        JOB_NAME="${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_r${RATIO}_s${SEED}"
                        
                        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                        CMD="$CMD -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,RATIO=$RATIO,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                        CMD="$CMD $JOB_SCRIPT"
                        
                        if $DRY_RUN; then
                            echo "[DRY-RUN] $CONDITION | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s=$SEED"
                        else
                            JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed to submit: $CMD"; ((SKIP_COUNT++)); continue; }
                            echo "[SUBMIT] $CONDITION | $DISTANCE | $DOMAIN | $MODE | r=$RATIO | s$SEED → $JOB_ID"
                            echo "$CONDITION:$DISTANCE:$DOMAIN:$MODE:$RATIO:$SEED:$JOB_ID" >> "$LOG_FILE"
                            ((JOB_COUNT++))
                            sleep 0.2
                        fi
                    done
                done
                
                # Balanced RF (no ratio)
                CONDITION="balanced_rf"
                RESOURCES=$(get_resources "$CONDITION")
                NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                
                JOB_NAME="bf_${DISTANCE:0:1}${DOMAIN:0:1}_${MODE:0:1}_s${SEED}"
                
                CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
                CMD="$CMD -v CONDITION=$CONDITION,MODE=$MODE,DISTANCE=$DISTANCE,DOMAIN=$DOMAIN,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
                CMD="$CMD $JOB_SCRIPT"
                
                if $DRY_RUN; then
                    echo "[DRY-RUN] balanced_rf | $DISTANCE | $DOMAIN | $MODE | seed=$SEED"
                else
                    JOB_ID=$(eval "$CMD" 2>&1) || { echo "[ERROR] Failed to submit: $CMD"; ((SKIP_COUNT++)); continue; }
                    echo "[SUBMIT] balanced_rf | $DISTANCE | $DOMAIN | $MODE | s$SEED → $JOB_ID"
                    echo "balanced_rf:$DISTANCE:$DOMAIN:$MODE:$SEED:$JOB_ID" >> "$LOG_FILE"
                    ((JOB_COUNT++))
                    sleep 0.2
                fi
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
echo "============================================================"

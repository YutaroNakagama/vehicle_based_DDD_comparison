#!/bin/bash
# ============================================================
# Domain shift experiment launcher for paper
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
#   - Domain groups: out_domain, in_domain
#   - Training mode: source_only (cross domain), target_only (single domain)
#
# Total: 3 distances × 2 domains × 2 modes × 2 seeds × 8 conditions = 192 jobs
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison.sh"

# Paper settings
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"

# Distance metrics and domain groups
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")

# Training mode (corresponding to paper notation)
# source_only = cross domain (train on subjects outside the domain)
# target_only = single domain (train on target subjects only)
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

# Resource configurations (Optimize based on queue status)
get_resources() {
    local condition="$1"
    case "$condition" in
        balanced_rf)
            # BalancedRF: 8 cores required, use LONG queue
            echo "ncpus=8:mem=8gb 08:00:00 LONG"
            ;;
        smote|smote_plain)
            # SMOTE-family: 4 cores, use SINGLE queue
            echo "ncpus=4:mem=8gb 08:00:00 SINGLE"
            ;;
        baseline|undersample)
            # lightweight experiment: 4cores、SINGLEQueue usage
            echo "ncpus=4:mem=8gb 06:00:00 SINGLE"
            ;;
        *)
            echo "ncpus=4:mem=8gb 06:00:00 SINGLE"
            ;;
    esac
}

# Log setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_paper_domain_${TIMESTAMP}.txt"

echo "============================================================"
echo "Domain shift experiment launcher for paper"
echo "============================================================"
echo "Distance metrics: ${DISTANCES[*]}"
echo "Domain groups: ${DOMAINS[*]}"
echo "Training mode: ${MODES[*]} (source_only=cross domain, target_only=single domain)"
echo "seeds: ${SEEDS[*]}"
echo "ratio: ${RATIOS[*]}"
echo "Optuna trials: $N_TRIALS"
echo "Ranking method: $RANKING"
echo "Imbalance methods: ${#CONDITIONS[@]}"
for cond in "${CONDITIONS[@]}"; do
    echo "  - ${cond##*:}"
done
echo "Dry run: $DRY_RUN"
echo "Log: $LOG_FILE"
echo ""
echo "Queue status:"
qstat -Q | grep -E "Queue|SINGLE|LONG|DEFAULT"
echo "============================================================"
echo ""

# Calculate total job count
TOTAL_JOBS=0
for dist in "${DISTANCES[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        for mode in "${MODES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for cond_entry in "${CONDITIONS[@]}"; do
                    CONDITION="${cond_entry%%:*}"
                    if [[ "$CONDITION" == "baseline" || "$CONDITION" == "balanced_rf" ]]; then
                        # ratio not needed
                        TOTAL_JOBS=$((TOTAL_JOBS + 1))
                    else
                        # Run for each ratio
                        for ratio in "${RATIOS[@]}"; do
                            TOTAL_JOBS=$((TOTAL_JOBS + 1))
                        done
                    fi
                done
            done
        done
    done
done

echo "Total $TOTAL_JOBS jobs to submit"
echo ""

echo "# Launched at $(date)" > "$LOG_FILE"

JOB_COUNT=0

# job(s)submit
for dist in "${DISTANCES[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        for mode in "${MODES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for cond_entry in "${CONDITIONS[@]}"; do
                    CONDITION="${cond_entry%%:*}"
                    
                    if [[ "$CONDITION" == "baseline" || "$CONDITION" == "balanced_rf" ]]; then
                        # Baseline Baseline and Balanced RF do not need ratio
                        RATIO=0
                        
                        RESOURCES=$(get_resources "$CONDITION")
                        NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                        WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                        QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                        
                        # Generate job name (15-char limit)
                        JOB_NAME="d_${CONDITION:0:4}_${dist:0:3}_${domain:0:3}_${mode:0:3}_s${seed}"
                        JOB_NAME="${JOB_NAME:0:15}"
                        
                        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v CONDITION=$CONDITION,MODE=$mode,DISTANCE=$dist,DOMAIN=$domain,RATIO=$RATIO,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true $JOB_SCRIPT"
                        
                        if $DRY_RUN; then
                            echo "[DRY-RUN] $CMD"
                        else
                            echo "[$((JOB_COUNT + 1))/$TOTAL_JOBS] Submitting: $CONDITION $dist $domain $mode s$seed"
                            JOBID=$(eval $CMD)
                            echo "$JOBID | $CONDITION | $dist | $domain | $mode | seed=$seed | $QUEUE" >> "$LOG_FILE"
                            echo "  -> JobID: $JOBID (Queue: $QUEUE)"
                        fi
                        JOB_COUNT=$((JOB_COUNT + 1))
                    else
                        # Run SMOTE-family and RUS for each ratio
                        for ratio in "${RATIOS[@]}"; do
                            RESOURCES=$(get_resources "$CONDITION")
                            NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
                            WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
                            QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
                            
                            # Generate job name (15-char limit)
                            JOB_NAME="d_${CONDITION:0:4}_r${ratio}_${dist:0:3}_${domain:0:3}_${mode:0:3}_s${seed}"
                            JOB_NAME="${JOB_NAME:0:15}"
                            
                            CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v CONDITION=$CONDITION,MODE=$mode,DISTANCE=$dist,DOMAIN=$domain,RATIO=$ratio,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true $JOB_SCRIPT"
                            
                            if $DRY_RUN; then
                                echo "[DRY-RUN] $CMD"
                            else
                                echo "[$((JOB_COUNT + 1))/$TOTAL_JOBS] Submitting: $CONDITION r$ratio $dist $domain $mode s$seed"
                                JOBID=$(eval $CMD)
                                echo "$JOBID | $CONDITION | ratio=$ratio | $dist | $domain | $mode | seed=$seed | $QUEUE" >> "$LOG_FILE"
                                echo "  -> JobID: $JOBID (Queue: $QUEUE)"
                            fi
                            JOB_COUNT=$((JOB_COUNT + 1))
                        done
                    fi
                done
            done
        done
    done
done

echo ""
echo "============================================================"
echo "Total $JOB_COUNT jobs submitted"
if ! $DRY_RUN; then
    echo "Log: $LOG_FILE"
    echo ""
    echo "Job status check:"
    echo "  qstat -u s2240011"
    echo ""
    echo "Check specific job logs:"
    echo "  tail -f $LOG_DIR/\${PBS_JOBID}.o*"
fi
echo "============================================================"

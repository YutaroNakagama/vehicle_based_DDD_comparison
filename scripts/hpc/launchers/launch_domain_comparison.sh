#!/bin/bash
# ============================================================
# Domain Analysis Comparison Launcher (HPC)
# ============================================================
# Replicates run_domain_parallel.sh for HPC environment
# Submits up to 54 parallel jobs (one per experiment combination)
#
# Usage:
#   ./launch_domain_comparison.sh                 # Default: SMOTE + Baseline (36 jobs)
#   ./launch_domain_comparison.sh --smote         # Subject-wise SMOTE only (18 jobs)
#   ./launch_domain_comparison.sh --baseline      # Baseline only (18 jobs)
#   ./launch_domain_comparison.sh --both          # SMOTE + Baseline (36 jobs)
#   ./launch_domain_comparison.sh --all           # All conditions (54 jobs)
#   ./launch_domain_comparison.sh --ratio 0.1     # Specific ratio
#   ./launch_domain_comparison.sh --ratio-all     # Both 0.1 and 0.5 ratios
#   ./launch_domain_comparison.sh --seeds "42 123"  # Multiple seeds
#   ./launch_domain_comparison.sh --dry-run       # Preview only
#
# Experiment Matrix (per condition):
#   Modes: source_only, target_only (2)
#   Distances: mmd, wasserstein, dtw (3)
#   Domains: in_domain, mid_domain, out_domain (3)
#   Total: 18 combinations per condition
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison.sh"

# Defaults
SEEDS="42 123"  # 論文用: 複数シードで統計的信頼性を確保
RATIOS="0.5"
TRIALS=100  # 論文用: 100 trials for statistical significance
DRY_RUN=false
RUN_EVAL=true
RANKING="knn"

# Conditions to run
RUN_BASELINE=false
RUN_SMOTE=false
RUN_SMOTE_PLAIN=false
RUN_UNDERSAMPLE=false
RUN_BALANCED_RF=false
EXPLICIT_SELECTION=false

# Experiment dimensions
MODES="source_only target_only"
DISTANCES="mmd wasserstein dtw"
DOMAINS="in_domain mid_domain out_domain"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEEDS="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --ratio)
            RATIOS="$2"
            shift 2
            ;;
        --ratio-all)
            RATIOS="0.1 0.5"
            shift
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --ranking)
            RANKING="$2"
            shift 2
            ;;
        --baseline)
            RUN_BASELINE=true
            EXPLICIT_SELECTION=true
            shift
            ;;
        --smote)
            RUN_SMOTE=true
            EXPLICIT_SELECTION=true
            shift
            ;;
        --smote-plain)
            RUN_SMOTE_PLAIN=true
            EXPLICIT_SELECTION=true
            shift
            ;;
        --both)
            RUN_BASELINE=true
            RUN_SMOTE=true
            EXPLICIT_SELECTION=true
            shift
            ;;
        --all)
            RUN_BASELINE=true
            RUN_SMOTE=true
            RUN_SMOTE_PLAIN=true
            RUN_UNDERSAMPLE=true
            RUN_BALANCED_RF=true
            EXPLICIT_SELECTION=true
            shift
            ;;
        --undersample)
            RUN_UNDERSAMPLE=true
            EXPLICIT_SELECTION=true
            shift
            ;;
        --balanced-rf)
            RUN_BALANCED_RF=true
            EXPLICIT_SELECTION=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-eval)
            RUN_EVAL=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default: run both baseline and smote if no explicit selection
if [[ "$EXPLICIT_SELECTION" == false ]]; then
    RUN_BASELINE=true
    RUN_SMOTE=true
fi

# Resource configuration (optimized for 100 trials)
get_resources() {
    local condition="$1"
    case "$condition" in
        baseline)
            echo "ncpus=4:mem=8gb 10:00:00 SINGLE"
            ;;
        smote|smote_plain)
            echo "ncpus=4:mem=8gb 12:00:00 SINGLE"
            ;;
        undersample)
            echo "ncpus=4:mem=8gb 08:00:00 SINGLE"
            ;;
        balanced_rf)
            echo "ncpus=8:mem=8gb 12:00:00 DEFAULT"
            ;;
        *)
            echo "ncpus=4:mem=8gb 12:00:00 SINGLE"
            ;;
    esac
}

# Setup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launcher_comparison_${TIMESTAMP}.txt"

# Calculate total jobs
NUM_SEEDS=$(echo "$SEEDS" | wc -w)
NUM_RATIOS=$(echo "$RATIOS" | wc -w)
BASE_COMBINATIONS=18  # 2 modes × 3 distances × 3 domains

TOTAL_JOBS=0
[[ "$RUN_BASELINE" == true ]] && TOTAL_JOBS=$((TOTAL_JOBS + BASE_COMBINATIONS * NUM_SEEDS))
[[ "$RUN_SMOTE" == true ]] && TOTAL_JOBS=$((TOTAL_JOBS + BASE_COMBINATIONS * NUM_SEEDS * NUM_RATIOS))
[[ "$RUN_SMOTE_PLAIN" == true ]] && TOTAL_JOBS=$((TOTAL_JOBS + BASE_COMBINATIONS * NUM_SEEDS * NUM_RATIOS))
[[ "$RUN_UNDERSAMPLE" == true ]] && TOTAL_JOBS=$((TOTAL_JOBS + BASE_COMBINATIONS * NUM_SEEDS * NUM_RATIOS))
[[ "$RUN_BALANCED_RF" == true ]] && TOTAL_JOBS=$((TOTAL_JOBS + BASE_COMBINATIONS * NUM_SEEDS))

echo "============================================================"
echo "Domain Analysis Comparison Launcher (HPC)"
echo "============================================================"
echo "Seeds: $SEEDS"
echo "Ratios: $RATIOS"
echo "Trials: $TRIALS"
echo "Ranking: $RANKING"
echo "Eval: $RUN_EVAL"
echo ""
echo "Conditions:"
echo "  Baseline: $RUN_BASELINE"
echo "  Subject-wise SMOTE: $RUN_SMOTE"
echo "  Plain SMOTE: $RUN_SMOTE_PLAIN"
echo "  Undersample RUS: $RUN_UNDERSAMPLE"
echo "  Balanced RF: $RUN_BALANCED_RF"
echo ""
echo "Dimensions:"
echo "  Modes: $MODES"
echo "  Distances: $DISTANCES"
echo "  Domains: $DOMAINS"
echo ""
echo "Total Jobs: $TOTAL_JOBS"
echo "Dry run: $DRY_RUN"
echo "Log: $LOG_FILE"
echo "============================================================"
echo ""

echo "# Launched at $(date)" > "$LOG_FILE"
echo "# Seeds: $SEEDS" >> "$LOG_FILE"
echo "# Ratios: $RATIOS" >> "$LOG_FILE"
echo "# Baseline: $RUN_BASELINE, SMOTE: $RUN_SMOTE, Plain: $RUN_SMOTE_PLAIN" >> "$LOG_FILE"

JOB_COUNT=0

# Function to submit job
submit_job() {
    local condition="$1"
    local mode="$2"
    local distance="$3"
    local domain="$4"
    local ratio="$5"
    local seed="$6"
    
    RESOURCES=$(get_resources "$condition")
    NCPUS_MEM=$(echo "$RESOURCES" | cut -d' ' -f1)
    WALLTIME=$(echo "$RESOURCES" | cut -d' ' -f2)
    QUEUE=$(echo "$RESOURCES" | cut -d' ' -f3)
    
    # Short job name: condition_distance_domain_mode
    local cond_short
    case "$condition" in
        baseline) cond_short="bl" ;;
        smote) cond_short="sw" ;;         # subject-wise
        smote_plain) cond_short="sp" ;;
        undersample) cond_short="us" ;;
        balanced_rf) cond_short="bf" ;;
        *) cond_short="${condition:0:2}" ;;
    esac
    local dist_short="${distance:0:1}"   # m, w, d
    local dom_short="${domain:0:2}"      # in, mi, ou
    local mode_short="${mode:0:1}"       # s, t
    JOB_NAME="${cond_short}${dist_short}${dom_short}${mode_short}_s${seed}"
    
    CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE"
    CMD="$CMD -v CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,RATIO=$ratio,SEED=$seed,N_TRIALS=$TRIALS,RANKING=$RANKING,RUN_EVAL=$RUN_EVAL"
    CMD="$CMD $JOB_SCRIPT"
    
    if $DRY_RUN; then
        echo "[DRY-RUN] $condition | $mode | $distance | $domain | r=$ratio | s=$seed"
    else
        JOB_ID=$(eval "$CMD" 2>&1)
        echo "[$condition] $mode/$distance/$domain -> $JOB_ID"
        echo "$condition:$mode:$distance:$domain:$ratio:$seed:$JOB_ID" >> "$LOG_FILE"
        JOB_COUNT=$((JOB_COUNT + 1))
        sleep 0.1
    fi
}

# Submit jobs
for SEED in $SEEDS; do
    echo ""
    echo "=== Seed: $SEED ==="
    
    # Baseline experiments
    if [[ "$RUN_BASELINE" == true ]]; then
        echo ""
        echo "--- Baseline (18 jobs) ---"
        for mode in $MODES; do
            for distance in $DISTANCES; do
                for domain in $DOMAINS; do
                    submit_job "baseline" "$mode" "$distance" "$domain" "0" "$SEED"
                done
            done
        done
    fi
    
    # Subject-wise SMOTE experiments
    if [[ "$RUN_SMOTE" == true ]]; then
        for ratio in $RATIOS; do
            echo ""
            echo "--- Subject-wise SMOTE ratio=$ratio (18 jobs) ---"
            for mode in $MODES; do
                for distance in $DISTANCES; do
                    for domain in $DOMAINS; do
                        submit_job "smote" "$mode" "$distance" "$domain" "$ratio" "$SEED"
                    done
                done
            done
        done
    fi
    
    # Plain SMOTE experiments
    if [[ "$RUN_SMOTE_PLAIN" == true ]]; then
        for ratio in $RATIOS; do
            echo ""
            echo "--- Plain SMOTE ratio=$ratio (18 jobs) ---"
            for mode in $MODES; do
                for distance in $DISTANCES; do
                    for domain in $DOMAINS; do
                        submit_job "smote_plain" "$mode" "$distance" "$domain" "$ratio" "$SEED"
                    done
                done
            done
        done
    fi
    
    # Undersample RUS experiments
    if [[ "$RUN_UNDERSAMPLE" == true ]]; then
        for ratio in $RATIOS; do
            echo ""
            echo "--- Undersample RUS ratio=$ratio (18 jobs) ---"
            for mode in $MODES; do
                for distance in $DISTANCES; do
                    for domain in $DOMAINS; do
                        submit_job "undersample" "$mode" "$distance" "$domain" "$ratio" "$SEED"
                    done
                done
            done
        done
    fi
    
    # Balanced RF experiments
    if [[ "$RUN_BALANCED_RF" == true ]]; then
        echo ""
        echo "--- Balanced RF (18 jobs) ---"
        for mode in $MODES; do
            for distance in $DISTANCES; do
                for domain in $DOMAINS; do
                    submit_job "balanced_rf" "$mode" "$distance" "$domain" "0" "$SEED"
                done
            done
        done
    fi
done

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "Dry run complete. No jobs submitted."
else
    echo "Submitted $JOB_COUNT jobs"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Monitor:"
    echo "  qstat -u \$USER"
    echo "  watch 'qstat -u \$USER | wc -l'"
fi
echo "============================================================"

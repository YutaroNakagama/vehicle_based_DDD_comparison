#!/bin/bash
# ============================================================
# Domain Analysis with Imbalance Handling v2
# ============================================================
# Expected experiment cases:
#   1. Baseline (no handling)
#   2. SMOTE (ratio 0.1, 0.5)
#   3. Subject-wise SMOTE (ratio 0.1, 0.5)
#   4. Balanced RF
#   5. Undersampling RUS (ratio 0.1, 0.5)
# Each with seed 42, 123
#
# Combinations:
#   - 3 distances (DTW, MMD, Wasserstein)
#   - 3 domains (in_domain, mid_domain, out_domain)
#   - 2 modes (source_only, target_only)
#   - 2 seeds (42, 123)
#   - 8 conditions (baseline, smote×2, sw-smote×2, balanced_rf, rus×2)
# Total: 3 × 3 × 2 × 2 × 8 = 288 jobs
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# PBS settings
QUEUE="DEFAULT"
WALLTIME="168:00:00"
NCPUS=4
MEM="32gb"
SCRIPT_PATH="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison.sh"

# Create log directory
LOG_DIR="${PROJECT_ROOT}/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"

# Dry run mode
DRY_RUN="${DRY_RUN:-false}"

# Experiment configurations
DISTANCES=("dtw" "mmd" "wasserstein")
DOMAINS=("in_domain" "mid_domain" "out_domain")
MODES=("source_only" "target_only")
SEEDS=(42 123)

# Imbalance handling conditions
# Format: "CONDITION:RATIO"
CONDITIONS=(
    "baseline:0"
    "smote_plain:0.1"
    "smote_plain:0.5"
    "smote:0.1"
    "smote:0.5"
    "balanced_rf:0"
    "undersample:0.1"
    "undersample:0.5"
)

N_TRIALS=150  # 2026-01-10 update: 100では18.6%が未収束のため150に増加
RANKING="knn"

echo "============================================================"
echo "DOMAIN ANALYSIS - IMBALANCE HANDLING v2"
echo "============================================================"
echo "Distances: ${DISTANCES[*]}"
echo "Domains: ${DOMAINS[*]}"
echo "Modes: ${MODES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Conditions: ${#CONDITIONS[@]}"
echo "N_TRIALS: $N_TRIALS"
echo "DRY_RUN: $DRY_RUN"
echo "============================================================"

job_count=0
submitted_jobs=()

for dist in "${DISTANCES[@]}"; do
    for domain in "${DOMAINS[@]}"; do
        for mode in "${MODES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                for cond_entry in "${CONDITIONS[@]}"; do
                    # Parse condition and ratio
                    CONDITION="${cond_entry%%:*}"
                    RATIO="${cond_entry##*:}"
                    
                    # Generate job name
                    if [[ "$CONDITION" == "baseline" || "$CONDITION" == "balanced_rf" ]]; then
                        JOB_NAME="dom_${CONDITION:0:4}_${dist:0:3}_${domain:0:3}_${mode:0:3}_s${seed}"
                    else
                        JOB_NAME="dom_${CONDITION:0:4}_r${RATIO}_${dist:0:3}_${domain:0:3}_${mode:0:3}_s${seed}"
                    fi
                    
                    # Truncate job name to 15 chars for PBS
                    JOB_NAME="${JOB_NAME:0:15}"
                    
                    ((job_count++))
                    
                    echo "[$job_count] $JOB_NAME: $CONDITION $dist $domain $mode s$seed r$RATIO"
                    
                    if [[ "$DRY_RUN" == "false" ]]; then
                        JOB_ID=$(qsub \
                            -N "$JOB_NAME" \
                            -q "$QUEUE" \
                            -l select=1:ncpus=${NCPUS}:mem=${MEM} \
                            -l walltime=${WALLTIME} \
                            -v CONDITION="$CONDITION",MODE="$mode",DISTANCE="$dist",DOMAIN="$domain",RATIO="$RATIO",SEED="$seed",N_TRIALS="$N_TRIALS",RANKING="$RANKING",RUN_EVAL="true" \
                            "$SCRIPT_PATH")
                        
                        submitted_jobs+=("$JOB_ID:$JOB_NAME")
                        echo "  -> Submitted: $JOB_ID"
                        
                        # Rate limiting
                        sleep 0.5
                    fi
                done
            done
        done
    done
done

echo ""
echo "============================================================"
echo "SUBMISSION COMPLETE"
echo "============================================================"
echo "Total jobs: $job_count"
if [[ "$DRY_RUN" == "false" ]]; then
    echo "Submitted: ${#submitted_jobs[@]}"
    echo ""
    echo "First 10 jobs:"
    for job in "${submitted_jobs[@]:0:10}"; do
        echo "  $job"
    done
fi
echo "============================================================"

# Save job list
if [[ "$DRY_RUN" == "false" && ${#submitted_jobs[@]} -gt 0 ]]; then
    JOB_LIST_FILE="${LOG_DIR}/domain_imbalv2_jobs_$(date +%Y%m%d_%H%M%S).txt"
    printf '%s\n' "${submitted_jobs[@]}" > "$JOB_LIST_FILE"
    echo "Job list saved to: $JOB_LIST_FILE"
fi

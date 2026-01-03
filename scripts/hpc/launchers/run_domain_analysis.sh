#!/bin/bash
# ==============================================================================
# Domain Analysis Launcher (Login Node)
# ==============================================================================
# Run domain analysis experiments with subject-wise SMOTE on the login node
# 
# Usage:
#   ./run_domain_analysis.sh                          # Run all combinations
#   ./run_domain_analysis.sh --mode target_only       # Single mode
#   ./run_domain_analysis.sh --distance mmd           # Single distance
#   ./run_domain_analysis.sh --domain out_domain      # Single domain
#   ./run_domain_analysis.sh --dry-run                # Show commands only
#
# Options:
#   --mode      : source_only, target_only, or all (default: all)
#   --distance  : mmd, wasserstein, dtw, or all (default: all)
#   --domain    : in_domain, mid_domain, out_domain, or all (default: all)
#   --ranking   : knn (default: knn)
#   --ratio     : Target ratio for SMOTE (default: 0.5)
#   --trials    : Number of Optuna trials (default: 50)
#   --seed      : Random seed (default: 42)
#   --model     : Model to use (default: RF)
#   --dry-run   : Print commands without executing
#   --help      : Show this help message
# ==============================================================================
set -euo pipefail

# Default parameters
MODE="all"
DISTANCE="all"
DOMAIN="all"
RANKING="knn"
RATIO="0.5"
TRIALS="10"
SEED="42"
MODEL="RF"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --distance)
            DISTANCE="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --ranking)
            RANKING="$2"
            shift 2
            ;;
        --ratio)
            RATIO="$2"
            shift 2
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            head -27 "$0" | tail -25
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            exit 1
            ;;
    esac
done

# Expand "all" to list
if [[ "$MODE" == "all" ]]; then
    MODES=("source_only" "target_only")
else
    MODES=("$MODE")
fi

if [[ "$DISTANCE" == "all" ]]; then
    DISTANCES=("mmd" "wasserstein" "dtw")
else
    DISTANCES=("$DISTANCE")
fi

if [[ "$DOMAIN" == "all" ]]; then
    DOMAINS=("in_domain" "mid_domain" "out_domain")
else
    DOMAINS=("$DOMAIN")
fi

# Setup environment
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Thread optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export JOBLIB_MULTIPROCESSING=0
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# Optuna trials override
export N_TRIALS_OVERRIDE="$TRIALS"

RANKS_DIR="results/analysis/domain/distance/subject-wise/ranks/ranks29/${RANKING}"
LOG_DIR="scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/domain_analysis_$(date +%Y%m%d_%H%M%S).log"

# Logging function
log() {
    echo "$@" | tee -a "$MASTER_LOG"
}

log "============================================================"
log "[DOMAIN ANALYSIS] Subject-wise SMOTE with ${RANKING^^} Ranking"
log "============================================================"
log "Modes     : ${MODES[*]}"
log "Distances : ${DISTANCES[*]}"
log "Domains   : ${DOMAINS[*]}"
log "Ranking   : $RANKING"
log "Model     : $MODEL"
log "SMOTE Ratio: $RATIO"
log "Trials    : $TRIALS"
log "Seed      : $SEED"
log "Dry run   : $DRY_RUN"
log "Log file  : $MASTER_LOG"
log "============================================================"
log ""

# Count total experiments
TOTAL=$(( ${#MODES[@]} * ${#DISTANCES[@]} * ${#DOMAINS[@]} ))
CURRENT=0
FAILED=0

# Run experiments
for mode in "${MODES[@]}"; do
    for distance in "${DISTANCES[@]}"; do
        for domain in "${DOMAINS[@]}"; do
            CURRENT=$((CURRENT + 1))
            
            SUBJECT_FILE="${RANKS_DIR}/${distance}_${domain}.txt"
            TAG="domain_${RANKING}_${distance}_${domain}_${mode}_subjectwise_s${SEED}"
            EXP_LOG="${LOG_DIR}/${TAG}.log"
            
            log "------------------------------------------------------------"
            log "[$CURRENT/$TOTAL] Mode: $mode | Distance: $distance | Domain: $domain"
            log "------------------------------------------------------------"
            
            if [[ ! -f "$SUBJECT_FILE" ]]; then
                log "[ERROR] Subject file not found: $SUBJECT_FILE"
                FAILED=$((FAILED + 1))
                continue
            fi
            
            log "[INFO] Subject file: $SUBJECT_FILE"
            log "[INFO] Subject count: $(wc -l < "$SUBJECT_FILE")"
            log "[INFO] Tag: $TAG"
            log "[INFO] Experiment log: $EXP_LOG"
            
            CMD="python scripts/python/train/train.py \
                --model $MODEL \
                --mode $mode \
                --seed $SEED \
                --target_file $SUBJECT_FILE \
                --tag $TAG \
                --time_stratify_labels \
                --use_oversampling \
                --oversample_method smote \
                --target_ratio $RATIO \
                --subject_wise_oversampling"
            
            log ""
            log "[CMD] $CMD"
            log ""
            
            if $DRY_RUN; then
                log "[DRY-RUN] Skipping execution"
            else
                if eval "$CMD" 2>&1 | tee -a "$EXP_LOG" "$MASTER_LOG"; then
                    log "[SUCCESS] Experiment completed"
                else
                    log "[ERROR] Experiment failed!"
                    FAILED=$((FAILED + 1))
                fi
            fi
            
            log ""
        done
    done
done

log "============================================================"
log "[SUMMARY]"
log "============================================================"
log "Total experiments: $TOTAL"
log "Completed: $((CURRENT - FAILED))"
log "Failed: $FAILED"
log "Finished at: $(date)"
log "Master log: $MASTER_LOG"
echo "============================================================"

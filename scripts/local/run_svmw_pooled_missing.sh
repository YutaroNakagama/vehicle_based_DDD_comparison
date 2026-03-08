#!/bin/bash
# ============================================================
# Run missing SvmW Pooled configs sequentially on interactive node
# ============================================================
# Usage:
#   1) qsub -I -q SINGLE -l select=1:ncpus=4:mem=16gb -l walltime=06:00:00
#   2) cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison
#   3) bash scripts/local/run_svmw_pooled_missing.sh
#
# Estimated time: ~42 min/run × 5 runs ≈ 3.5 hours
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Activate conda environment if needed
source ~/.bashrc 2>/dev/null || true
conda activate ddd 2>/dev/null || true

PBS_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research.sh"

# ---- Missing config list: CONDITION|SEED|RATIO ----
MISSING_CONFIGS=(
    "baseline|42|0.5"
    "smote_plain|123|0.5"
    "smote|42|0.5"
    "smote|123|0.5"
    "undersample|42|0.5"
)

# ---- Check if eval results exist ----
has_eval() {
    local cond="$1" seed="$2" ratio="$3"
    local eval_dir="results/outputs/evaluation/SvmW"
    local pattern
    case "$cond" in
        baseline)
            pattern="eval_results_SvmW_pooled_prior_SvmW_baseline_s${seed}"
            ;;
        smote_plain)
            pattern="eval_results_SvmW_pooled_prior_SvmW_smote_plain_ratio${ratio}_s${seed}"
            ;;
        smote)
            pattern="eval_results_SvmW_pooled_prior_SvmW_imbalv3_subjectwise_ratio${ratio}_s${seed}"
            ;;
        undersample)
            pattern="eval_results_SvmW_pooled_prior_SvmW_undersample_rus_ratio${ratio}_s${seed}"
            ;;
    esac
    find "$eval_dir" -name "${pattern}*.json" 2>/dev/null | grep -v _invalidated | grep -q .
}

# ---- Main execution ----
TOTAL=${#MISSING_CONFIGS[@]}
DONE=0
SKIPPED=0

echo "=============================================="
echo "  SvmW Pooled – Missing configs: $TOTAL"
echo "  Start: $(date)"
echo "=============================================="

for config in "${MISSING_CONFIGS[@]}"; do
    IFS='|' read -r COND SEED RATIO <<< "$config"

    echo ""
    echo "----------------------------------------------"
    echo "  [$((DONE+SKIPPED+1))/$TOTAL] COND=$COND SEED=$SEED RATIO=$RATIO"
    echo "----------------------------------------------"

    # Skip if already done
    if has_eval "$COND" "$SEED" "$RATIO"; then
        echo "  → SKIP: eval result already exists"
        ((SKIPPED++))
        continue
    fi

    # Build tag
    case "$COND" in
        baseline)
            TAG="prior_SvmW_baseline_s${SEED}"
            ;;
        smote_plain)
            TAG="prior_SvmW_smote_plain_ratio${RATIO}_s${SEED}"
            ;;
        smote)
            TAG="prior_SvmW_imbalv3_subjectwise_ratio${RATIO}_s${SEED}"
            ;;
        undersample)
            TAG="prior_SvmW_undersample_rus_ratio${RATIO}_s${SEED}"
            ;;
    esac

    # Build training command
    TRAIN_CMD="python scripts/python/train/train.py \
        --model SvmW \
        --mode pooled \
        --subject_wise_split \
        --seed $SEED \
        --time_stratify_labels \
        --tag $TAG"

    case "$COND" in
        baseline)
            ;;
        smote_plain)
            TRAIN_CMD="$TRAIN_CMD --use_oversampling --oversample_method smote --target_ratio $RATIO"
            ;;
        smote)
            TRAIN_CMD="$TRAIN_CMD --use_oversampling --oversample_method smote --target_ratio $RATIO --subject_wise_oversampling"
            ;;
        undersample)
            TRAIN_CMD="$TRAIN_CMD --use_oversampling --oversample_method undersample_rus --target_ratio $RATIO"
            ;;
    esac

    echo "  [TRAIN] $TRAIN_CMD"
    START=$(date +%s)
    eval $TRAIN_CMD
    TRAIN_EXIT=$?
    ELAPSED=$(( $(date +%s) - START ))
    echo "  [TRAIN] exit=$TRAIN_EXIT elapsed=${ELAPSED}s"

    if [[ $TRAIN_EXIT -ne 0 ]]; then
        echo "  [ERROR] Training failed, skipping eval"
        continue
    fi

    # Run evaluation (jobid auto-detected from model file)
    EVAL_CMD="python scripts/python/evaluation/evaluate.py \
        --model SvmW \
        --tag $TAG \
        --mode pooled"

    echo "  [EVAL] $EVAL_CMD"
    eval $EVAL_CMD || echo "  [WARNING] Eval failed"

    ((DONE++))
    echo "  [DONE] $DONE/$TOTAL completed (${SKIPPED} skipped)"
done

echo ""
echo "=============================================="
echo "  FINISHED: $DONE completed, $SKIPPED skipped"
echo "  End: $(date)"
echo "=============================================="

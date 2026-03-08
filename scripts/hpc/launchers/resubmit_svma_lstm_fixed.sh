#!/bin/bash
# ============================================================
# SvmA + Lstm fixed version resubmitscript
# ============================================================
# Resubmit only SvmA / Lstm with corrections to match the prior paper.
# SvmW no changes, so resubmission not needed.
#
# Fix details:
#   Lstm:
#     - learning_rate: 0.01 → 0.001 (Wang et al. 2022)
#     - batch_size:    32   → 64
#     - n_splits:      5    → 10
#     - epochs:        100  → 50
#   SvmA:
#     - Features: 22 → 14 (per paper Table 1, SVMA_PAPER_FEATURE_SUFFIXES)
#     - PSO objective: -accuracy → MSE (paper Eq.11)
#     - class_weight='balanced' removed (not mentioned in paper)
#
# Procedure:
#   1. Quarantine old SvmA/Lstm evaluation results
#   2. domain_train job(s)submit  (SvmA:84 + Lstm:84 = 168)
#   3. mixed job(s)submit          (SvmA:84 + Lstm:84 = 168)
#   Total: 336 job(s)
#
# Usage:
#   bash scripts/hpc/launchers/resubmit_svma_lstm_fixed.sh [--dry-run] [--skip-quarantine]
#   bash scripts/hpc/launchers/resubmit_svma_lstm_fixed.sh --domain-train-only [--dry-run]
#   bash scripts/hpc/launchers/resubmit_svma_lstm_fixed.sh --mixed-only [--dry-run]
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
DOMAIN_TRAIN_JOB="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
MIXED_JOB="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODELS=("SvmA" "Lstm")  # SvmW excluded — no changes needed
CONDITIONS=("baseline" "smote_plain" "smote" "undersample")

USE_MULTI_QUEUE=true
QUARANTINE_TAG="_invalidated_paper_deviation_fix"

# ---- Argument parsing ----
DRY_RUN=false
SKIP_QUARANTINE=false
DOMAIN_TRAIN_ONLY=false
MIXED_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)            DRY_RUN=true;            shift ;;
        --skip-quarantine)    SKIP_QUARANTINE=true;     shift ;;
        --domain-train-only)  DOMAIN_TRAIN_ONLY=true;   shift ;;
        --mixed-only)         MIXED_ONLY=true;          shift ;;
        *)                    echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Queue counter ----
QUEUE_COUNTER=0

# ---- Resource definitions ----
get_resources_domain_train() {
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
        SvmA) echo "ncpus=8:mem=32gb 48:00:00 $queue" ;;
        Lstm) echo "ncpus=8:mem=32gb 20:00:00 $queue" ;;   # Slightly increased for 10-fold
    esac
}

get_resources_mixed() {
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
        SvmA) echo "ncpus=8:mem=48gb 30:00:00 $queue" ;;
        Lstm) echo "ncpus=8:mem=48gb 24:00:00 $queue" ;;   # Slightly increased for 10-fold
    esac
}

# ---- Log ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/resubmit_svma_lstm_fixed_${TIMESTAMP}.log"

echo "============================================================"
echo "SvmA + Lstm fixed version resubmit"
echo "============================================================"
echo "Model         : ${MODELS[*]}"
echo "Dry run        : $DRY_RUN"
echo "Skip quarantine: $SKIP_QUARANTINE"
echo "Domain-train   : $($MIXED_ONLY && echo 'SKIP' || echo 'YES')"
echo "Mixed          : $($DOMAIN_TRAIN_ONLY && echo 'SKIP' || echo 'YES')"
echo "============================================================"
echo ""

# ============================================================
# STEP 1: Quarantine old eval + model results
# ============================================================
if ! $SKIP_QUARANTINE; then
    echo "[STEP 1] Quarantining old SvmA/Lstm results..."
    for MODEL in "${MODELS[@]}"; do
        EVAL_DIR="$PROJECT_ROOT/results/outputs/evaluation/$MODEL"
        QUARANTINE_EVAL="$EVAL_DIR/${QUARANTINE_TAG}"

        # Count non-quarantined eval dirs
        NON_Q_COUNT=$(find "$EVAL_DIR" -maxdepth 1 -mindepth 1 -not -name "_*" 2>/dev/null | wc -l)
        if [[ $NON_Q_COUNT -gt 0 ]]; then
            if $DRY_RUN; then
                echo "[DRY-RUN] Would quarantine $NON_Q_COUNT $MODEL eval result dirs → $QUARANTINE_EVAL/"
            else
                mkdir -p "$QUARANTINE_EVAL"
                find "$EVAL_DIR" -maxdepth 1 -mindepth 1 -not -name "_*" -exec mv {} "$QUARANTINE_EVAL/" \;
                echo "[QUARANTINE] Moved $NON_Q_COUNT $MODEL eval result dirs → $QUARANTINE_EVAL/"
            fi
        else
            echo "[SKIP] No non-quarantined $MODEL eval results found."
        fi
    done
    echo ""
fi

# ============================================================
# STEP 2: Submit domain_train jobs
# ============================================================
JOB_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0

submit_job() {
    local mode="$1"     # domain_train | mixed
    local model="$2"
    local condition="$3"
    local distance="$4"
    local domain="$5"
    local seed="$6"
    local ratio="${7:-}"

    local job_script resources ncpus_mem walltime queue job_name cmd

    if [[ "$mode" == "domain_train" ]]; then
        job_script="$DOMAIN_TRAIN_JOB"
        resources=$(get_resources_domain_train "$model")
    else
        job_script="$MIXED_JOB"
        resources=$(get_resources_mixed "$model")
    fi

    ncpus_mem=$(echo "$resources" | cut -d' ' -f1)
    walltime=$(echo "$resources" | cut -d' ' -f2)
    queue=$(echo "$resources" | cut -d' ' -f3)

    # Build short job name
    local model_short="${model:0:2}"
    local cond_short
    case "$condition" in
        baseline)     cond_short="bs" ;;
        smote_plain)  cond_short="sp" ;;
        smote)        cond_short="sm" ;;
        undersample)  cond_short="un" ;;
    esac
    local mode_short
    case "$mode" in
        domain_train) mode_short="dt" ;;
        mixed)        mode_short="mi" ;;
    esac

    if [[ "$condition" == "baseline" ]]; then
        job_name="${model_short}_${cond_short}_${distance:0:1}${domain:0:1}_${mode_short}_s${seed}"
    else
        job_name="${model_short}_${cond_short}_${distance:0:1}${domain:0:1}_${mode_short}_r${ratio}_s${seed}"
    fi

    # Build qsub command
    cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue"

    if [[ "$mode" == "domain_train" ]]; then
        cmd="$cmd -v MODEL=$model,CONDITION=$condition,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    else
        cmd="$cmd -v MODEL=$model,CONDITION=$condition,DISTANCE=$distance,DOMAIN=$domain,MODE=mixed,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    fi

    if [[ -n "$ratio" && "$condition" != "baseline" ]]; then
        cmd="$cmd,RATIO=$ratio"
    fi

    # mixed uses different job script — append RATIO variable
    cmd="$cmd $job_script"

    local detail="$model | $condition | $distance | $domain | $mode"
    [[ -n "$ratio" && "$condition" != "baseline" ]] && detail="$detail | r=$ratio"
    detail="$detail | s=$seed | $queue"

    if $DRY_RUN; then
        echo "[DRY-RUN] $detail"
        ((JOB_COUNT++))
    else
        JOB_ID=$(eval "$cmd" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[SUBMIT] $detail → $JOB_ID"
            echo "$model:$condition:$distance:$domain:$mode:${ratio:-none}:$seed:$JOB_ID" >> "$LOG_FILE"
            ((JOB_COUNT++))
            sleep 0.15
        else
            echo "[FAIL] $detail → $JOB_ID"
            echo "FAIL:$model:$condition:$distance:$domain:$mode:${ratio:-none}:$seed:$JOB_ID" >> "$LOG_FILE"
            ((FAIL_COUNT++))
        fi
    fi
}

# ---- domain_train job(s) ----
if ! $MIXED_ONLY; then
    echo "============================================================"
    echo "[STEP 2] Submitting domain_train jobs (SvmA + Lstm)"
    echo "============================================================"

    {
        echo "# Resubmit started at $(date)"
        echo "# Command: $0 $*"
        echo ""
    } > "$LOG_FILE"

    for MODEL in "${MODELS[@]}"; do
        for DISTANCE in "${DISTANCES[@]}"; do
            for DOMAIN in "${DOMAINS[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    # Baseline (no ratio)
                    submit_job "domain_train" "$MODEL" "baseline" "$DISTANCE" "$DOMAIN" "$SEED"

                    # Ratio-based conditions
                    for RATIO in "${RATIOS[@]}"; do
                        for COND in "smote_plain" "smote" "undersample"; do
                            submit_job "domain_train" "$MODEL" "$COND" "$DISTANCE" "$DOMAIN" "$SEED" "$RATIO"
                        done
                    done
                done
            done
        done
    done

    echo ""
    echo "[domain_train] Submitted: $JOB_COUNT | Failed: $FAIL_COUNT"
    echo ""
fi

# ---- mixed job(s) ----
DT_COUNT=$JOB_COUNT
if ! $DOMAIN_TRAIN_ONLY; then
    echo "============================================================"
    echo "[STEP 3] Submitting mixed jobs (SvmA + Lstm)"
    echo "============================================================"

    for MODEL in "${MODELS[@]}"; do
        for DISTANCE in "${DISTANCES[@]}"; do
            for DOMAIN in "${DOMAINS[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    # Baseline
                    submit_job "mixed" "$MODEL" "baseline" "$DISTANCE" "$DOMAIN" "$SEED"

                    # Ratio-based
                    for RATIO in "${RATIOS[@]}"; do
                        for COND in "smote_plain" "smote" "undersample"; do
                            submit_job "mixed" "$MODEL" "$COND" "$DISTANCE" "$DOMAIN" "$SEED" "$RATIO"
                        done
                    done
                done
            done
        done
    done

    echo ""
    MIXED_COUNT=$((JOB_COUNT - DT_COUNT))
    echo "[mixed] Submitted: $MIXED_COUNT | Failed: $FAIL_COUNT"
fi

# ============================================================
# Summary
# ============================================================
{
    echo ""
    echo "# Completed at $(date)"
    echo "# Total submitted: $JOB_COUNT"
    echo "# Failed: $FAIL_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
if $DRY_RUN; then
    echo "  Dry run — no jobs submitted"
    echo "  Expected jobs: $JOB_COUNT"
else
    echo "  Total submitted: $JOB_COUNT"
    echo "  Failed: $FAIL_COUNT"
    echo "  Log: $LOG_FILE"
fi
echo ""
echo "  SvmA: 84 domain_train + 84 mixed = 168"
echo "  Lstm: 84 domain_train + 84 mixed = 168"
echo "  Total: 336 job(s)"
echo ""
echo "  ※ If queue limit (~156) exceeded, split submission required:"
echo "    --domain-train-only  : domain_train only (168)"
echo "    --mixed-only         : mixed only (168)"
echo "============================================================"

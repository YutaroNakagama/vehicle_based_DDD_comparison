#!/bin/bash
# ============================================================
# Re-execution daemon — resubmit SvmA jobs after old code fix
# ============================================================
# SvmA code from 2026-02-14/15 fundamentally revised in:
#   - 6bb19f3: class_weight balanced, PSO F1 objective, F2 threshold
#   - 1c3742c: proper ANFIS implementation (Gaussian MF + Takagi-Sugeno rules)
#   - 5418e47: Align PSO hyperparameters to Table 3 in the paper
#   - 0152185: MinMaxnormalization, KSS6→Alert, PSO→accuracy
#   - adf7802: scaler save bug fix
#
# All SvmA baseline + imbalv3 jobs run before the fix are invalid.
# (smote_plain / undersample  — oversampling daemon handles this)
#
# Target:
#   - SvmA baseline: 3mode × 3 distances x 2 domains x 2 seeds = 36 configs
#   - SvmA imbalv3:  3mode × 3 distances x 2 domains x 2 seeds × 2ratio = 72 configs
#   Total: max 108 configs (92 items excluding completed)
#
# Usage:
#   nohup bash scripts/hpc/launchers/rerun_svma_old_code_fix.sh &
#   # Log: /tmp/rerun_svma_old_code_fix.log
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
LOG="/tmp/rerun_svma_old_code_fix.log"
SUBMITTED_KEYS="/tmp/rerun_svma_old_code_fix_keys.txt"
POLL_INTERVAL=300  # 5 minutes
N_TRIALS=100
RANKING="knn"

# ---- Error trap ----
trap 'echo "[$(date +%H:%M)] TRAP: daemon exiting (line $LINENO, exit=$?)" >> "$LOG"' EXIT
trap 'echo "[$(date +%H:%M)] TRAP: received signal, exiting" >> "$LOG"; exit 1' INT TERM HUP

# ---- Queue limit ----
declare -A QUEUE_MAX=( [SINGLE]=40 [DEFAULT]=40 [SMALL]=30 [LONG]=15 )
declare -A QUEUE_CURRENT=()
CPU_QUEUES=("SINGLE" "DEFAULT" "SMALL" "LONG")

touch "$SUBMITTED_KEYS"

# ---- Resource definitions ----
get_resources() {
    local mode="$1" cond="$2"

    # SvmA: PSO optimization takes significant time (actual RSS max ~1.3GB)
    # reduced from ncpus=8:mem=48gb
    if [[ "$mode" == "mixed" ]]; then
        echo "ncpus=4:mem=8gb 48:00:00"
    else
        echo "ncpus=4:mem=8gb 30:00:00"
    fi
}

# ---- New check if evaluation results exist ----
has_eval_result() {
    local cond="$1" dist="$2" dom="$3" mode="$4" seed="$5" ratio="$6"
    local eval_dir="results/outputs/evaluation/SvmA"

    local pattern
    if [[ "$cond" == "baseline" ]]; then
        pattern="eval_results_SvmA_${mode}_prior_SvmA_baseline_knn_${dist}_${dom}_${mode}_split2_s${seed}"
    else
        # imbalv3 → subjectwise in filename
        pattern="eval_results_SvmA_${mode}_prior_SvmA_imbalv3_knn_${dist}_${dom}_${mode}_split2_subjectwise_ratio${ratio}_s${seed}"
    fi

    # Check if matching file exists (excluding invalidated)
    find "$eval_dir" -name "${pattern}*.json" 2>/dev/null | grep -v _invalidated | grep -q .
}

# ---- Check queue status ----
get_queue_counts() {
    local qstat_output
    qstat_output=$(qstat -u s2240011 2>/dev/null | tail -n +6 || true)

    for q in "${CPU_QUEUES[@]}"; do
        QUEUE_CURRENT[$q]=$(echo "$qstat_output" | awk -v q="$q" '$3==q' | wc -l || echo 0)
    done
}

find_available_queue() {
    for q in "${CPU_QUEUES[@]}"; do
        local current="${QUEUE_CURRENT[$q]:-0}"
        local max="${QUEUE_MAX[$q]:-0}"
        if (( current < max )); then
            echo "$q"
            return 0
        fi
    done
    return 1
}

# ---- Enumerate all experiment conditions ----
ALL_JOBS=()
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("in_domain" "out_domain")
SEEDS=(42 123)
RATIOS=(0.1 0.5)
MODES=("source_only" "target_only" "mixed")
CONDITIONS=("baseline" "imbalv3")

for MODE in "${MODES[@]}"; do
    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # baseline (no ratio)
                ALL_JOBS+=("baseline|$DIST|$DOM|$MODE|$SEED|none")
                # imbalv3 (with ratios)
                for RATIO in "${RATIOS[@]}"; do
                    ALL_JOBS+=("imbalv3|$DIST|$DOM|$MODE|$SEED|$RATIO")
                done
            done
        done
    done
done

echo "[$(date +%H:%M)] SvmA old-code-fix daemon started. Total configs: ${#ALL_JOBS[@]}" >> "$LOG"
echo "[$(date +%H:%M)] Conditions: ${CONDITIONS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Modes: ${MODES[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Polling every ${POLL_INTERVAL}s" >> "$LOG"

# ---- Main loop ----
while true; do
    get_queue_counts || true

    SUBMITTED_THIS_ROUND=0
    REMAINING=0

    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r COND DIST DOM MODE SEED RATIO <<< "$job_spec"

        KEY="SvmA:${COND}:${DIST}:${DOM}:${MODE}:s${SEED}:r${RATIO}"

        # Skip if eval result already exists
        if has_eval_result "$COND" "$DIST" "$DOM" "$MODE" "$SEED" "$RATIO"; then
            continue
        fi

        # Skip if already submitted
        if grep -qF "$KEY" "$SUBMITTED_KEYS" 2>/dev/null; then
            ((REMAINING++)) || true
            continue
        fi

        # Find available queue
        QUEUE=""
        QUEUE=$(find_available_queue) || true
        if [[ -z "$QUEUE" ]]; then
            ((REMAINING++)) || true
            continue
        fi

        # Get resources
        RES=$(get_resources "$MODE" "$COND")
        NCPUS_MEM=$(echo "$RES" | cut -d' ' -f1)
        WALLTIME=$(echo "$RES" | cut -d' ' -f2)

        # Generate job name (compact)
        COND_SHORT="${COND:0:2}"
        MODE_SHORT="${MODE:0:2}"
        DIST_SHORT="${DIST:0:2}"
        DOM_SHORT="${DOM:0:1}"
        JOB_NAME="SA_${COND_SHORT}_${DIST_SHORT}${DOM_SHORT}_${MODE_SHORT}_s${SEED}"
        if [[ "$RATIO" != "none" ]]; then
            JOB_NAME="${JOB_NAME}_r${RATIO}"
        fi

        # Build qsub command
        # Map internal condition name to PBS script expected name
        # PBS script expects "smote" for the imbalv3 condition
        PBS_COND="$COND"
        if [[ "$COND" == "imbalv3" ]]; then
            PBS_COND="smote"
        fi
        VARS="MODEL=SvmA,CONDITION=$PBS_COND,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        if [[ "$RATIO" != "none" ]]; then
            VARS="${VARS},RATIO=$RATIO"
        fi

        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v $VARS $JOB_SCRIPT"

        JOB_ID=$(eval "$CMD" 2>&1) || {
            echo "  [ERR] Failed: $KEY ($CMD)" >> "$LOG"
            ((REMAINING++)) || true
            continue
        }

        # Record submission
        echo "$KEY:$JOB_ID" >> "$SUBMITTED_KEYS"
        ((SUBMITTED_THIS_ROUND++)) || true

        # Update queue count
        QUEUE_CURRENT[$QUEUE]=$(( ${QUEUE_CURRENT[$QUEUE]:-0} + 1 ))

        echo "  [SUB] SvmA | $COND | $MODE | $DIST | $DOM | r=${RATIO} | s$SEED | $QUEUE → $JOB_ID" >> "$LOG"

        sleep 0.3
    done

    TOTAL_QUEUED=$(qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l)
    TOTAL_SUBMITTED=$(wc -l < "$SUBMITTED_KEYS")
    echo "[POLL] $(date +%H:%M) | queued=$TOTAL_QUEUED | submitted=$TOTAL_SUBMITTED | new=$SUBMITTED_THIS_ROUND | remaining=$REMAINING" >> "$LOG"

    if [[ "$SUBMITTED_THIS_ROUND" -eq 0 && "$REMAINING" -eq 0 ]]; then
        echo "[DONE] All SvmA old-code-fix re-run jobs submitted or completed. Exiting." >> "$LOG"
        break
    fi

    if [[ "$SUBMITTED_THIS_ROUND" -eq 0 ]]; then
        echo "  (all queues full or waiting for results, sleeping...)" >> "$LOG"
    fi

    sleep "$POLL_INTERVAL"
done

echo "[$(date +%H:%M)] Daemon finished." >> "$LOG"

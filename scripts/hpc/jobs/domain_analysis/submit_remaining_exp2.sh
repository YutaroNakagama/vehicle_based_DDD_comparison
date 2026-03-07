#!/bin/bash
# ============================================================
# Submit remaining Experiment 2 jobs across multiple queues
# Distributes across DEFAULT, SINGLE, SMALL to bypass per-queue limits
# Re-runnable: skips already queued/running jobs
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

LOG_DIR="scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUBMIT_LOG="${LOG_DIR}/exp2_remaining_${TIMESTAMP}.log"
SUBMITTED_FILE="${LOG_DIR}/exp2_remaining_submitted_${TIMESTAMP}.txt"

PBS_SCRIPT="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

# Resource settings
RF_WALLTIME="10:00:00"
RF_MEM="8gb"
RF_NCPUS=4

BRF_WALLTIME="24:00:00"
BRF_MEM="8gb"
BRF_NCPUS=4

# Queue rotation: try each queue in order
QUEUES=("DEFAULT" "SINGLE" "SMALL")

# Get current job count per queue for this user
declare -A queue_current
declare -A queue_max
queue_max[DEFAULT]=40
queue_max[SINGLE]=40
queue_max[SMALL]=30

for q in "${QUEUES[@]}"; do
    queue_current[$q]=$(qstat -u "$USER" 2>/dev/null | awk -v q="$q" '$3==q {count++} END {print count+0}')
done

echo "Current queue usage:" | tee "$SUBMIT_LOG"
for q in "${QUEUES[@]}"; do
    echo "  $q: ${queue_current[$q]}/${queue_max[$q]}" | tee -a "$SUBMIT_LOG"
done

# Get already running/queued Exp2 job names (from qstat)
mapfile -t existing_jobs < <(qstat -u "$USER" 2>/dev/null | awk 'NR>5 {print $4}' | sed 's/\*$//')

# Experiment parameters
CONDITIONS="baseline smote_plain smote undersample balanced_rf"
MODES="source_only target_only mixed"
DISTANCES="mmd dtw wasserstein"
DOMAINS="in_domain out_domain"
SEEDS="42 123"
RATIOS="0.1 0.5"

total=0
submitted=0
skipped=0
failed=0

# Build job name (same logic as original script)
make_job_name() {
    local cond="$1" mode="$2" dist="$3" dom="$4" seed="$5" ratio="${6:-}"
    local cond_abbr mode_abbr dist_abbr dom_abbr ratio_short=""
    
    case "$cond" in
        baseline)     cond_abbr="bs" ;;
        smote_plain)  cond_abbr="sp" ;;
        smote)        cond_abbr="sm" ;;
        undersample)  cond_abbr="us" ;;
        balanced_rf)  cond_abbr="bf" ;;
    esac
    case "$mode" in
        source_only) mode_abbr="so" ;;
        target_only) mode_abbr="to" ;;
        mixed)       mode_abbr="mx" ;;
    esac
    case "$dist" in
        mmd)          dist_abbr="mmd" ;;
        dtw)          dist_abbr="dtw" ;;
        wasserstein)  dist_abbr="was" ;;
    esac
    case "$dom" in
        in_domain)   dom_abbr="in" ;;
        out_domain)  dom_abbr="ou" ;;
    esac
    
    if [[ -n "$ratio" ]]; then
        ratio_short=$(echo "$ratio" | sed 's/0\./r/')
        echo "${cond_abbr}${ratio_short}_${dist_abbr}_${dom_abbr}_${mode_abbr}_s${seed}"
    else
        echo "${cond_abbr}_${dist_abbr}_${dom_abbr}_${mode_abbr}_s${seed}"
    fi
}

# Check if job is already queued/running
is_already_queued() {
    local job_name="$1"
    for ej in "${existing_jobs[@]}"; do
        if [[ "$ej" == "$job_name" ]]; then
            return 0
        fi
    done
    return 1
}

# Find a queue with available slots
find_available_queue() {
    for q in "${QUEUES[@]}"; do
        if (( ${queue_current[$q]} < ${queue_max[$q]} )); then
            echo "$q"
            return 0
        fi
    done
    return 1
}

submit_job() {
    local cond="$1" mode="$2" dist="$3" dom="$4" seed="$5" ratio="${6:-}"
    
    local job_name
    job_name=$(make_job_name "$cond" "$mode" "$dist" "$dom" "$seed" "$ratio")
    total=$((total + 1))
    
    # Skip if already queued
    if is_already_queued "$job_name"; then
        skipped=$((skipped + 1))
        return
    fi
    
    # Find available queue
    local queue
    queue=$(find_available_queue) || {
        failed=$((failed + 1))
        echo "  ⛔ $job_name: all queues full" | tee -a "$SUBMIT_LOG"
        echo "REMAINING $job_name $cond $mode $dist $dom $seed $ratio" >> "$SUBMITTED_FILE"
        return
    }
    
    # Select resources
    local walltime mem
    if [[ "$cond" == "balanced_rf" ]]; then
        walltime="$BRF_WALLTIME"
        mem="$BRF_MEM"
    else
        walltime="$RF_WALLTIME"
        mem="$RF_MEM"
    fi
    
    # Environment variables
    local env_vars="CONDITION=${cond},MODE=${mode},DISTANCE=${dist},DOMAIN=${dom},SEED=${seed}"
    if [[ -n "$ratio" ]]; then
        env_vars="${env_vars},RATIO=${ratio}"
    fi
    
    # Submit
    local result
    result=$(qsub \
        -N "$job_name" \
        -q "$queue" \
        -l select=1:ncpus=${RF_NCPUS}:mem=${mem} \
        -l walltime=${walltime} \
        -v "$env_vars" \
        "$PBS_SCRIPT" 2>&1) || true
    
    if [[ "$result" == *".spcc-adm1"* ]]; then
        submitted=$((submitted + 1))
        queue_current[$queue]=$((${queue_current[$queue]} + 1))
        local jid=$(echo "$result" | grep -oP '\d+\.spcc' | grep -oP '\d+')
        echo "SUBMITTED $jid $job_name $queue $cond $mode $dist $dom $seed $ratio" >> "$SUBMITTED_FILE"
        echo "  ✅ [$submitted] $job_name → $jid ($queue)" | tee -a "$SUBMIT_LOG"
    else
        failed=$((failed + 1))
        echo "  ❌ $job_name ($queue): $result" | tee -a "$SUBMIT_LOG"
        echo "REMAINING $job_name $cond $mode $dist $dom $seed $ratio" >> "$SUBMITTED_FILE"
        # Mark this queue as full
        queue_current[$queue]=${queue_max[$queue]}
    fi
}

echo "" | tee -a "$SUBMIT_LOG"
echo "============================================================" | tee -a "$SUBMIT_LOG"
echo "Experiment 2 Remaining Jobs Submission" | tee -a "$SUBMIT_LOG"
echo "Date: $(date)" | tee -a "$SUBMIT_LOG"
echo "Commit: $(git rev-parse --short HEAD)" | tee -a "$SUBMIT_LOG"
echo "============================================================" | tee -a "$SUBMIT_LOG"

# Submit all combinations
for cond in $CONDITIONS; do
    echo "" | tee -a "$SUBMIT_LOG"
    echo "── Condition: $cond ──" | tee -a "$SUBMIT_LOG"
    
    case "$cond" in
        baseline|balanced_rf)
            for mode in $MODES; do
                for dist in $DISTANCES; do
                    for dom in $DOMAINS; do
                        for seed in $SEEDS; do
                            submit_job "$cond" "$mode" "$dist" "$dom" "$seed"
                        done
                    done
                done
            done
            ;;
        smote_plain|smote|undersample)
            for ratio in $RATIOS; do
                for mode in $MODES; do
                    for dist in $DISTANCES; do
                        for dom in $DOMAINS; do
                            for seed in $SEEDS; do
                                submit_job "$cond" "$mode" "$dist" "$dom" "$seed" "$ratio"
                            done
                        done
                    done
                done
            done
            ;;
    esac
done

echo "" | tee -a "$SUBMIT_LOG"
echo "============================================================" | tee -a "$SUBMIT_LOG"
echo "SUMMARY" | tee -a "$SUBMIT_LOG"
echo "  Total combinations: $total" | tee -a "$SUBMIT_LOG"
echo "  Already queued:     $skipped" | tee -a "$SUBMIT_LOG"
echo "  Submitted now:      $submitted" | tee -a "$SUBMIT_LOG"
echo "  Failed/remaining:   $failed" | tee -a "$SUBMIT_LOG"
echo "  Log:                $SUBMIT_LOG" | tee -a "$SUBMIT_LOG"
echo "  Tracking:           $SUBMITTED_FILE" | tee -a "$SUBMIT_LOG"
echo "============================================================" | tee -a "$SUBMIT_LOG"

if (( failed > 0 )); then
    remaining=$(grep -c "^REMAINING" "$SUBMITTED_FILE" 2>/dev/null || echo 0)
    echo "" | tee -a "$SUBMIT_LOG"
    echo "⚠ $remaining jobs remaining. Re-run this script when queue slots open:" | tee -a "$SUBMIT_LOG"
    echo "  bash $0" | tee -a "$SUBMIT_LOG"
fi

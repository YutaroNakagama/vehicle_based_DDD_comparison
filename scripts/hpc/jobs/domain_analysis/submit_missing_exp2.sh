#!/bin/bash
# ============================================================
# Auto-submit remaining Experiment 2 jobs as queue slots open
# Re-runnable: reads from remaining list, submits what it can
# Usage: bash submit_missing_exp2.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

PBS_SCRIPT="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
MISSING_FILE="/tmp/missing_norm.txt"
STILL_REMAINING="/tmp/still_missing_exp2.txt"

LOG_DIR="scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUBMIT_LOG="${LOG_DIR}/exp2_fill_${TIMESTAMP}.log"

# Queue config
QUEUES=("DEFAULT" "SINGLE" "SMALL")
declare -A queue_max
queue_max[DEFAULT]=40
queue_max[SINGLE]=40
queue_max[SMALL]=30

if [[ ! -f "$MISSING_FILE" ]]; then
    echo "Missing file not found: $MISSING_FILE"
    exit 1
fi

total_missing=$(wc -l < "$MISSING_FILE")
echo "============================================================" | tee "$SUBMIT_LOG"
echo "Exp2 fill submission - $(date)" | tee -a "$SUBMIT_LOG"
echo "Missing configs: $total_missing" | tee -a "$SUBMIT_LOG"
echo "============================================================" | tee -a "$SUBMIT_LOG"

# Get current queue usage
declare -A queue_current
for q in "${QUEUES[@]}"; do
    queue_current[$q]=$(qstat -u "$USER" 2>/dev/null | awk -v q="$q" '$3==q {c++} END {print c+0}')
done

available=0
for q in "${QUEUES[@]}"; do
    slots=$(( ${queue_max[$q]} - ${queue_current[$q]} ))
    (( slots < 0 )) && slots=0
    available=$((available + slots))
    echo "  $q: ${queue_current[$q]}/${queue_max[$q]} (${slots} free)" | tee -a "$SUBMIT_LOG"
done
echo "  Total available: $available" | tee -a "$SUBMIT_LOG"

if (( available == 0 )); then
    echo "" | tee -a "$SUBMIT_LOG"
    echo "⚠ No queue slots available. Re-run later:" | tee -a "$SUBMIT_LOG"
    echo "  bash $0" | tee -a "$SUBMIT_LOG"
    cp "$MISSING_FILE" "$STILL_REMAINING"
    exit 0
fi

submitted=0
failed=0
> "$STILL_REMAINING"

while IFS=' ' read -r cond mode dist dom seed ratio; do
    # Find available queue
    queue=""
    for q in "${QUEUES[@]}"; do
        if (( ${queue_current[$q]} < ${queue_max[$q]} )); then
            queue="$q"
            break
        fi
    done
    
    if [[ -z "$queue" ]]; then
        echo "$cond $mode $dist $dom $seed $ratio" >> "$STILL_REMAINING"
        failed=$((failed + 1))
        continue
    fi
    
    # Resources
    walltime="10:00:00"
    [[ "$cond" == "balanced_rf" ]] && walltime="24:00:00"
    
    # Environment variables
    env_vars="CONDITION=${cond},MODE=${mode},DISTANCE=${dist},DOMAIN=${dom},SEED=${seed}"
    [[ -n "$ratio" ]] && env_vars="${env_vars},RATIO=${ratio}"
    
    # Job name
    case "$cond" in baseline) ca="bs";; smote_plain) ca="sp";; smote) ca="sm";; undersample) ca="us";; balanced_rf) ca="bf";; esac
    case "$mode" in source_only) ma="so";; target_only) ma="to";; mixed) ma="mx";; *) ma="xx";; esac
    case "$dist" in mmd) da="mmd";; dtw) da="dtw";; wasserstein) da="was";; esac
    case "$dom" in in_domain) dma="in";; out_domain) dma="ou";; esac
    rs=""
    [[ -n "$ratio" ]] && rs=$(echo "$ratio" | sed 's/0\./r/')
    jn="${ca}_${da}_${dma}_${ma}_s${seed}"
    [[ -n "$rs" ]] && jn="${ca}${rs}_${da}_${dma}_${ma}_s${seed}"
    
    result=$(qsub -N "$jn" -q "$queue" -l select=1:ncpus=4:mem=8gb -l walltime="${walltime}" -v "$env_vars" "$PBS_SCRIPT" 2>&1) || true
    
    if [[ "$result" == *".spcc-adm1"* ]]; then
        submitted=$((submitted + 1))
        queue_current[$queue]=$((${queue_current[$queue]} + 1))
        jid=$(echo "$result" | grep -oP '\d+(?=\.spcc)')
        echo "  ✅ [$submitted] $jn → $jid ($queue)" | tee -a "$SUBMIT_LOG"
    else
        echo "$cond $mode $dist $dom $seed $ratio" >> "$STILL_REMAINING"
        failed=$((failed + 1))
        queue_current[$queue]=${queue_max[$queue]}
    fi
done < "$MISSING_FILE"

still=$(wc -l < "$STILL_REMAINING")
echo "" | tee -a "$SUBMIT_LOG"
echo "============================================================" | tee -a "$SUBMIT_LOG"
echo "  Submitted: $submitted" | tee -a "$SUBMIT_LOG"
echo "  Remaining: $still" | tee -a "$SUBMIT_LOG"
echo "============================================================" | tee -a "$SUBMIT_LOG"

if (( still > 0 )); then
    # Update the missing file for next run
    cp "$STILL_REMAINING" "$MISSING_FILE"
    echo "" | tee -a "$SUBMIT_LOG"
    echo "⚠ $still jobs still remaining. Re-run when slots open:" | tee -a "$SUBMIT_LOG"
    echo "  bash scripts/hpc/jobs/domain_analysis/submit_missing_exp2.sh" | tee -a "$SUBMIT_LOG"
else
    echo "🎉 All Exp2 jobs submitted!" | tee -a "$SUBMIT_LOG"
fi

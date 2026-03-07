#!/bin/bash
# ============================================================
# Submit ALL remaining Exp2 jobs across all available queues
# Then loop to fill freed slots automatically
# ============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

PBS_SCRIPT="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
MISSING_FILE="/tmp/missing_norm.txt"
LOG_DIR="scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
DAEMON_LOG="${LOG_DIR}/exp2_daemon_$(date +%Y%m%d_%H%M%S).log"

# All usable queues for RF (4 cpus, 8gb, 10h walltime)
# LARGE/LONG-L have min CPU constraints too high
QUEUES=("DEFAULT" "SINGLE" "SMALL" "LONG")
declare -A queue_max
queue_max[DEFAULT]=40
queue_max[SINGLE]=40
queue_max[SMALL]=30
queue_max[LONG]=15

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$DAEMON_LOG"
}

get_queue_usage() {
    for q in "${QUEUES[@]}"; do
        queue_current[$q]=$(qstat -u "$USER" 2>/dev/null | awk -v q="$q" '$3==q {c++} END {print c+0}')
    done
}

submit_batch() {
    if [[ ! -f "$MISSING_FILE" ]] || [[ ! -s "$MISSING_FILE" ]]; then
        log "No remaining jobs. All done!"
        return 1
    fi

    declare -A queue_current
    get_queue_usage

    local available=0
    for q in "${QUEUES[@]}"; do
        local slots=$(( ${queue_max[$q]} - ${queue_current[$q]} ))
        (( slots < 0 )) && slots=0
        available=$((available + slots))
        log "  $q: ${queue_current[$q]}/${queue_max[$q]} ($slots free)"
    done

    if (( available == 0 )); then
        log "No slots available."
        return 0
    fi

    local remaining=$(wc -l < "$MISSING_FILE")
    log "Remaining jobs: $remaining, Available slots: $available"

    local submitted=0
    local still_remaining="/tmp/still_missing_exp2_$$.txt"
    > "$still_remaining"

    while IFS=' ' read -r cond mode dist dom seed ratio; do
        # Find available queue
        local queue=""
        for q in "${QUEUES[@]}"; do
            if (( ${queue_current[$q]} < ${queue_max[$q]} )); then
                queue="$q"
                break
            fi
        done

        if [[ -z "$queue" ]]; then
            echo "$cond $mode $dist $dom $seed $ratio" >> "$still_remaining"
            continue
        fi

        # Resources
        local walltime="10:00:00"
        [[ "$cond" == "balanced_rf" ]] && walltime="24:00:00"

        # Env vars
        local env_vars="CONDITION=${cond},MODE=${mode},DISTANCE=${dist},DOMAIN=${dom},SEED=${seed}"
        [[ -n "$ratio" ]] && env_vars="${env_vars},RATIO=${ratio}"

        # Job name
        local ca ma da dma rs jn
        case "$cond" in baseline) ca="bs";; smote_plain) ca="sp";; smote) ca="sm";; undersample) ca="us";; balanced_rf) ca="bf";; esac
        case "$mode" in source_only) ma="so";; target_only) ma="to";; mixed) ma="mx";; *) ma="xx";; esac
        case "$dist" in mmd) da="mmd";; dtw) da="dtw";; wasserstein) da="was";; esac
        case "$dom" in in_domain) dma="in";; out_domain) dma="ou";; esac
        rs=""; [[ -n "$ratio" ]] && rs=$(echo "$ratio" | sed 's/0\./r/')
        jn="${ca}_${da}_${dma}_${ma}_s${seed}"
        [[ -n "$rs" ]] && jn="${ca}${rs}_${da}_${dma}_${ma}_s${seed}"

        local result
        result=$(qsub -N "$jn" -q "$queue" -l select=1:ncpus=4:mem=8gb -l walltime="${walltime}" -v "$env_vars" "$PBS_SCRIPT" 2>&1) || true

        if [[ "$result" == *".spcc-adm1"* ]]; then
            submitted=$((submitted + 1))
            queue_current[$queue]=$((${queue_current[$queue]} + 1))
            local jid=$(echo "$result" | grep -oP '\d+(?=\.spcc)')
            log "  ✅ [$submitted] $jn → $jid ($queue)"
        else
            echo "$cond $mode $dist $dom $seed $ratio" >> "$still_remaining"
            queue_current[$queue]=${queue_max[$queue]}
        fi
    done < "$MISSING_FILE"

    # Update missing file
    cp "$still_remaining" "$MISSING_FILE"
    rm -f "$still_remaining"

    local still=$(wc -l < "$MISSING_FILE")
    log "Batch done: submitted=$submitted, remaining=$still"

    if (( still == 0 )); then
        log "🎉 All Exp2 jobs submitted!"
        return 1
    fi
    return 0
}

# === Main ===
log "============================================================"
log "Exp2 auto-submit daemon started (PID $$)"
log "============================================================"

# Initial batch
submit_batch
status=$?

if (( status == 1 )); then
    exit 0
fi

# Loop: check every 5 minutes
INTERVAL=300
log "Starting periodic check (every ${INTERVAL}s)..."

while true; do
    sleep $INTERVAL

    remaining=$(wc -l < "$MISSING_FILE" 2>/dev/null || echo 0)
    if (( remaining == 0 )); then
        log "🎉 All jobs submitted. Daemon exiting."
        break
    fi

    log "--- Periodic check (remaining: $remaining) ---"
    submit_batch
    status=$?
    if (( status == 1 )); then
        break
    fi
done

log "Daemon finished."

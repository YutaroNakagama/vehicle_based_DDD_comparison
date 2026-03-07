#!/bin/bash
# ============================================================
# Experiment 2: Continuous Monitor & Submit Daemon
# ============================================================
# Periodically checks queue slots and submits remaining Exp2
# extra-seed jobs as slots become available.
#
# Tracks submitted jobs across all previous logs to avoid duplicates.
#
# Usage:
#   nohup bash scripts/hpc/launchers/exp2_continuous_submit.sh &
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

SEEDS=(0 7 2024)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODES=("target_only" "source_only" "mixed")

# Queue limits (hard per-user limits on KAGAYAKI)
SINGLE_MAX=40
DEFAULT_MAX=10
SMALL_MAX=10
LONG_MAX=15

POLL_INTERVAL=90  # seconds between checks

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/exp2_continuous_${TIMESTAMP}.log"

# ============================================================
# Build full job list and track what's already submitted
# ============================================================
declare -a ALL_JOBS=()

for MODE in "${MODES[@]}"; do
    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # Baseline
                ALL_JOBS+=("baseline||$MODE|$DIST|$DOM|$SEED")
                # Ratio-based
                for RATIO in "${RATIOS[@]}"; do
                    for COND in smote_plain smote undersample; do
                        ALL_JOBS+=("$COND|$RATIO|$MODE|$DIST|$DOM|$SEED")
                    done
                done
                # Balanced RF
                ALL_JOBS+=("balanced_rf||$MODE|$DIST|$DOM|$SEED")
            done
        done
    done
done

echo "Total jobs in plan: ${#ALL_JOBS[@]}"

# Collect all previously submitted keys from all log files
collect_submitted_keys() {
    declare -gA SUBMITTED=()
    for logf in "$LOG_DIR"/exp2_extra_seeds_submit.log "$LOG_DIR"/exp2_extra_seeds_bulk_*.log "$LOG_FILE"; do
        [[ -f "$logf" ]] || continue
        while IFS= read -r line; do
            # OK:train:COND:MODE:DIST:DOM:rRATIO:SEED:JOBID  or  OK:train:COND:MODE:DIST:DOM:SEED:JOBID
            key=$(echo "$line" | sed 's/^OK://' | sed 's/:[0-9]*\.spcc.*$//')
            SUBMITTED["$key"]=1
        done < <(grep "^OK:" "$logf" 2>/dev/null)
    done
    # Also check [SUBMIT] lines from old log
    local old_log="$LOG_DIR/exp2_extra_seeds_submit.log"
    if [[ -f "$old_log" ]]; then
        while IFS= read -r line; do
            cleaned=$(echo "$line" | sed 's/.*\[SUBMIT\] //' | sed 's/ →.*//')
            cond=$(echo "$cleaned" | awk -F' \\| ' '{print $1}' | sed 's/ r[0-9.]*//')
            mode=$(echo "$cleaned" | awk -F' \\| ' '{print $2}' | tr -d ' ')
            dist=$(echo "$cleaned" | awk -F' \\| ' '{print $3}' | tr -d ' ')
            dom=$(echo "$cleaned" | awk -F' \\| ' '{print $4}' | tr -d ' ')
            seed_part=$(echo "$cleaned" | awk -F' \\| ' '{print $5}' | tr -d ' ')
            seed=${seed_part#s}
            ratio_part=$(echo "$cleaned" | awk -F' \\| ' '{print $1}' | grep -oP 'r[0-9.]+' || echo "")
            if [[ -n "$ratio_part" ]]; then
                key="train:${cond}:${mode}:${dist}:${dom}:${ratio_part}:${seed}"
            else
                key="train:${cond}:${mode}:${dist}:${dom}:${seed}"
            fi
            SUBMITTED["$key"]=1
        done < <(grep "\[SUBMIT\]" "$old_log" 2>/dev/null)
    fi
}

# ============================================================
# Queue helpers
# ============================================================
get_queue_count() {
    qstat -u s2240011 2>/dev/null | awk -v q="$1" '/s2240011/ && $3==q{n++} END{print n+0}'
}

get_queue_slots() {
    local q="$1"
    local max count
    case "$q" in
        SINGLE)  max=$SINGLE_MAX ;;
        DEFAULT) max=$DEFAULT_MAX ;;
        SMALL)   max=$SMALL_MAX ;;
        LONG)    max=$LONG_MAX ;;
    esac
    count=$(get_queue_count "$q")
    echo $(( max - count ))
}

get_resources() {
    local cond="$1"
    case "$cond" in
        balanced_rf)       echo "ncpus=8:mem=12gb 10:00:00 LONG" ;;
        smote|smote_plain) echo "ncpus=4:mem=10gb 10:00:00 NORMAL" ;;
        *)                 echo "ncpus=4:mem=8gb 10:00:00 NORMAL" ;;
    esac
}

make_job_key() {
    local cond="$1" ratio="$2" mode="$3" dist="$4" dom="$5" seed="$6"
    if [[ -n "$ratio" ]]; then
        echo "train:${cond}:${mode}:${dist}:${dom}:r${ratio}:${seed}"
    else
        echo "train:${cond}:${mode}:${dist}:${dom}:${seed}"
    fi
}

# ============================================================
# Main loop
# ============================================================
echo "============================================================" | tee -a "$LOG_FILE"
echo "  Exp2 Continuous Submit Daemon — $(date)" | tee -a "$LOG_FILE"
echo "  Poll interval: ${POLL_INTERVAL}s" | tee -a "$LOG_FILE"
echo "  Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Round-robin index for SINGLE/DEFAULT/SMALL
RR_IDX=0
RR_QUEUES=("SINGLE" "DEFAULT" "SMALL")

while true; do
    collect_submitted_keys
    submitted_count=${#SUBMITTED[@]}

    # Count remaining
    remaining=0
    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r cond ratio mode dist dom seed <<< "$job_spec"
        key=$(make_job_key "$cond" "$ratio" "$mode" "$dist" "$dom" "$seed")
        if [[ -z "${SUBMITTED[$key]:-}" ]]; then
            ((remaining++))
        fi
    done

    if (( remaining == 0 )); then
        echo "[$(date +%H:%M:%S)] All 432 jobs submitted! Exiting." | tee -a "$LOG_FILE"
        break
    fi

    # Check available slots
    s_slots=$(get_queue_slots "SINGLE")
    d_slots=$(get_queue_slots "DEFAULT")
    m_slots=$(get_queue_slots "SMALL")
    l_slots=$(get_queue_slots "LONG")
    normal_slots=$(( s_slots + d_slots + m_slots ))

    echo "[$(date +%H:%M:%S)] Submitted: $submitted_count/432 | Remaining: $remaining | Slots: SINGLE=$s_slots DEFAULT=$d_slots SMALL=$m_slots LONG=$l_slots" | tee -a "$LOG_FILE"

    if (( normal_slots <= 0 && l_slots <= 0 )); then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Submit jobs that fit available slots
    normal_used=0
    long_used=0

    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r cond ratio mode dist dom seed <<< "$job_spec"
        key=$(make_job_key "$cond" "$ratio" "$mode" "$dist" "$dom" "$seed")

        # Skip already submitted
        if [[ -n "${SUBMITTED[$key]:-}" ]]; then
            continue
        fi

        res=$(get_resources "$cond")
        ncpus_mem=$(echo "$res" | cut -d' ' -f1)
        walltime=$(echo "$res" | cut -d' ' -f2)
        qtype=$(echo "$res" | cut -d' ' -f3)

        if [[ "$qtype" == "LONG" ]]; then
            if (( long_used >= l_slots )); then
                continue
            fi
            queue="LONG"
        else
            if (( normal_used >= normal_slots )); then
                continue
            fi
            # Round-robin across SINGLE/DEFAULT/SMALL
            # Find a queue with room
            local_found=0
            for _try in 0 1 2; do
                q="${RR_QUEUES[$RR_IDX]}"
                RR_IDX=$(( (RR_IDX + 1) % 3 ))
                q_avail=$(get_queue_slots "$q")
                if (( q_avail > 0 )); then
                    queue="$q"
                    local_found=1
                    break
                fi
            done
            if (( local_found == 0 )); then
                continue
            fi
        fi

        # Build job name
        if [[ -n "$ratio" ]]; then
            rr="${ratio/0./}"
            jname="e2_${cond:0:2}r${rr}_${mode:0:1}${dist:0:2}${dom:0:1}_s${seed}"
        elif [[ "$cond" == "balanced_rf" ]]; then
            jname="e2_bf_${mode:0:1}${dist:0:2}${dom:0:1}_s${seed}"
        else
            jname="e2_bs_${mode:0:1}${dist:0:2}${dom:0:1}_s${seed}"
        fi

        cmd="qsub -N $jname -l select=1:$ncpus_mem -l walltime=$walltime -q $queue"
        if [[ -n "$ratio" ]]; then
            cmd="$cmd -v CONDITION=$cond,MODE=$mode,DISTANCE=$dist,DOMAIN=$dom,RATIO=$ratio,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        else
            cmd="$cmd -v CONDITION=$cond,MODE=$mode,DISTANCE=$dist,DOMAIN=$dom,SEED=$seed,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
        fi
        cmd="$cmd $TRAIN_SCRIPT"

        job_id=$(eval "$cmd" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[SUBMIT] [$queue] $cond ${ratio:+r$ratio }| $mode | $dist | $dom | s$seed → $job_id" | tee -a "$LOG_FILE"
            echo "OK:$key:$job_id" >> "$LOG_FILE"
            SUBMITTED["$key"]=1

            if [[ "$qtype" == "LONG" ]]; then
                ((long_used++))
            else
                ((normal_used++))
            fi
            sleep 0.2
        else
            # Queue full — stop trying this queue type
            if [[ "$job_id" == *"exceed"*"limit"* ]]; then
                if [[ "$qtype" == "LONG" ]]; then
                    l_slots=0
                else
                    # Mark the specific queue as full
                    case "$queue" in
                        SINGLE)  s_slots=0 ;;
                        DEFAULT) d_slots=0 ;;
                        SMALL)   m_slots=0 ;;
                    esac
                    normal_slots=$(( s_slots + d_slots + m_slots ))
                fi
            else
                echo "[ERROR] $cond ${ratio:+r$ratio }| $mode | $dist | $dom | s$seed → $job_id" | tee -a "$LOG_FILE"
            fi
        fi
    done

    sleep "$POLL_INTERVAL"
done

echo ""
echo "============================================================" | tee -a "$LOG_FILE"
echo "  All jobs submitted! — $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

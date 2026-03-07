#!/bin/bash
# ============================================================
# Experiment 2: 10-Seed Continuous Submit Daemon (No BalancedRF)
# ============================================================
# Periodically checks queue slots and submits remaining Exp2
# jobs as slots become available.
#
# Seeds: 42, 123, 0, 7, 2024, 1, 13, 256, 512, 1337
# Conditions: baseline, smote_plain, smote (sw_smote), undersample
# (BalancedRF excluded)
#
# Tracks submitted jobs across ALL previous exp2 logs to avoid duplicates.
#
# Usage:
#   nohup bash scripts/hpc/launchers/exp2_10seeds_submit.sh \
#     > scripts/hpc/logs/domain/exp2_10seeds_output.log 2>&1 &
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

# === Extra 8 seeds (42 and 123 already 100% complete) ===
SEEDS=(0 7 2024 1 13 256 512 1337)

RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("out_domain" "in_domain")
MODES=("target_only" "source_only" "mixed")

# Conditions (NO balanced_rf)
CONDITIONS_BASELINE=("baseline")
CONDITIONS_RATIO=("smote_plain" "smote" "undersample")

# Queue limits — max_queued per user (from qmgr config)
# max_run (simultaneous) is lower, but PBS handles that automatically.
# We fill up to max_queued so jobs wait in queue and start immediately
# when execution slots open, instead of polling every 90s.
SINGLE_MAX=40   # max_queued=40, max_run=10
DEFAULT_MAX=40  # max_queued=40, max_run=20
SMALL_MAX=30    # max_queued=30, max_run=7
LONG_MAX=15     # max_queued=15, max_run=2

POLL_INTERVAL=120  # seconds between checks (less urgent now that queues are pre-filled)

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/exp2_10seeds_${TIMESTAMP}.log"

# ============================================================
# Build full job list (no balanced_rf)
# ============================================================
declare -a ALL_JOBS=()

for MODE in "${MODES[@]}"; do
    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # Baseline (no ratio)
                ALL_JOBS+=("baseline||$MODE|$DIST|$DOM|$SEED")
                # Ratio-based conditions
                for RATIO in "${RATIOS[@]}"; do
                    for COND in "${CONDITIONS_RATIO[@]}"; do
                        ALL_JOBS+=("$COND|$RATIO|$MODE|$DIST|$DOM|$SEED")
                    done
                done
            done
        done
    done
done

TOTAL_JOBS=${#ALL_JOBS[@]}
echo "Total jobs in plan: $TOTAL_JOBS (10 seeds × 4 conditions × no BalancedRF)"

# ============================================================
# Collect all previously submitted keys from ALL exp2 log files
# ============================================================
collect_submitted_keys() {
    declare -gA SUBMITTED=()

    # Scan all exp2 log files for OK: lines
    for logf in "$LOG_DIR"/exp2_*.log; do
        [[ -f "$logf" ]] || continue
        while IFS= read -r line; do
            # Format: OK:train:COND:MODE:DIST:DOM:rRATIO:SEED:JOBID
            #      or OK:train:COND:MODE:DIST:DOM:SEED:JOBID
            key=$(echo "$line" | sed 's/^OK://' | sed 's/:[0-9]*\.spcc.*$//')
            SUBMITTED["$key"]=1
        done < <(grep "^OK:" "$logf" 2>/dev/null)
    done

    # Also check [SUBMIT] lines from older log formats
    for logf in "$LOG_DIR"/exp2_extra_seeds_submit.log "$LOG_DIR"/exp2_daemon_*.log; do
        [[ -f "$logf" ]] || continue
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
        done < <(grep "\[SUBMIT\]" "$logf" 2>/dev/null)
    done

    # Also check the rerun/fill/remaining logs (arrow format: → JOBID)
    for logf in "$LOG_DIR"/exp2_rerun_*.log "$LOG_DIR"/exp2_remaining_*.log "$LOG_DIR"/exp2_fill_*.log; do
        [[ -f "$logf" ]] || continue
        while IFS= read -r line; do
            # ✅ [1] bs_mmd_in_so_s42 → 14876092
            if [[ "$line" == *"→"* ]] && [[ "$line" == *"_s"* ]]; then
                tag=$(echo "$line" | grep -oP '\] \K\S+(?= →)')
                # Parse tag: e.g. bs_mmd_in_so_s42, smr1_was_in_to_s42
                seed=$(echo "$tag" | grep -oP 's\K\d+$')
                [[ -n "$seed" ]] || continue
                # We can't perfectly reconstruct the key from the short tag,
                # but we can check if a .OU log exists with success marker
                jobid=$(echo "$line" | grep -oP '→ \K\d+')
                if [[ -n "$jobid" ]] && [[ -f "$LOG_DIR/${jobid}.spcc-adm1.OU" ]]; then
                    if grep -q "\[DONE\] Job completed successfully" "$LOG_DIR/${jobid}.spcc-adm1.OU" 2>/dev/null; then
                        # Extract TAG from the .OU file to build proper key
                        real_tag=$(grep '^TAG:' "$LOG_DIR/${jobid}.spcc-adm1.OU" 2>/dev/null | head -1 | awk '{print $2}')
                        if [[ -n "$real_tag" ]]; then
                            # Parse: baseline_domain_knn_mmd_in_domain_source_only_split2_s42
                            local_seed=$(echo "$real_tag" | grep -oP '_s\K\d+$')
                            # Build the key matching our format
                            local_mode=$(echo "$real_tag" | grep -oP '(source_only|target_only|mixed)')
                            local_dist=$(echo "$real_tag" | grep -oP 'knn_\K(mmd|dtw|wasserstein)')
                            local_dom=$(echo "$real_tag" | grep -oP '(in_domain|out_domain)')
                            local_ratio=$(echo "$real_tag" | grep -oP 'ratio\K[0-9.]+' || echo "")
                            # Determine condition
                            if echo "$real_tag" | grep -q "^baseline_domain"; then
                                local_cond="baseline"
                            elif echo "$real_tag" | grep -q "^smote_plain"; then
                                local_cond="smote_plain"
                            elif echo "$real_tag" | grep -q "^smote_knn\|^imbalv3\|^smote_subjectwise\|^swsmote"; then
                                local_cond="smote"
                            elif echo "$real_tag" | grep -q "^undersample\|^undersample_rus"; then
                                local_cond="undersample"
                            elif echo "$real_tag" | grep -q "^balanced_rf"; then
                                local_cond="balanced_rf"
                            else
                                continue
                            fi
                            if [[ -n "$local_ratio" ]]; then
                                key="train:${local_cond}:${local_mode}:${local_dist}:${local_dom}:r${local_ratio}:${local_seed}"
                            else
                                key="train:${local_cond}:${local_mode}:${local_dist}:${local_dom}:${local_seed}"
                            fi
                            SUBMITTED["$key"]=1
                        fi
                    fi
                fi
            fi
        done < "$logf"
    done
}

# ============================================================
# Queue helpers
# ============================================================
get_queue_count() {
    qstat -u s2240011 2>/dev/null | awk -v q="$1" '/s2240011/ && $3==q{n++} END{print n+0}'
}

get_queue_slots() {
    local q="$1"
    local max
    case "$q" in
        SINGLE)  max=$SINGLE_MAX ;;
        DEFAULT) max=$DEFAULT_MAX ;;
        SMALL)   max=$SMALL_MAX ;;
        LONG)    max=$LONG_MAX ;;
    esac
    local count
    count=$(get_queue_count "$q")
    echo $(( max - count ))
}

get_resources() {
    local cond="$1"
    case "$cond" in
        smote|smote_plain) echo "ncpus=4:mem=10gb 20:00:00 NORMAL" ;;
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
echo "  Exp2 10-Seed Submit Daemon — $(date)" | tee -a "$LOG_FILE"
echo "  Seeds: ${SEEDS[*]}" | tee -a "$LOG_FILE"
echo "  Conditions: baseline, smote_plain, smote, undersample" | tee -a "$LOG_FILE"
echo "  (BalancedRF excluded)" | tee -a "$LOG_FILE"
echo "  Poll interval: ${POLL_INTERVAL}s" | tee -a "$LOG_FILE"
echo "  Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "  Total plan: $TOTAL_JOBS jobs" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# Round-robin index for SINGLE/DEFAULT/SMALL/LONG
RR_IDX=0
RR_QUEUES=("SINGLE" "DEFAULT" "SMALL" "LONG")

while true; do
    collect_submitted_keys
    submitted_count=${#SUBMITTED[@]}

    # Count remaining (only for non-balanced_rf jobs)
    remaining=0
    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r cond ratio mode dist dom seed <<< "$job_spec"
        key=$(make_job_key "$cond" "$ratio" "$mode" "$dist" "$dom" "$seed")
        if [[ -z "${SUBMITTED[$key]:-}" ]]; then
            ((remaining++))
        fi
    done

    if (( remaining == 0 )); then
        echo "[$(date +%H:%M:%S)] All $TOTAL_JOBS jobs submitted! Exiting." | tee -a "$LOG_FILE"
        break
    fi

    # Check available slots
    s_slots=$(get_queue_slots "SINGLE")
    d_slots=$(get_queue_slots "DEFAULT")
    m_slots=$(get_queue_slots "SMALL")
    l_slots=$(get_queue_slots "LONG")
    normal_slots=$(( s_slots + d_slots + m_slots + l_slots ))

    echo "[$(date +%H:%M:%S)] Submitted: ~$submitted_count | Remaining: $remaining/$TOTAL_JOBS | Slots: SINGLE=$s_slots DEFAULT=$d_slots SMALL=$m_slots LONG=$l_slots" | tee -a "$LOG_FILE"

    if (( normal_slots <= 0 )); then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Submit jobs that fit available slots
    normal_used=0

    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r cond ratio mode dist dom seed <<< "$job_spec"
        key=$(make_job_key "$cond" "$ratio" "$mode" "$dist" "$dom" "$seed")

        # Skip already submitted
        if [[ -n "${SUBMITTED[$key]:-}" ]]; then
            continue
        fi

        # Check if we still have slots
        if (( normal_used >= normal_slots )); then
            break
        fi

        res=$(get_resources "$cond")
        ncpus_mem=$(echo "$res" | cut -d' ' -f1)
        walltime=$(echo "$res" | cut -d' ' -f2)

        # Round-robin across SINGLE/DEFAULT/SMALL/LONG
        local_found=0
        for _try in 0 1 2 3; do
            q="${RR_QUEUES[$RR_IDX]}"
            RR_IDX=$(( (RR_IDX + 1) % 4 ))
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

        # Build job name (short, fits PBS limit)
        if [[ -n "$ratio" ]]; then
            rr="${ratio/0./}"
            jname="e2_${cond:0:2}r${rr}_${mode:0:1}${dist:0:2}${dom:0:1}_s${seed}"
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
            ((normal_used++))
            sleep 0.2
        else
            if [[ "$job_id" == *"exceed"*"limit"* ]]; then
                case "$queue" in
                    SINGLE)  s_slots=0 ;;
                    DEFAULT) d_slots=0 ;;
                    SMALL)   m_slots=0 ;;
                    LONG)    l_slots=0 ;;
                esac
                normal_slots=$(( s_slots + d_slots + m_slots + l_slots ))
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

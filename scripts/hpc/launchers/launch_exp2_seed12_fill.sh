#!/bin/bash
# ============================================================
# Experiment 2: 12-Seed 完全化ランチャー
# ============================================================
# 欠損 seed の補完 + 12番目の seed (999) を全セルに追加
#
# 作業内訳:
#   1. 欠損 seed 補完 (seed 3: 78, seed 1: 1, seed 7: 1) = 80 ジョブ
#   2. 新 seed 999 (全126セル)                            = 126 ジョブ
#   合計: 206 ジョブ
#
# 完了後の seed 構成: [0,1,3,7,13,42,123,256,512,999,1337,2024] (n=12)
#
# Step 1: ジョブリスト生成
#   python scripts/hpc/launchers/gen_exp2_seed12_joblist.py
#
# Step 2: ランチャー実行
#   bash scripts/hpc/launchers/launch_exp2_seed12_fill.sh --dry-run
#   nohup bash scripts/hpc/launchers/launch_exp2_seed12_fill.sh \
#     > scripts/hpc/logs/domain/exp2_seed12_output.log 2>&1 &
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

TRAIN_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
JOBLIST_FILE="$PROJECT_ROOT/scripts/hpc/launchers/exp2_seed12_joblist.txt"

N_TRIALS=100
RANKING="knn"

# Queue rotation & limits
QUEUES=("SINGLE" "DEFAULT" "SMALL")
declare -A queue_max
queue_max[SINGLE]=40
queue_max[DEFAULT]=40
queue_max[SMALL]=30

POLL_INTERVAL=120
DRY_RUN=false

export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Generate job list from CSV data if not exists
if [[ ! -f "$JOBLIST_FILE" ]]; then
    echo "[INFO] Generating job list from CSV data..."
    python scripts/hpc/launchers/gen_exp2_seed12_joblist.py --output "$JOBLIST_FILE"
fi

# Read job list (skip comment lines)
declare -a ALL_JOBS=()
while IFS= read -r line; do
    [[ "$line" =~ ^# ]] && continue
    [[ -z "$line" ]] && continue
    ALL_JOBS+=("$line")
done < "$JOBLIST_FILE"

TOTAL_JOBS=${#ALL_JOBS[@]}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/exp2_seed12_fill_${TIMESTAMP}.log"

echo "============================================================" | tee -a "$LOG_FILE"
echo "  Exp2 12-Seed Fill Launcher — $(date)" | tee -a "$LOG_FILE"
echo "  Job list: $JOBLIST_FILE" | tee -a "$LOG_FILE"
echo "  Total plan: $TOTAL_JOBS jobs" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# ============================================================
# Collect previously submitted keys from exp2 logs
# ============================================================
collect_submitted_keys() {
    declare -gA SUBMITTED=()
    for logf in "$LOG_DIR"/exp2_*.log; do
        [[ -f "$logf" ]] || continue
        while IFS= read -r line; do
            key=$(echo "$line" | sed 's/^OK://' | sed 's/:[0-9]*\.spcc.*$//')
            SUBMITTED["$key"]=1
        done < <(grep "^OK:" "$logf" 2>/dev/null)
    done
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
# Queue helpers
# ============================================================
get_queue_count() {
    qstat -u "$USER" 2>/dev/null | awk -v q="$1" '/'"$USER"'/ && $3==q{n++} END{print n+0}'
}

get_queue_slots() {
    local q="$1"
    local max=${queue_max[$q]}
    local count
    count=$(get_queue_count "$q")
    echo $(( max - count ))
}

get_resources() {
    local cond="$1"
    case "$cond" in
        smote|smote_plain) echo "ncpus=4:mem=10gb 20:00:00" ;;
        *)                 echo "ncpus=4:mem=8gb 10:00:00" ;;
    esac
}

make_job_name() {
    local cond="$1" mode="$2" dist="$3" dom="$4" seed="$5" ratio="${6:-}"
    local ca ma da
    case "$cond" in
        baseline)     ca="bs" ;;
        smote_plain)  ca="sp" ;;
        smote)        ca="sm" ;;
        undersample)  ca="us" ;;
    esac
    case "$mode" in
        source_only) ma="so" ;;
        target_only) ma="to" ;;
        mixed)       ma="mx" ;;
    esac
    case "$dom" in
        in_domain)  da="in" ;;
        out_domain) da="ou" ;;
    esac
    if [[ -n "$ratio" && "$ratio" != "" ]]; then
        local rr="${ratio/0./r}"
        echo "e2_${ca}${rr}_${dist}_${da}_${ma}_s${seed}"
    else
        echo "e2_${ca}_${dist}_${da}_${ma}_s${seed}"
    fi
}

# Queue round-robin index
RR_IDX=0

# ============================================================
# Main loop
# ============================================================
submitted_total=0
failed_total=0

while true; do
    collect_submitted_keys

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
        echo "[$(date +%H:%M:%S)] All $TOTAL_JOBS jobs submitted! Done." | tee -a "$LOG_FILE"
        break
    fi

    # Check available slots
    s_slots=$(get_queue_slots "SINGLE")
    d_slots=$(get_queue_slots "DEFAULT")
    m_slots=$(get_queue_slots "SMALL")
    total_slots=$(( s_slots + d_slots + m_slots ))

    echo "[$(date +%H:%M:%S)] Submitted: $submitted_total | Remaining: $remaining/$TOTAL_JOBS | Slots: S=$s_slots D=$d_slots M=$m_slots" | tee -a "$LOG_FILE"

    if $DRY_RUN; then
        for job_spec in "${ALL_JOBS[@]}"; do
            IFS='|' read -r cond ratio mode dist dom seed <<< "$job_spec"
            key=$(make_job_key "$cond" "$ratio" "$mode" "$dist" "$dom" "$seed")
            if [[ -z "${SUBMITTED[$key]:-}" ]]; then
                jname=$(make_job_name "$cond" "$mode" "$dist" "$dom" "$seed" "$ratio")
                echo "  [DRY] $jname ($cond ${ratio:+r$ratio }| $mode | $dist | $dom | s$seed)" | tee -a "$LOG_FILE"
            fi
        done
        echo "[DRY-RUN] Would submit $remaining jobs. Exiting." | tee -a "$LOG_FILE"
        break
    fi

    if (( total_slots <= 0 )); then
        sleep "$POLL_INTERVAL"
        continue
    fi

    # Submit jobs that fit available slots
    batch_submitted=0
    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r cond ratio mode dist dom seed <<< "$job_spec"
        key=$(make_job_key "$cond" "$ratio" "$mode" "$dist" "$dom" "$seed")

        [[ -n "${SUBMITTED[$key]:-}" ]] && continue

        if (( batch_submitted >= total_slots )); then
            break
        fi

        res=$(get_resources "$cond")
        ncpus_mem=$(echo "$res" | cut -d' ' -f1)
        walltime=$(echo "$res" | cut -d' ' -f2)

        # Round-robin queue selection
        local_found=0
        for _try in 0 1 2; do
            q="${QUEUES[$RR_IDX]}"
            RR_IDX=$(( (RR_IDX + 1) % ${#QUEUES[@]} ))
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

        jname=$(make_job_name "$cond" "$mode" "$dist" "$dom" "$seed" "$ratio")

        env_vars="CONDITION=${cond},MODE=${mode},DISTANCE=${dist},DOMAIN=${dom},SEED=${seed},N_TRIALS=${N_TRIALS},RANKING=${RANKING},RUN_EVAL=true"
        [[ -n "$ratio" ]] && env_vars="${env_vars},RATIO=${ratio}"

        result=$(qsub \
            -N "$jname" \
            -q "$queue" \
            -l "select=1:${ncpus_mem}" \
            -l "walltime=${walltime}" \
            -v "$env_vars" \
            "$TRAIN_SCRIPT" 2>&1) || true

        if [[ "$result" == *".spcc-adm1"* ]]; then
            submitted_total=$((submitted_total + 1))
            batch_submitted=$((batch_submitted + 1))
            echo "OK:${key}:${result}" >> "$LOG_FILE"
            echo "  ✅ [$submitted_total] $jname → $result ($queue)" | tee -a "$LOG_FILE"
        else
            failed_total=$((failed_total + 1))
            echo "  ❌ $jname ($queue): $result" | tee -a "$LOG_FILE"
        fi
    done

    if (( batch_submitted == 0 )); then
        sleep "$POLL_INTERVAL"
    else
        sleep 5
    fi
done

echo ""
echo "============================================================"
echo "  Summary"
echo "  Submitted: $submitted_total"
echo "  Failed:    $failed_total"
echo "  Log:       $LOG_FILE"
echo "============================================================"

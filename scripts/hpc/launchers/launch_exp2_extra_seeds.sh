#!/bin/bash
# ============================================================
# Experiment 2: additional 3 seeds submit launcher (n=10 → n=13)
# ============================================================
# Existing 10 seeds [0,1,7,13,42,123,256,512,1337,2024] in addition,
# 3seeds [3,99,777] adding to n=13 aiming to pass Bonferroni correction with
#
# Experiment C results:
#   n=10 → min p_perm = 1/1024 ≈ 0.00098 > Bonferroni α' = 0.00119 → FAIL
#   n=13 → min p_perm = 1/8192 ≈ 0.00012 << Bonferroni α' = 0.00119 → PASS
#
# Conditions (4 base x 7 variants, excluding balanced_rf):
#   baseline      (no ratio)   → 3 modes × 3 dists × 2 doms × 3 seeds =  54
#   smote_plain   (r0.1, r0.5) → 3 × 3 × 2 × 2 × 3                   = 108
#   smote/sw      (r0.1, r0.5) → 3 × 3 × 2 × 2 × 3                   = 108
#   undersample   (r0.1, r0.5) → 3 × 3 × 2 × 2 × 3                   = 108
#   Total: 378 jobs
#
# Usage:
#   bash scripts/hpc/launchers/launch_exp2_extra_seeds.sh --dry-run
#   nohup bash scripts/hpc/launchers/launch_exp2_extra_seeds.sh &
# ============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

PBS_SCRIPT="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

# New seeds (existing: 0,1,7,13,42,123,256,512,1337,2024)
SEEDS=(3 99 777)

DISTANCES=("dtw" "mmd" "wasserstein")
DOMAINS=("in_domain" "out_domain")
MODES=("source_only" "target_only" "mixed")
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"

# Resources
RF_WALLTIME="10:00:00"
RF_NCPUS=4
RF_MEM="8gb"
SMOTE_MEM="10gb"

# Queue rotation & limits
QUEUES=("DEFAULT" "SINGLE" "SMALL")
declare -A queue_max
queue_max[DEFAULT]=40
queue_max[SINGLE]=40
queue_max[SMALL]=30

WAIT_INTERVAL=60
DRY_RUN=false

# Workaround: /var/tmp and /tmp may be full on head node
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/exp2_extra_seeds_${TIMESTAMP}.log"
SUBMITTED_FILE="$LOG_DIR/exp2_extra_seeds_submitted_${TIMESTAMP}.txt"

# ============================================================
# Helper Functions
# ============================================================
declare -A queue_current
refresh_queue_counts() {
    for q in "${QUEUES[@]}"; do
        queue_current[$q]=$(qstat -u "$USER" 2>/dev/null | awk -v q="$q" '$3==q {count++} END {print count+0}')
    done
}

find_available_queue() {
    for q in "${QUEUES[@]}"; do
        if (( ${queue_current[$q]} < ${queue_max[$q]} )); then
            echo "$q"
            return 0
        fi
    done
    return 1
}

wait_for_queue_slot() {
    if $DRY_RUN; then echo "DEFAULT"; return 0; fi

    while true; do
        refresh_queue_counts
        local q
        q=$(find_available_queue) && { echo "$q"; return 0; }
        echo "  ⏳ All queues full, waiting ${WAIT_INTERVAL}s..." >&2
        sleep "$WAIT_INTERVAL"
    done
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
        echo "${ca}${rr}_${dist}_${da}_${ma}_s${seed}"
    else
        echo "${ca}_${dist}_${da}_${ma}_s${seed}"
    fi
}

# Get already running/queued job names
mapfile -t existing_jobs < <(qstat -u "$USER" 2>/dev/null | awk 'NR>5 {print $4}' | sed 's/\*$//' 2>/dev/null || true)

is_already_queued() {
    local job_name="$1"
    for ej in "${existing_jobs[@]:-}"; do
        [[ "$ej" == "$job_name" ]] && return 0
    done
    return 1
}

total=0
submitted=0
skipped=0
failed=0

submit_one() {
    local cond="$1" mode="$2" dist="$3" dom="$4" seed="$5" ratio="${6:-}"

    local job_name
    job_name=$(make_job_name "$cond" "$mode" "$dist" "$dom" "$seed" "$ratio")
    total=$((total + 1))

    # Skip if already queued
    if is_already_queued "$job_name"; then
        skipped=$((skipped + 1))
        return 0
    fi

    # Select memory
    local mem="$RF_MEM"
    [[ "$cond" == "smote" || "$cond" == "smote_plain" ]] && mem="$SMOTE_MEM"

    # Environment variables
    local env_vars="CONDITION=${cond},MODE=${mode},DISTANCE=${dist},DOMAIN=${dom},SEED=${seed},N_TRIALS=${N_TRIALS},RANKING=${RANKING},RUN_EVAL=true"
    [[ -n "$ratio" ]] && env_vars="${env_vars},RATIO=${ratio}"

    if $DRY_RUN; then
        echo "  [DRY] $job_name ($cond ${ratio:+r$ratio }| $mode | $dist | $dom | s$seed)"
        return 0
    fi

    local queue
    queue=$(wait_for_queue_slot)

    local result
    result=$(qsub \
        -N "$job_name" \
        -q "$queue" \
        -l select=1:ncpus=${RF_NCPUS}:mem=${mem} \
        -l walltime=${RF_WALLTIME} \
        -v "$env_vars" \
        "$PBS_SCRIPT" 2>&1) || true

    if [[ "$result" == *".spcc-adm1"* ]]; then
        submitted=$((submitted + 1))
        queue_current[$queue]=$((${queue_current[$queue]} + 1))
        local jid
        jid=$(echo "$result" | grep -oP '\d+')
        echo "SUBMITTED $jid $job_name $queue" >> "$SUBMITTED_FILE"
        echo "  ✅ [$submitted] $job_name → $jid ($queue)" | tee -a "$LOG_FILE"
    else
        failed=$((failed + 1))
        echo "  ❌ $job_name ($queue): $result" | tee -a "$LOG_FILE"
        echo "FAILED $job_name $result" >> "$SUBMITTED_FILE"
        # If queue error, mark it full so we rotate to next
        queue_current[$queue]=${queue_max[$queue]}
    fi

    sleep 0.3
}

# ============================================================
# Main
# ============================================================
refresh_queue_counts

echo "============================================================" | tee "$LOG_FILE"
echo "  Experiment 2: Extra Seeds (n=10 → n=13)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "  New seeds : ${SEEDS[*]}" | tee -a "$LOG_FILE"
echo "  Dry run   : $DRY_RUN" | tee -a "$LOG_FILE"
echo "  Date      : $(date)" | tee -a "$LOG_FILE"
echo "  Commit    : $(git rev-parse --short HEAD 2>/dev/null || echo unknown)" | tee -a "$LOG_FILE"
echo "  Queue load:" | tee -a "$LOG_FILE"
for q in "${QUEUES[@]}"; do
    echo "    $q: ${queue_current[$q]}/${queue_max[$q]}" | tee -a "$LOG_FILE"
done
echo "============================================================" | tee -a "$LOG_FILE"

for seed in "${SEEDS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "━━━ Seed: $seed ━━━" | tee -a "$LOG_FILE"

    for mode in "${MODES[@]}"; do
        for dist in "${DISTANCES[@]}"; do
            for dom in "${DOMAINS[@]}"; do

                # Baseline (no ratio)
                submit_one "baseline" "$mode" "$dist" "$dom" "$seed"

                # Ratio-based conditions
                for ratio in "${RATIOS[@]}"; do
                    submit_one "smote_plain" "$mode" "$dist" "$dom" "$seed" "$ratio"
                    submit_one "smote"       "$mode" "$dist" "$dom" "$seed" "$ratio"
                    submit_one "undersample" "$mode" "$dist" "$dom" "$seed" "$ratio"
                done

            done
        done
    done
done

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "  SUMMARY" | tee -a "$LOG_FILE"
echo "  Total combinations: $total" | tee -a "$LOG_FILE"
echo "  Already queued:     $skipped" | tee -a "$LOG_FILE"
echo "  Submitted now:      $submitted" | tee -a "$LOG_FILE"
echo "  Failed/remaining:   $failed" | tee -a "$LOG_FILE"
echo "  Log:                $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

if (( failed > 0 )); then
    echo "" | tee -a "$LOG_FILE"
    echo "⚠ $failed jobs failed. Re-run this script when queue slots open:" | tee -a "$LOG_FILE"
    echo "  bash $0" | tee -a "$LOG_FILE"
fi

#!/bin/bash
# ============================================================
# Prior research experiment non-baseline condition resubmission launcher (2026-02-07)
# ============================================================
# Resubmit after KeyError: 'Steering_Range' bug fix
#
# Target: SvmW, SvmA, Lstm × smote, smote_plain, undersample
#       × mmd, dtw, wasserstein × in_domain, out_domain
#       × source_only, target_only × seed 42, 123 × ratio 0.1, 0.5
#
# Total: 3 models × 3 conditions × 3 distances × 2 domains
#       × 2 modes × 2 seeds × 2 ratios = 432 jobs
#
# Queue distribution: DEFAULT, SINGLE, SMALL, LONG, SEMINAR
# ============================================================

set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"

MODELS=(SvmW SvmA Lstm)
CONDITIONS=(smote smote_plain undersample)
DISTANCES=(mmd dtw wasserstein)
DOMAINS=(in_domain out_domain)
MODES=(source_only target_only)
SEEDS=(42 123)
RATIOS=(0.1 0.5)
N_TRIALS=100
RANKING="knn"

# Queues to distribute across (round-robin)
QUEUES=(DEFAULT SINGLE SMALL LONG SEMINAR)
QUEUE_COUNT=${#QUEUES[@]}
queue_idx=0

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_prior_nonbaseline_${TIMESTAMP}.log"

JOB_COUNT=0
FAIL_COUNT=0

# Queue counters
declare -A Q_COUNT
for q in "${QUEUES[@]}"; do Q_COUNT[$q]=0; done

# Resource settings per model
get_resources() {
    local model="$1"
    case "$model" in
        SvmW) echo "ncpus=4:mem=8gb 08:00:00" ;;
        SvmA) echo "ncpus=4:mem=8gb 10:00:00" ;;
        Lstm) echo "ncpus=4:mem=16gb 12:00:00" ;;
    esac
}

submit_job() {
    local model="$1" condition="$2" distance="$3" domain="$4" mode="$5" seed="$6" ratio="$7"

    # Pick queue (round-robin)
    local queue="${QUEUES[$queue_idx]}"
    queue_idx=$(( (queue_idx + 1) % QUEUE_COUNT ))

    # Resources
    local res
    res=$(get_resources "$model")
    local ncpus_mem="${res% *}"
    local walltime="${res#* }"

    # Short job name: Sv_sm_mi_s_r01_42
    local m_short="${model:0:2}"
    local c_short="${condition:0:2}"
    local dist_short="${distance:0:1}"
    local dom_short="${domain:0:1}"
    local mode_short="${mode:0:1}"
    local r_short=$(echo "$ratio" | tr -d '.')
    local job_name="p${m_short}_${c_short}_${dist_short}${dom_short}_${mode_short}_r${r_short}_s${seed}"

    # Environment variables
    local env_vars="MODEL=$model,CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,RATIO=$ratio,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"

    local cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $queue -v $env_vars $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY-RUN] [$queue] $model | $condition | $distance | $domain | $mode | r=$ratio | s$seed"
        ((JOB_COUNT++))
        ((Q_COUNT[$queue]++))
    else
        local job_id
        if job_id=$(eval "$cmd" 2>&1); then
            echo "[OK] [$queue] $model | $condition | $distance | $domain | $mode | r=$ratio | s$seed → $job_id"
            echo "$queue:$model:$condition:$distance:$domain:$mode:$ratio:$seed:$job_id" >> "$LOG_FILE"
            ((JOB_COUNT++))
            ((Q_COUNT[$queue]++))
        else
            echo "[FAIL] [$queue] $model | $condition | $distance | $domain | $mode | r=$ratio | s$seed — $job_id"
            ((FAIL_COUNT++))
        fi
        sleep 0.1
    fi
}

# ============================================================
echo "============================================================"
echo "  Prior research experiment non-baseline resubmit (after KeyError fix)"
echo "  $(date)"
echo "============================================================"
echo "  Dry run : $DRY_RUN"
echo "  Models  : ${MODELS[*]}"
echo "  Conditions: ${CONDITIONS[*]}"
echo "  Queues  : ${QUEUES[*]}"
echo "  Expected: 432 jobs"
echo ""

{
    echo "# Prior research non-baseline launch: $(date)"
    echo "# Models: ${MODELS[*]}"
    echo "# Conditions: ${CONDITIONS[*]}"
    echo "# Queues: ${QUEUES[*]}"
    echo ""
} > "$LOG_FILE"

# Main loop
for model in "${MODELS[@]}"; do
    echo "--- $model ---"
    for condition in "${CONDITIONS[@]}"; do
        for distance in "${DISTANCES[@]}"; do
            for domain in "${DOMAINS[@]}"; do
                for mode in "${MODES[@]}"; do
                    for seed in "${SEEDS[@]}"; do
                        for ratio in "${RATIOS[@]}"; do
                            submit_job "$model" "$condition" "$distance" "$domain" "$mode" "$seed" "$ratio"
                        done
                    done
                done
            done
        done
    done
done

# Summary
{
    echo ""
    echo "# Completed: $(date)"
    echo "# Submitted: $JOB_COUNT"
    echo "# Failed: $FAIL_COUNT"
    for q in "${QUEUES[@]}"; do
        echo "# $q: ${Q_COUNT[$q]}"
    done
} >> "$LOG_FILE"

echo ""
echo "============================================================"
if $DRY_RUN; then
    echo "  [DRY-RUN] Would submit $JOB_COUNT jobs"
else
    echo "  Submitted: $JOB_COUNT jobs"
    echo "  Failed:    $FAIL_COUNT jobs"
fi
for q in "${QUEUES[@]}"; do
    echo "  $q: ${Q_COUNT[$q]}"
done
echo "  Log: $LOG_FILE"
echo "============================================================"

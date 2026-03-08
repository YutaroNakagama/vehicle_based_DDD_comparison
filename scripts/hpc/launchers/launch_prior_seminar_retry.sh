#!/bin/bash
# ============================================================
# Submit remaining 319 jobs to SEMINAR queue (no per-user limit)
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
FAILED_LIST="/tmp/failed_jobs.txt"

N_TRIALS=100
RANKING="knn"
QUEUE="SEMINAR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
LOG_FILE="$LOG_DIR/launch_prior_seminar_retry_${TIMESTAMP}.log"

JOB_COUNT=0
FAIL_COUNT=0

get_resources() {
    local model="$1"
    case "$model" in
        SvmW) echo "ncpus=4:mem=8gb 08:00:00" ;;
        SvmA) echo "ncpus=4:mem=8gb 10:00:00" ;;
        Lstm) echo "ncpus=4:mem=16gb 12:00:00" ;;
    esac
}

echo "============================================================"
echo "  SEMINAR Queue remaining submissions (319 jobs)"
echo "  $(date)"
echo "============================================================"

{
    echo "# SEMINAR retry: $(date)"
    echo ""
} > "$LOG_FILE"

while IFS=' ' read -r model condition distance domain mode ratio seed; do
    # Trim whitespace
    model=$(echo "$model" | xargs)
    condition=$(echo "$condition" | xargs)
    distance=$(echo "$distance" | xargs)
    domain=$(echo "$domain" | xargs)
    mode=$(echo "$mode" | xargs)
    ratio=$(echo "$ratio" | xargs)
    seed=$(echo "$seed" | xargs)

    res=$(get_resources "$model")
    ncpus_mem="${res% *}"
    walltime="${res#* }"

    m_short="${model:0:2}"
    c_short="${condition:0:2}"
    dist_short="${distance:0:1}"
    dom_short="${domain:0:1}"
    mode_short="${mode:0:1}"
    r_short=$(echo "$ratio" | tr -d '.')
    job_name="p${m_short}_${c_short}_${dist_short}${dom_short}_${mode_short}_r${r_short}_s${seed}"

    env_vars="MODEL=$model,CONDITION=$condition,MODE=$mode,DISTANCE=$distance,DOMAIN=$domain,SEED=$seed,RATIO=$ratio,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"

    cmd="qsub -N $job_name -l select=1:$ncpus_mem -l walltime=$walltime -q $QUEUE -v $env_vars $JOB_SCRIPT"

    if job_id=$(eval "$cmd" 2>&1); then
        echo "[OK] $model | $condition | $distance | $domain | $mode | r=$ratio | s$seed → $job_id"
        echo "$model:$condition:$distance:$domain:$mode:$ratio:$seed:$job_id" >> "$LOG_FILE"
        ((JOB_COUNT++))
    else
        echo "[FAIL] $model | $condition | $distance | $domain | $mode | r=$ratio | s$seed — $job_id"
        ((FAIL_COUNT++))
    fi
    sleep 0.05
done < "$FAILED_LIST"

{
    echo ""
    echo "# Completed: $(date)"
    echo "# Submitted: $JOB_COUNT"
    echo "# Failed: $FAIL_COUNT"
} >> "$LOG_FILE"

echo ""
echo "============================================================"
echo "  Submitted: $JOB_COUNT / 319"
echo "  Failed:    $FAIL_COUNT"
echo "  Log: $LOG_FILE"
echo "============================================================"

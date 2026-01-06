#!/bin/bash
# ============================================================
# Submit Remaining Domain Analysis Jobs
# ============================================================
# Run this script after current jobs complete to submit remaining jobs
# 
# Remaining: 48 jobs (smote dtw + target_only for seed 42, all for seed 123)
# ============================================================

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/domain_analysis/pbs_domain_comparison.sh"
TRIALS=100
RANKING="knn"
RUN_EVAL=true

submit() {
    local cond=$1 mode=$2 dist=$3 dom=$4 ratio=$5 seed=$6
    local cond_short
    case "$cond" in
        baseline) cond_short="bl" ;;
        smote) cond_short="sw" ;;
        *) cond_short="${cond:0:2}" ;;
    esac
    local dist_short="${dist:0:1}"
    local dom_short="${dom:0:2}"
    local mode_short="${mode:0:1}"
    local JOB_NAME="${cond_short}${dist_short}${dom_short}${mode_short}_s${seed}"
    
    result=$(qsub -N "$JOB_NAME" -l select=1:ncpus=4:mem=8gb -l walltime=12:00:00 -q SINGLE \
        -v CONDITION=$cond,MODE=$mode,DISTANCE=$dist,DOMAIN=$dom,RATIO=$ratio,SEED=$seed,N_TRIALS=$TRIALS,RANKING=$RANKING,RUN_EVAL=$RUN_EVAL \
        "$JOB_SCRIPT" 2>&1)
    
    if [[ "$result" == *"would exceed"* ]]; then
        echo "[LIMIT] $JOB_NAME - queue limit reached"
        return 1
    else
        echo "[OK] $JOB_NAME -> $result"
        sleep 0.1
        return 0
    fi
}

echo "============================================================"
echo "Submitting Remaining Domain Analysis Jobs"
echo "============================================================"
echo "Current queue status:"
qstat -u $USER 2>&1 | grep -E "Running|Queued" || echo "  Running: $(qstat -u $USER 2>&1 | grep ' R ' | wc -l), Queued: $(qstat -u $USER 2>&1 | grep ' Q ' | wc -l)"
echo ""

submitted=0

# smote source_only dtw (3 jobs) for seed 42
echo "--- smote source_only dtw seed=42 (3 jobs) ---"
for dom in in_domain mid_domain out_domain; do
    submit smote source_only dtw $dom 0.5 42 && ((submitted++)) || break
done

# smote target_only all (9 jobs) for seed 42
echo "--- smote target_only seed=42 (9 jobs) ---"
for dist in mmd wasserstein dtw; do
    for dom in in_domain mid_domain out_domain; do
        submit smote target_only $dist $dom 0.5 42 && ((submitted++)) || break 2
    done
done

# seed 123 - baseline (18 jobs)
echo "--- baseline seed=123 (18 jobs) ---"
for mode in source_only target_only; do
    for dist in mmd wasserstein dtw; do
        for dom in in_domain mid_domain out_domain; do
            submit baseline $mode $dist $dom 0 123 && ((submitted++)) || break 3
        done
    done
done

# seed 123 - smote (18 jobs)
echo "--- smote seed=123 (18 jobs) ---"
for mode in source_only target_only; do
    for dist in mmd wasserstein dtw; do
        for dom in in_domain mid_domain out_domain; do
            submit smote $mode $dist $dom 0.5 123 && ((submitted++)) || break 3
        done
    done
done

echo ""
echo "============================================================"
echo "Submitted: $submitted jobs"
echo "============================================================"

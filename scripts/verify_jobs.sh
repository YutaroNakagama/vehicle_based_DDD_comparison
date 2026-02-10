#!/bin/bash
# Verify completed Exp3 jobs: exit codes, error logs, expected outputs
set -euo pipefail

BASE="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
LOGDIR="$BASE/scripts/hpc/logs/train"

echo "=============================================="
echo " Exp3 Job Verification Report"
echo " $(date)"
echo "=============================================="

check_model() {
    local model="$1"
    local model_dir="$BASE/models/$model"
    local eval_dir="$BASE/results/outputs/evaluation/$model"
    local train_dir="$BASE/results/outputs/training/$model"
    
    echo ""
    echo "====== $model ======"
    
    # Get unique job IDs (exclude .spcc-adm1 duplicates and latest_job.txt)
    local job_ids=$(ls "$model_dir" 2>/dev/null | grep -v spcc | grep -v latest | sort -n)
    local total=$(echo "$job_ids" | wc -l)
    echo "Total unique model dirs: $total"
    
    local ok=0
    local missing_log=0
    local error_jobs=""
    local missing_model_files=""
    local missing_eval=""
    local walltime_killed=""
    
    for jid in $job_ids; do
        local logf="$LOGDIR/${jid}.spcc-adm1.OU"
        local mdir="$model_dir/$jid"
        
        # 1. Check log exists
        if [ ! -f "$logf" ]; then
            missing_log=$((missing_log+1))
            error_jobs="$error_jobs MISSING_LOG:$jid"
            continue
        fi
        
        # 2. Check for walltime kill (PBS writes this)
        if grep -qi "walltime" "$logf" 2>/dev/null | grep -qi "exceeded\|killed" 2>/dev/null; then
            walltime_killed="$walltime_killed $jid"
        fi
        
        # 3. Check for Python Traceback
        if grep -q "Traceback (most recent call last)" "$logf" 2>/dev/null; then
            err_msg=$(grep -A1 "Traceback" "$logf" | tail -1)
            error_jobs="$error_jobs TRACEBACK:$jid($err_msg)"
            continue
        fi
        
        # 4. Check for fatal errors (excluding TF warnings)
        if grep -qP '(?<!Deprecation)Error:' "$logf" 2>/dev/null; then
            real=$(grep -P '(?<!Deprecation)Error:' "$logf" | grep -v "OMP:" | grep -v "WARNING" | grep -v "DeprecationWarning" | grep -v "FutureWarning" | grep -v "tensorflow" | grep -v "np.float" | head -1)
            if [ -n "$real" ]; then
                error_jobs="$error_jobs ERROR:$jid($real)"
                continue
            fi
        fi
        
        # 5. Check model directory has expected files
        if [ "$model" = "Lstm" ]; then
            # Lstm should have .keras model + metadata
            if ! ls "$mdir"/*.keras >/dev/null 2>&1 && ! ls "$mdir"/*.h5 >/dev/null 2>&1; then
                missing_model_files="$missing_model_files $jid(no_model_file)"
            fi
        else
            # SvmA/SvmW should have .pkl or .joblib 
            if ! ls "$mdir"/*.pkl >/dev/null 2>&1 && ! ls "$mdir"/*.joblib >/dev/null 2>&1; then
                missing_model_files="$missing_model_files $jid(no_model_file)"
            fi
        fi
        
        ok=$((ok+1))
    done
    
    # Count eval outputs
    local eval_count=$(ls "$eval_dir" 2>/dev/null | grep -v latest | wc -l)
    local train_count=$(ls "$train_dir" 2>/dev/null | grep -v latest | wc -l)
    
    echo "Logs OK (no errors): $ok / $total"
    echo "Missing logs: $missing_log"
    echo "Eval outputs: $eval_count"
    echo "Training outputs: $train_count"
    
    if [ -n "$error_jobs" ]; then
        echo "*** ERROR JOBS: $error_jobs"
    fi
    if [ -n "$missing_model_files" ]; then
        echo "*** MISSING MODEL FILES: $missing_model_files"
    fi
    if [ -n "$walltime_killed" ]; then
        echo "*** WALLTIME KILLED: $walltime_killed"
    fi
    
    # Check eval completeness for split2 jobs
    # Pooled jobs (first 2) don't have auto-eval
    echo ""
    echo "  -- Models without eval (excl pooled) --"
    comm -23 <(echo "$job_ids" | sort) <(ls "$eval_dir" 2>/dev/null | grep -v latest | sort) | while read jid; do
        # Check if this is a pooled job
        if grep -q "pooled\|mode=pooled" "$LOGDIR/${jid}.spcc-adm1.OU" 2>/dev/null; then
            echo "  $jid (pooled - eval not auto-run, expected)"
        else
            echo "  $jid (MISSING EVAL - unexpected)"
        fi
    done
}

# Check Lstm
check_model "Lstm"

# Check SvmA (only completed jobs)
echo ""
echo "====== SvmA (checking only completed) ======"
svma_model_dir="$BASE/models/SvmA"
svma_job_ids=$(ls "$svma_model_dir" 2>/dev/null | grep -v spcc | grep -v latest | sort -n)
svma_total=$(echo "$svma_job_ids" | wc -l)
echo "Total unique SvmA model dirs: $svma_total"

# Separate completed vs in-queue
running_jids=$(qstat -u s2240011 2>/dev/null | grep s2240011 | awk '{print $1}' | sed 's/.spcc-adm1//')
echo "Currently in queue: $(echo "$running_jids" | wc -l)"

svma_ok=0
svma_err=""
svma_no_eval=""
svma_no_model=""
for jid in $svma_job_ids; do
    # Skip if still running
    if echo "$running_jids" | grep -q "^${jid}$"; then
        continue
    fi
    
    logf="$LOGDIR/${jid}.spcc-adm1.OU"
    mdir="$svma_model_dir/$jid"
    
    if [ ! -f "$logf" ]; then
        svma_err="$svma_err MISSING_LOG:$jid"
        continue
    fi
    
    if grep -q "Traceback (most recent call last)" "$logf" 2>/dev/null; then
        err_line=$(grep -A2 "Traceback" "$logf" | tail -1)
        svma_err="$svma_err TRACEBACK:$jid"
        continue
    fi
    
    # Check model files
    if ! ls "$mdir"/*.pkl >/dev/null 2>&1 && ! ls "$mdir"/*.joblib >/dev/null 2>&1; then
        svma_no_model="$svma_no_model $jid"
    fi
    
    # Check eval
    if [ ! -d "$BASE/results/outputs/evaluation/SvmA/$jid" ]; then
        # Check if pooled
        if grep -q "mode=pooled\|pooled" "$logf" 2>/dev/null; then
            svma_no_eval="$svma_no_eval $jid(pooled)"
        else
            svma_no_eval="$svma_no_eval $jid"
        fi
    fi
    
    svma_ok=$((svma_ok+1))
done
echo "Completed & OK: $svma_ok"
if [ -n "$svma_err" ]; then echo "*** ERRORS: $svma_err"; fi
if [ -n "$svma_no_model" ]; then echo "*** MISSING MODEL FILES: $svma_no_model"; fi
if [ -n "$svma_no_eval" ]; then echo "*** MISSING EVAL: $svma_no_eval"; fi

# Check SvmW
check_model "SvmW"

echo ""
echo "=============================================="
echo " Verification Complete"
echo "=============================================="

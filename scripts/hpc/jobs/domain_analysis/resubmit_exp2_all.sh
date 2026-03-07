#!/bin/bash
# ============================================================
# Experiment 2: Full resubmission script
# 70/15/15 split ratio + source_only test partition fix
# ============================================================
# Total: 8 conditions × 3 modes × 3 distances × 2 domains × 2 seeds = 288 jobs
# RF: baseline, smote_plain, smote, undersample (7 configs × 3 × 3 × 2 × 2 = 252)
# BalancedRF: balanced_rf (1 config × 3 × 3 × 2 × 2 = 36)
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

LOG_DIR="scripts/hpc/logs/domain"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUBMIT_LOG="${LOG_DIR}/exp2_rerun_${TIMESTAMP}.log"
SUBMITTED_FILE="${LOG_DIR}/exp2_rerun_submitted_${TIMESTAMP}.txt"

PBS_SCRIPT="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"

# Resource settings
RF_WALLTIME="10:00:00"
RF_MEM="8gb"
RF_NCPUS=4

BRF_WALLTIME="24:00:00"
BRF_MEM="8gb"
BRF_NCPUS=4

# Queue selection: DEFAULT for short jobs
QUEUE="DEFAULT"

# Experiment parameters
CONDITIONS="baseline smote_plain smote undersample balanced_rf"
MODES="source_only target_only mixed"
DISTANCES="mmd dtw wasserstein"
DOMAINS="in_domain out_domain"
SEEDS="42 123"

# Ratios for conditions that need them
RATIOS_SMOTE_PLAIN="0.1 0.5"
RATIOS_SMOTE="0.1 0.5"
RATIOS_UNDERSAMPLE="0.1 0.5"

total=0
submitted=0
failed=0

echo "============================================================" | tee "$SUBMIT_LOG"
echo "Experiment 2 Full Resubmission" | tee -a "$SUBMIT_LOG"
echo "Date: $(date)" | tee -a "$SUBMIT_LOG"
echo "Commit: $(git rev-parse --short HEAD)" | tee -a "$SUBMIT_LOG"
echo "Changes: 70/15/15 split + source_only test partition fix" | tee -a "$SUBMIT_LOG"
echo "============================================================" | tee -a "$SUBMIT_LOG"

submit_job() {
    local cond="$1"
    local mode="$2"
    local dist="$3"
    local dom="$4"
    local seed="$5"
    local ratio="${6:-}"
    
    # Select model and resources
    local model walltime mem
    if [[ "$cond" == "balanced_rf" ]]; then
        model="BalancedRF"
        walltime="$BRF_WALLTIME"
        mem="$BRF_MEM"
    else
        model="RF"
        walltime="$RF_WALLTIME"
        mem="$RF_MEM"
    fi
    
    # Build job name (max ~15 chars for PBS display)
    local ratio_short=""
    if [[ -n "$ratio" ]]; then
        ratio_short=$(echo "$ratio" | sed 's/0\./r/')
    fi
    
    # Abbreviations for job name
    local cond_abbr
    case "$cond" in
        baseline)     cond_abbr="bs" ;;
        smote_plain)  cond_abbr="sp" ;;
        smote)        cond_abbr="sm" ;;
        undersample)  cond_abbr="us" ;;
        balanced_rf)  cond_abbr="bf" ;;
    esac
    
    local mode_abbr
    case "$mode" in
        source_only) mode_abbr="so" ;;
        target_only) mode_abbr="to" ;;
        mixed)       mode_abbr="mx" ;;
    esac
    
    local dist_abbr
    case "$dist" in
        mmd)          dist_abbr="mmd" ;;
        dtw)          dist_abbr="dtw" ;;
        wasserstein)  dist_abbr="was" ;;
    esac
    
    local dom_abbr
    case "$dom" in
        in_domain)   dom_abbr="in" ;;
        out_domain)  dom_abbr="ou" ;;
    esac
    
    local job_name="${cond_abbr}_${dist_abbr}_${dom_abbr}_${mode_abbr}_s${seed}"
    if [[ -n "$ratio_short" ]]; then
        job_name="${cond_abbr}${ratio_short}_${dist_abbr}_${dom_abbr}_${mode_abbr}_s${seed}"
    fi
    
    # Environment variables for PBS script
    local env_vars="CONDITION=${cond},MODE=${mode},DISTANCE=${dist},DOMAIN=${dom},SEED=${seed}"
    if [[ -n "$ratio" ]]; then
        env_vars="${env_vars},RATIO=${ratio}"
    fi
    
    total=$((total + 1))
    
    # Submit
    local result
    result=$(qsub \
        -N "$job_name" \
        -q "$QUEUE" \
        -l select=1:ncpus=${RF_NCPUS}:mem=${mem} \
        -l walltime=${walltime} \
        -v "$env_vars" \
        "$PBS_SCRIPT" 2>&1) || true
    
    if [[ "$result" == *".spcc-adm1"* ]]; then
        submitted=$((submitted + 1))
        local jid=$(echo "$result" | grep -oP '\d+\.spcc' | grep -oP '\d+')
        echo "$jid $job_name $cond $mode $dist $dom $seed $ratio" >> "$SUBMITTED_FILE"
        echo "  ✅ [$submitted] $job_name → $jid" | tee -a "$SUBMIT_LOG"
    else
        failed=$((failed + 1))
        echo "  ❌ $job_name: $result" | tee -a "$SUBMIT_LOG"
    fi
}

# ── Submit all combinations ──
for cond in $CONDITIONS; do
    echo "" | tee -a "$SUBMIT_LOG"
    echo "── Condition: $cond ──" | tee -a "$SUBMIT_LOG"
    
    case "$cond" in
        baseline|balanced_rf)
            # No ratio needed
            for mode in $MODES; do
                for dist in $DISTANCES; do
                    for dom in $DOMAINS; do
                        for seed in $SEEDS; do
                            submit_job "$cond" "$mode" "$dist" "$dom" "$seed"
                        done
                    done
                done
            done
            ;;
        smote_plain)
            for ratio in $RATIOS_SMOTE_PLAIN; do
                for mode in $MODES; do
                    for dist in $DISTANCES; do
                        for dom in $DOMAINS; do
                            for seed in $SEEDS; do
                                submit_job "$cond" "$mode" "$dist" "$dom" "$seed" "$ratio"
                            done
                        done
                    done
                done
            done
            ;;
        smote)
            for ratio in $RATIOS_SMOTE; do
                for mode in $MODES; do
                    for dist in $DISTANCES; do
                        for dom in $DOMAINS; do
                            for seed in $SEEDS; do
                                submit_job "$cond" "$mode" "$dist" "$dom" "$seed" "$ratio"
                            done
                        done
                    done
                done
            done
            ;;
        undersample)
            for ratio in $RATIOS_UNDERSAMPLE; do
                for mode in $MODES; do
                    for dist in $DISTANCES; do
                        for dom in $DOMAINS; do
                            for seed in $SEEDS; do
                                submit_job "$cond" "$mode" "$dist" "$dom" "$seed" "$ratio"
                            done
                        done
                    done
                done
            done
            ;;
    esac
done

echo "" | tee -a "$SUBMIT_LOG"
echo "============================================================" | tee -a "$SUBMIT_LOG"
echo "SUMMARY" | tee -a "$SUBMIT_LOG"
echo "  Total:     $total" | tee -a "$SUBMIT_LOG"
echo "  Submitted: $submitted" | tee -a "$SUBMIT_LOG"
echo "  Failed:    $failed" | tee -a "$SUBMIT_LOG"
echo "  Log:       $SUBMIT_LOG" | tee -a "$SUBMIT_LOG"
echo "  Tracking:  $SUBMITTED_FILE" | tee -a "$SUBMIT_LOG"
echo "============================================================" | tee -a "$SUBMIT_LOG"

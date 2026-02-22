#!/bin/bash
# ============================================================
# Submit missing SvmW domain_train jobs
# 35 configs: 21 imbalv3 + 14 smote_plain
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
cd "$PROJECT_ROOT"

WALLTIME="12:00:00"
RES="ncpus=8:mem=16gb"
N_TRIALS=100
RANKING="knn"

CPU_QUEUE_IDX=0
get_cpu_queue() {
    local queues=("DEFAULT" "SINGLE" "SMALL" "DEFAULT" "SINGLE" "LONG")
    local q="${queues[$((CPU_QUEUE_IDX % ${#queues[@]}))]}"
    ((CPU_QUEUE_IDX++))
    echo "$q"
}

submit_count=0

submit_job() {
    local CONDITION="$1"
    local DISTANCE="$2"
    local DOMAIN="$3"
    local RATIO="$4"
    local SEED="$5"
    
    local QUEUE
    QUEUE=$(get_cpu_queue)
    
    # Build job name: Sw_{cond:0:2}_{dist:0:1}{dom:0:1}_dt_r{ratio}_s{seed}
    local COND_SHORT="${CONDITION:0:2}"
    local JOB_NAME="Sw_${COND_SHORT}_${DISTANCE:0:1}${DOMAIN:0:1}_dt_r${RATIO}_s${SEED}"
    
    echo "[SUBMIT] $JOB_NAME → $QUEUE (MODEL=SvmW, COND=$CONDITION, DIST=$DISTANCE, DOM=$DOMAIN, R=$RATIO, S=$SEED)"
    
    qsub \
        -N "$JOB_NAME" \
        -l "select=1:${RES}" \
        -l "walltime=${WALLTIME}" \
        -q "$QUEUE" \
        -v "MODEL=SvmW,CONDITION=${CONDITION},DISTANCE=${DISTANCE},DOMAIN=${DOMAIN},RATIO=${RATIO},SEED=${SEED},N_TRIALS=${N_TRIALS},RANKING=${RANKING},RUN_EVAL=true" \
        "$JOB_SCRIPT"
    
    ((submit_count++))
    sleep 1
}

echo "=== Submitting missing SvmW domain_train jobs ==="
echo ""

# imbalv3 missing configs (CONDITION=smote → generates imbalv3 tag)
# dtw in_domain: all 4 (r0.1 s42, r0.1 s123, r0.5 s42, r0.5 s123)
submit_job "smote" "dtw" "in_domain" "0.1" "42"
submit_job "smote" "dtw" "in_domain" "0.1" "123"
submit_job "smote" "dtw" "in_domain" "0.5" "42"
submit_job "smote" "dtw" "in_domain" "0.5" "123"

# dtw out_domain: all 4
submit_job "smote" "dtw" "out_domain" "0.1" "42"
submit_job "smote" "dtw" "out_domain" "0.1" "123"
submit_job "smote" "dtw" "out_domain" "0.5" "42"
submit_job "smote" "dtw" "out_domain" "0.5" "123"

# mmd in_domain: r0.5 only
submit_job "smote" "mmd" "in_domain" "0.5" "42"
submit_job "smote" "mmd" "in_domain" "0.5" "123"

# mmd out_domain: r0.1 s123, r0.5 both
submit_job "smote" "mmd" "out_domain" "0.1" "123"
submit_job "smote" "mmd" "out_domain" "0.5" "42"
submit_job "smote" "mmd" "out_domain" "0.5" "123"

# wasserstein in_domain: all 4
submit_job "smote" "wasserstein" "in_domain" "0.1" "42"
submit_job "smote" "wasserstein" "in_domain" "0.1" "123"
submit_job "smote" "wasserstein" "in_domain" "0.5" "42"
submit_job "smote" "wasserstein" "in_domain" "0.5" "123"

# wasserstein out_domain: all 4
submit_job "smote" "wasserstein" "out_domain" "0.1" "42"
submit_job "smote" "wasserstein" "out_domain" "0.1" "123"
submit_job "smote" "wasserstein" "out_domain" "0.5" "42"
submit_job "smote" "wasserstein" "out_domain" "0.5" "123"

# smote_plain missing configs (CONDITION=smote_plain)
# dtw in_domain: r0.1 s123, r0.5 both
submit_job "smote_plain" "dtw" "in_domain" "0.1" "123"
submit_job "smote_plain" "dtw" "in_domain" "0.5" "42"
submit_job "smote_plain" "dtw" "in_domain" "0.5" "123"

# dtw out_domain: r0.5 both
submit_job "smote_plain" "dtw" "out_domain" "0.5" "42"
submit_job "smote_plain" "dtw" "out_domain" "0.5" "123"

# mmd in_domain: r0.5 both
submit_job "smote_plain" "mmd" "in_domain" "0.5" "42"
submit_job "smote_plain" "mmd" "in_domain" "0.5" "123"

# mmd out_domain: r0.5 both
submit_job "smote_plain" "mmd" "out_domain" "0.5" "42"
submit_job "smote_plain" "mmd" "out_domain" "0.5" "123"

# wasserstein in_domain: r0.1 s123, r0.5 both
submit_job "smote_plain" "wasserstein" "in_domain" "0.1" "123"
submit_job "smote_plain" "wasserstein" "in_domain" "0.5" "42"
submit_job "smote_plain" "wasserstein" "in_domain" "0.5" "123"

# wasserstein out_domain: r0.5 both
submit_job "smote_plain" "wasserstein" "out_domain" "0.5" "42"
submit_job "smote_plain" "wasserstein" "out_domain" "0.5" "123"

echo ""
echo "=== Submitted $submit_count SvmW domain_train jobs ==="

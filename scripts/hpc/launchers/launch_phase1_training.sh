#!/bin/bash
# ============================================================
# Phase 1 Training Launcher — baseline + balanced_rf target_only
# ============================================================
# These 24 configs (baseline×12 + balanced_rf×12) were NEVER TRAINED
# for exp2/split2. This launcher submits training+eval jobs.
#
# The training script (pbs_domain_comparison_split2.sh) has Bug #4 fix
# (--jobid $JOBID_CLEAN), so eval will also work correctly.
#
# Usage:
#   bash scripts/hpc/launchers/launch_phase1_training.sh          # dry-run
#   bash scripts/hpc/launchers/launch_phase1_training.sh --submit # submit
# ============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

PBS_SCRIPT="scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh"
SUBMIT="${1:-}"
TMPDIR_SAFE="${HOME}/tmp"
mkdir -p "$TMPDIR_SAFE"

# Queue configuration (training → heavier resources)
QUEUE="SEMINAR"
NCPUS=4
MEM="8gb"
WALLTIME="04:00:00"

CONDITIONS=("baseline" "balanced_rf")
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("in_domain" "out_domain")
SEEDS=("42" "123")
MODE="target_only"

count_submit=0
count_error=0

echo "============================================================"
echo "  Exp2 Phase 1: Missing Training (baseline + balanced_rf)"
echo "  $(date)"
echo "============================================================"
if [[ "$SUBMIT" != "--submit" ]]; then
    echo "  MODE: DRY-RUN (add --submit to actually submit)"
else
    echo "  MODE: SUBMIT"
fi
echo "  PBS script: $PBS_SCRIPT"
echo "  Queue: $QUEUE  Resources: ncpus=$NCPUS mem=$MEM walltime=$WALLTIME"
echo "  Configs: ${#CONDITIONS[@]} conds × ${#DISTANCES[@]} dists × ${#DOMAINS[@]} doms × ${#SEEDS[@]} seeds = $(( ${#CONDITIONS[@]} * ${#DISTANCES[@]} * ${#DOMAINS[@]} * ${#SEEDS[@]} ))"
echo "============================================================"
echo ""

for COND in "${CONDITIONS[@]}"; do
    for DIST in "${DISTANCES[@]}"; do
        for DOM in "${DOMAINS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                # Short job name
                C="${COND:0:2}"
                D="${DIST:0:2}"
                DM="${DOM:0:1}"
                JOB_NAME="ft_${C}_${D}${DM}_s${SEED}"

                VARS="CONDITION=${COND},MODE=${MODE},DISTANCE=${DIST},DOMAIN=${DOM},SEED=${SEED},RUN_EVAL=true"

                if [[ "$SUBMIT" == "--submit" ]]; then
                    RESULT=$(TMPDIR="$TMPDIR_SAFE" qsub \
                        -N "$JOB_NAME" \
                        -l select=1:ncpus=${NCPUS}:mem=${MEM} \
                        -l walltime=${WALLTIME} \
                        -q ${QUEUE} \
                        -v "$VARS" \
                        "$PBS_SCRIPT" 2>&1)

                    if [[ $? -eq 0 ]]; then
                        echo "[SUBMIT] ${JOB_NAME}  ${MODE}/${COND}/${DIST}/${DOM}/s${SEED}  → ${RESULT}"
                        count_submit=$((count_submit + 1))
                    else
                        echo "[ERROR]  ${JOB_NAME} — qsub failed: ${RESULT}"
                        count_error=$((count_error + 1))
                    fi
                else
                    echo "[DRY-RUN] ${JOB_NAME}  ${MODE}/${COND}/${DIST}/${DOM}/s${SEED}"
                    count_submit=$((count_submit + 1))
                fi
            done
        done
    done
done

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Submitted/Would submit: $count_submit"
echo "  Errors:                 $count_error"
echo "============================================================"

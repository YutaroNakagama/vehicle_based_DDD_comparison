#!/bin/bash
# ============================================================
# Phase 1 Re-eval Launcher — smote/smote_plain/undersample target_only
# ============================================================
# These 36 configs have trained models BUT their Phase 1 re-eval
# failed due to Bug #4 (eval loaded wrong model via latest_job.txt).
#
# This launcher submits eval-only jobs with the correct TRAIN_JOBID.
# Uses the same approach as launch_reeval_phase23.sh.
#
# Pre-requisite:
#   ~/tmp/exp2_phase1_jobid_map.tsv must exist (built by build_phase1_jobid_map.py)
#
# Usage:
#   bash scripts/hpc/launchers/launch_phase1_reeval.sh          # dry-run
#   bash scripts/hpc/launchers/launch_phase1_reeval.sh --submit # submit
# ============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

PBS_SCRIPT="scripts/hpc/jobs/domain_analysis/pbs_reeval_split2_v2.sh"
JOBID_MAP="${HOME}/tmp/exp2_phase1_jobid_map.tsv"
SUBMIT="${1:-}"
TMPDIR_SAFE="${HOME}/tmp"
mkdir -p "$TMPDIR_SAFE"

# Queue configuration (eval-only → lightweight)
QUEUE="SEMINAR"
NCPUS=2
MEM="4gb"
WALLTIME="01:00:00"

if [[ ! -f "$JOBID_MAP" ]]; then
    echo "[ERROR] JOBID map not found: $JOBID_MAP"
    echo "  Run: cd $PROJECT_ROOT && python3 ~/tmp/build_phase1_jobid_map.py"
    exit 1
fi

count_submit=0
count_error=0

ENTRIES=$(( $(wc -l < "$JOBID_MAP") - 1 ))

echo "============================================================"
echo "  Exp2 Phase 1 Re-evaluation (smote variants)"
echo "  $(date)"
echo "============================================================"
if [[ "$SUBMIT" != "--submit" ]]; then
    echo "  MODE: DRY-RUN (add --submit to actually submit)"
else
    echo "  MODE: SUBMIT"
fi
echo "  JOBID map: $JOBID_MAP ($ENTRIES entries)"
echo "  Queue: $QUEUE  Resources: ncpus=$NCPUS mem=$MEM walltime=$WALLTIME"
echo "============================================================"
echo ""

while IFS=$'\t' read -r MODE COND DIST DOM SEED JOBID; do
    # Skip header
    [[ "$MODE" == "mode" ]] && continue

    # Build short job name
    C="${COND:0:2}"
    D="${DIST:0:2}"
    DM="${DOM:0:1}"
    JOB_NAME="re1_${C}_${D}${DM}_s${SEED}"

    VARS="CONDITION=${COND},MODE=${MODE},DISTANCE=${DIST},DOMAIN=${DOM},SEED=${SEED},TRAIN_JOBID=${JOBID}"

    if [[ "$SUBMIT" == "--submit" ]]; then
        RESULT=$(TMPDIR="$TMPDIR_SAFE" qsub \
            -N "$JOB_NAME" \
            -l select=1:ncpus=${NCPUS}:mem=${MEM} \
            -l walltime=${WALLTIME} \
            -q ${QUEUE} \
            -v "$VARS" \
            "$PBS_SCRIPT" 2>&1)

        if [[ $? -eq 0 ]]; then
            echo "[SUBMIT] ${JOB_NAME}  train_jobid=${JOBID}  ${MODE}/${COND}/${DIST}/${DOM}/s${SEED}  → ${RESULT}"
            count_submit=$((count_submit + 1))
        else
            echo "[ERROR]  ${JOB_NAME} — qsub failed: ${RESULT}"
            count_error=$((count_error + 1))
        fi
    else
        echo "[DRY-RUN] ${JOB_NAME}  train_jobid=${JOBID}  ${MODE}/${COND}/${DIST}/${DOM}/s${SEED}"
        count_submit=$((count_submit + 1))
    fi

done < "$JOBID_MAP"

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Submitted/Would submit: $count_submit"
echo "  Errors:                 $count_error"
echo "============================================================"

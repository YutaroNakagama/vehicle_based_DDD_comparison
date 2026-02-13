#!/bin/bash
# ============================================================
# Re-evaluation Launcher for Exp2 Phase 2+3
# ============================================================
# Reads JOBID mapping from ~/tmp/exp2_jobid_map.tsv (built by
# build_jobid_map.py) and submits eval-only jobs.
#
# Fixes Bug #4: running jobs used OLD PBS script without --jobid,
# so eval loaded wrong model. This re-eval uses the correct JOBID.
#
# Pre-requisite:
#   cd $PROJECT_ROOT && python3 ~/tmp/build_jobid_map.py
#
# Usage:
#   bash scripts/hpc/launchers/launch_reeval_phase23.sh          # dry-run
#   bash scripts/hpc/launchers/launch_reeval_phase23.sh --submit # submit
# ============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

PBS_SCRIPT="scripts/hpc/jobs/domain_analysis/pbs_reeval_split2_v2.sh"
JOBID_MAP="${HOME}/tmp/exp2_jobid_map.tsv"
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
    echo "  Run: cd $PROJECT_ROOT && python3 ~/tmp/build_jobid_map.py"
    exit 1
fi

count_submit=0
count_error=0

echo "============================================================"
echo "  Exp2 Phase 2+3 Re-evaluation Launcher"
echo "  $(date)"
echo "============================================================"
if [[ "$SUBMIT" != "--submit" ]]; then
    echo "  MODE: DRY-RUN (add --submit to actually submit)"
else
    echo "  MODE: SUBMIT"
fi
echo "  JOBID map: $JOBID_MAP ($(( $(wc -l < "$JOBID_MAP") - 1 )) entries)"
echo "  Queue: $QUEUE  Resources: ncpus=$NCPUS mem=$MEM walltime=$WALLTIME"
echo "============================================================"
echo ""

# Read TSV and submit (skip header)
while IFS=$'\t' read -r MODE COND DIST DOM SEED JOBID; do
    # Skip header
    [[ "$MODE" == "mode" ]] && continue
    
    # Build short job name
    C="${COND:0:2}"
    D="${DIST:0:2}"
    DM="${DOM:0:1}"  # i or o
    M="${MODE:0:1}"  # s or m
    JOB_NAME="re_${M}${C}_${D}${DM}_s${SEED}"
    
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

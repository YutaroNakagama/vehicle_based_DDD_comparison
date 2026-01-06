#!/bin/bash
# =============================================================================
# Launcher: Prior Research Replication Experiments (SvmA, SvmW, Lstm)
# =============================================================================
# Submits training jobs for all three prior research models with multiple seeds.
#
# Usage:
#   ./scripts/hpc/launchers/launch_prior_research.sh
#   ./scripts/hpc/launchers/launch_prior_research.sh --dry-run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PBS_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research.sh"

# ===== Configuration =====
MODELS=("SvmA" "SvmW" "Lstm")
SEEDS=(42 123)

# Walltime settings per model
declare -A WALLTIMES
WALLTIMES["SvmA"]="24:00:00"  # PSO optimization takes time
WALLTIMES["SvmW"]="24:00:00"  # Same as SvmA
WALLTIMES["Lstm"]="12:00:00"  # Deep learning, relatively faster

# Queue selection
QUEUE="SINGLE"

# ===== Parse arguments =====
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[DRY-RUN] No jobs will be submitted."
fi

# ===== Ensure log directory exists =====
LOG_DIR="$PROJECT_ROOT/scripts/hpc/logs/train"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "  Prior Research Replication Launcher"
echo "=============================================="
echo "  Models:  ${MODELS[*]}"
echo "  Seeds:   ${SEEDS[*]}"
echo "  Queue:   $QUEUE"
echo "  PBS:     $PBS_SCRIPT"
echo "=============================================="
echo ""

JOB_COUNT=0
SUBMITTED_JOBS=()

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        WALLTIME="${WALLTIMES[$MODEL]}"
        JOB_NAME="${MODEL}_s${SEED}"
        
        echo -n "[$((JOB_COUNT+1))] $JOB_NAME (walltime=$WALLTIME) ... "
        
        if $DRY_RUN; then
            echo "[DRY-RUN] qsub -N $JOB_NAME -v MODEL=$MODEL,SEED=$SEED -l walltime=$WALLTIME -q $QUEUE $PBS_SCRIPT"
        else
            JOB_ID=$(qsub \
                -N "$JOB_NAME" \
                -v "MODEL=$MODEL,SEED=$SEED" \
                -l "walltime=$WALLTIME" \
                -q "$QUEUE" \
                "$PBS_SCRIPT")
            
            echo "Submitted: $JOB_ID"
            SUBMITTED_JOBS+=("$JOB_NAME:$JOB_ID")
        fi
        
        JOB_COUNT=$((JOB_COUNT + 1))
    done
done

echo ""
echo "=============================================="
echo "  Submission Complete"
echo "=============================================="
echo "  Total Jobs: $JOB_COUNT"
if ! $DRY_RUN; then
    echo ""
    echo "  Submitted:"
    for job in "${SUBMITTED_JOBS[@]}"; do
        echo "    - $job"
    done
fi
echo ""
echo "  Monitor: qstat -u \$USER"
echo "  Logs:    $LOG_DIR/"
echo "=============================================="

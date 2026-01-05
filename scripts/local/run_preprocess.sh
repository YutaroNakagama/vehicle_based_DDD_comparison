#!/bin/bash
# =============================================================================
# Preprocessing Pipeline - Parallel Execution
# =============================================================================
# Usage: ./scripts/local/run_preprocess.sh [OPTIONS]
#
# Options:
#   --model <name>    Model to preprocess (common, SvmA, SvmW, Lstm)
#                     Default: common (generates features for all models)
#   --jobs N          Number of parallel jobs (default: auto-detect)
#   --jittering       Enable jittering augmentation
#   --fg              Run in foreground (default: background)
#
# Examples:
#   ./scripts/local/run_preprocess.sh                    # common model, background
#   ./scripts/local/run_preprocess.sh --model SvmA       # SvmA model only
#   ./scripts/local/run_preprocess.sh --jobs 10          # Limit to 10 parallel jobs
#   ./scripts/local/run_preprocess.sh --jittering        # With jittering augmentation
# =============================================================================

# Check for foreground mode - default is background
if [[ "$*" != *"--fg"* ]] && [[ -z "$_PREPROCESS_RUNNING" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    LOG_DIR="$SCRIPT_DIR/logs/preprocess"
    mkdir -p "$LOG_DIR"
    
    # Timestamp for this run
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/preprocess_${TIMESTAMP}.log"
    
    export _PREPROCESS_RUNNING=1
    nohup "$0" "$@" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "✅ Started preprocessing in background (PID: $PID)"
    echo "   Log: $LOG_FILE"
    echo ""
    echo "Monitor:"
    echo "   tail -f $LOG_FILE"
    echo "   watch \"ps aux | grep preprocess.py | grep -v grep\""
    echo ""
    echo "Stop:"
    echo "   kill $PID"
    exit 0
fi

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Parse arguments
MODEL="common"
N_JOBS=""
USE_JITTERING=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --jobs) N_JOBS="$2"; shift 2 ;;
        --jittering) USE_JITTERING=true; shift ;;
        --fg) shift ;;
        *) shift ;;
    esac
done

# Validate model choice
VALID_MODELS="common SvmA SvmW Lstm"
if [[ ! " $VALID_MODELS " =~ " $MODEL " ]]; then
    echo "ERROR: Invalid model '$MODEL'. Valid choices: $VALID_MODELS"
    exit 1
fi

# Activate virtual environment
if [[ -d ".venv-linux" ]]; then
    source .venv-linux/bin/activate
    echo "[OK] Activated .venv-linux environment"
elif [[ -d ".venv" ]]; then
    source .venv/bin/activate
    echo "[OK] Activated .venv environment"
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Set dataset path (override default in config.py)
export DDD_DATASET_PATH="../vehicle_ddd_eval/dataset/mdapbe/physio"

# Set number of processes
if [[ -z "$N_JOBS" ]]; then
    # Auto-detect: use all available CPUs for maximum parallelization
    N_JOBS=$(nproc)
fi
export N_PROC="$N_JOBS"

# Optimize for parallel processing - prevent thread oversubscription
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Count subjects
SUBJECT_COUNT=$(ls data/interim/eeg/common/*.csv 2>/dev/null | wc -l)

echo "========================================"
echo "PREPROCESSING PIPELINE"
echo "========================================"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Model: ${MODEL}"
echo "Subjects: ${SUBJECT_COUNT}"
echo "Parallel jobs: ${N_JOBS}"
echo "Jittering: ${USE_JITTERING}"
echo "CPU cores available: $(nproc)"
echo ""

# Build command
CMD="python scripts/python/preprocess/preprocess.py --model ${MODEL} --multi_process"
if [[ "$USE_JITTERING" == true ]]; then
    CMD="$CMD --jittering"
fi

echo "Running: $CMD"
echo ""

# Run preprocessing
START_TIME=$(date +%s)

eval "$CMD"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================"
echo "PREPROCESSING COMPLETED!"
echo "========================================"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Duration: ${MINUTES}m ${SECONDS}s"
echo ""

# Show output files
echo "=== Output Files ==="
if [[ "$MODEL" == "common" ]]; then
    echo "Features generated for all models (SvmA, SvmW, Lstm, RF)"
    ls -lh data/processed/common/*.parquet 2>/dev/null | tail -5 || echo "  (check data/processed/common/)"
else
    echo "Features generated for: $MODEL"
    ls -lh data/processed/${MODEL}/*.parquet 2>/dev/null | tail -5 || echo "  (check data/processed/${MODEL}/)"
fi

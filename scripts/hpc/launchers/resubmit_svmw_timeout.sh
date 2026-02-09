#!/bin/bash
# ============================================================
# SvmW タイムアウト再投入 (walltime=12:00:00)
# ============================================================
# SvmWのimbalv3/smote_plain条件が6h制限でタイムアウトしたため
# 12時間のwalltimeでLONG/SEMINARキューに再投入する
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
MISSING_FILE="/tmp/svmw_missing_clean.txt"

if [ ! -f "$MISSING_FILE" ]; then
    echo "[ERROR] Missing file not found: $MISSING_FILE"
    exit 1
fi

# Queue distribution - use LONG and SEMINAR (both support >12h walltime)
declare -A Q_LIMIT=([LONG]=15 [SEMINAR]=999)
declare -A Q_COUNT

# Count current usage
for q in LONG SEMINAR; do
    Q_COUNT[$q]=$(qstat -u s2240011 2>/dev/null | awk -v q="$q" 'NR>5 && $3==q{n++} END{print n+0}')
done

echo "============================================================"
echo "  SvmW Timeout Resubmit (walltime=12h)"
echo "  $(date)"
echo "============================================================"
echo "  Queue usage: LONG=${Q_COUNT[LONG]}/${Q_LIMIT[LONG]}, SEMINAR=${Q_COUNT[SEMINAR]}/${Q_LIMIT[SEMINAR]}"
echo "  Missing conditions: $(wc -l < "$MISSING_FILE")"
echo ""

SUBMIT_COUNT=0
FAIL_COUNT=0

pick_queue() {
    for q in LONG SEMINAR; do
        if [[ ${Q_COUNT[$q]} -lt ${Q_LIMIT[$q]} ]]; then
            echo "$q"
            return
        fi
    done
    echo "SEMINAR"  # fallback
}

while IFS= read -r condition; do
    [[ -z "$condition" ]] && continue

    # Parse: MODE_prior_SvmW_IMBAL_...
    MODE="${condition%%_prior_*}"
    TAG="${condition#${MODE}_}"

    # Extract parts for job naming
    imbal=$(echo "$TAG" | grep -oP 'prior_SvmW_\K(undersample_rus|smote_plain|imbalv3)')
    imbal_short="${imbal:0:2}"  # un/sm/im
    dist=$(echo "$TAG" | grep -oP 'knn_\K(mmd|wasserstein|dtw)')
    dist_short="${dist:0:2}"   # mm/wa/dt
    domain=$(echo "$TAG" | grep -oP '(in|out)_domain')
    dom_short="${domain:0:2}"  # in/ou
    mode_short="${MODE:0:1}"   # s/t
    seed=$(echo "$TAG" | grep -oP 's\K(42|123)$')
    ratio=$(echo "$TAG" | grep -oP 'ratio\K[0-9.]+')

    JOB_NAME="rW_${imbal_short}_${dist_short}_${dom_short}_${mode_short}_r${ratio/./}_s${seed}"
    JOB_NAME="${JOB_NAME:0:15}"

    # Map imbalance method tag → PBS CONDITION parameter
    case "$imbal" in
        imbalv3)          PBS_CONDITION="smote" ;;
        smote_plain)      PBS_CONDITION="smote_plain" ;;
        undersample_rus)  PBS_CONDITION="undersample" ;;
        *)
            echo "  [SKIP] Unknown imbal method: $imbal in $condition"
            ((FAIL_COUNT++))
            continue
            ;;
    esac

    local_queue=$(pick_queue)
    # DOMAIN must keep full form (in_domain / out_domain) for PBS script
    env_vars="MODEL=SvmW,MODE=$MODE,CONDITION=${PBS_CONDITION},DISTANCE=${dist},DOMAIN=${domain},RATIO=$ratio,SEED=$seed,RUN_EVAL=true,N_TRIALS=100"

    if qsub -N "$JOB_NAME" \
        -l select=1:ncpus=4:mem=8gb \
        -l walltime=12:00:00 \
        -q "$local_queue" \
        -v "$env_vars" \
        "$JOB_SCRIPT" 2>&1; then
        ((SUBMIT_COUNT++))
        ((Q_COUNT[$local_queue]++))
    else
        echo "  [FAIL] $condition"
        ((FAIL_COUNT++))
    fi

    sleep 0.1
done < "$MISSING_FILE"

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Submitted: $SUBMIT_COUNT"
echo "  Failed:    $FAIL_COUNT"
echo "============================================================"

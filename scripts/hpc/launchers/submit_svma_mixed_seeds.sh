#!/bin/bash
# ============================================================
# Submit SvmA MODE=mixed jobs for all 15 seeds (none currently
# valid — earlier mixed runs were invalidated by the oversampling bug).
#
# Per seed: 7 cond-variants × 3 dist × 2 dom = 42 jobs
# Total: 15 seeds × 42 = 630 jobs
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# CPU_QUEUES: short-walltime (24h) pool for baseline/undersample.  Includes
# TINY (separate QOS, max_run=5) and MS_Amorphous (separate QOS, unlimited)
# which share the same backing 256-core MS_* node pool but provide extra
# scheduler lanes when other queues fill up.
CPU_QUEUES=("SINGLE" "SMALL" "LONG" "LONG-L" "LARGE" "DEF" "XLARGE" "X2LARGE" "VM-CPU" "VM-LM" "TINY" "MS_Amorphous")
CPU_IDX=0
# Long-walltime pool for SMOTE conditions (24h+ runtime).  All entries below
# allow >=48h walltime: LONG/LONG-L are dedicated long queues; MS_*/MatStudio
# share the lcpcc-002/005/047 nodes (256 cores each) and have no per-job
# walltime cap as observed during exp3.
LONG_QUEUES=("LONG" "LONG-L" "MS_Castep" "MS_Compass" "MS_Dftbplus" "MS_Dmol3" "MS_Forcite" "MatStudio" "MS_Amorphous" "TINY")
LONG_IDX=0

ALL_SEEDS=(0 1 3 7 13 42 99 123 256 512 777 999 1234 1337 2024)
DISTANCES=(mmd dtw wasserstein)
DOMAINS=(in_domain out_domain)
declare -a COND_VARIANTS=(
    "baseline:" "smote:0.1" "smote:0.5" "smote_plain:0.1" "smote_plain:0.5"
    "undersample:0.1" "undersample:0.5"
)

short_dist() { case "$1" in dtw) echo d;; mmd) echo m;; wasserstein) echo w;; esac; }
short_dom()  { case "$1" in in_domain) echo i;; out_domain) echo o;; esac; }
short_cond() {
    case "$1" in
        baseline) echo ba;; smote) echo iv;;
        smote_plain) echo sm;; undersample) echo un;;
    esac
}

echo "[INFO] Building dedup set from current queue..."
ACTIVE_NAMES=$(qstat -u s2240011 2>/dev/null | awk 'NR>5 && $10 != "C" {print $4}' | sort -u)
in_queue() { echo "$ACTIVE_NAMES" | grep -qx "$1"; }

existing_completed() {
    find results/outputs/evaluation/SvmA -name "eval_results_SvmA_mixed_${1}.json" 2>/dev/null \
        | grep -v "_invalidated" | grep -q .
}

submit_one() {
    local COND="$1" RATIO="$2" DIST="$3" DOM="$4" SEED="$5"

    local DL=$(short_dist "$DIST"); local DM=$(short_dom "$DOM"); local CS=$(short_cond "$COND")
    local RTAG=""
    [[ -n "$RATIO" ]] && RTAG="_r${RATIO}"
    local JOB_NAME="Sa_${CS}_${DL}${DM}_mx${RTAG}_s${SEED}"

    local TAG
    case "$COND" in
        baseline)    TAG="prior_SvmA_baseline_knn_${DIST}_${DOM}_mixed_split2_s${SEED}" ;;
        smote)       TAG="prior_SvmA_imbalv3_knn_${DIST}_${DOM}_mixed_split2_subjectwise_ratio${RATIO}_s${SEED}" ;;
        smote_plain) TAG="prior_SvmA_smote_plain_knn_${DIST}_${DOM}_mixed_split2_ratio${RATIO}_s${SEED}" ;;
        undersample) TAG="prior_SvmA_undersample_rus_knn_${DIST}_${DOM}_mixed_split2_ratio${RATIO}_s${SEED}" ;;
    esac

    if existing_completed "$TAG"; then return 1; fi
    if in_queue "$JOB_NAME"; then return 1; fi

    # SMOTE-family conditions average 20.7h (mean) / 23.9h (max) and routinely
    # TIMEOUT on 24h-capped queues (DEF/SINGLE/SMALL etc.).  Pin them to queues
    # that allow >=48h walltime, and bump walltime accordingly.
    local QUEUE WALLTIME
    case "$COND" in
        smote|smote_plain)
            QUEUE="${LONG_QUEUES[$((LONG_IDX % ${#LONG_QUEUES[@]}))]}"
            ((LONG_IDX++))
            WALLTIME="48:00:00"
            ;;
        *)
            QUEUE="${CPU_QUEUES[$((CPU_IDX % ${#CPU_QUEUES[@]}))]}"
            ((CPU_IDX++))
            WALLTIME="24:00:00"
            ;;
    esac

    local CMD="qsub -N $JOB_NAME -l select=1:ncpus=8:mem=32gb -l walltime=$WALLTIME -q $QUEUE"
    CMD="$CMD -v MODEL=SvmA,CONDITION=$COND,MODE=mixed,DISTANCE=$DIST,DOMAIN=$DOM"
    CMD="$CMD,RATIO=${RATIO:-0.5},SEED=$SEED,RANKING=knn,RUN_EVAL=true"
    CMD="$CMD $JOB_SCRIPT"

    if $DRY_RUN; then echo "[DRY] $JOB_NAME [$QUEUE]"; return 0; fi
    local OUT
    OUT=$(eval "$CMD" 2>&1)
    if [[ $? -eq 0 ]]; then
        echo "[SUB] $JOB_NAME [$QUEUE] -> $OUT"
        return 0
    else
        echo "[ERR] $JOB_NAME -> $(echo "$OUT" | grep -oE 'QOS[A-Za-z]*' | head -1)"
        return 1
    fi
}

N_SUB=0; N_SKIP=0
for SEED in "${ALL_SEEDS[@]}"; do
    for COND_RATIO in "${COND_VARIANTS[@]}"; do
        COND="${COND_RATIO%:*}"
        RATIO="${COND_RATIO#*:}"
        for DIST in "${DISTANCES[@]}"; do
            for DOM in "${DOMAINS[@]}"; do
                if submit_one "$COND" "$RATIO" "$DIST" "$DOM" "$SEED"; then
                    ((N_SUB++))
                else
                    ((N_SKIP++))
                fi
                $DRY_RUN || sleep 0.2
            done
        done
    done
done

echo ""
echo "[DONE] Submitted=$N_SUB  Skipped=$N_SKIP"

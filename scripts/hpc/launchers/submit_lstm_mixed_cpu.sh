#!/bin/bash
# ============================================================
# Submit Lstm MODE=mixed jobs for all 15 seeds — CPU variant.
#
# Lstm GPU is currently broken (TF 2.19 + H100 driver 570 + CUDA 12.8u1
# can't initialize the device, even though nvidia-smi sees it). Routing
# Lstm mixed via CPU queues with the legacy pbs_prior_research_split2.sh
# instead — slower per-job (~2-3h vs 5min on GPU) but at least runs.
#
# Per seed: 7 cond-variants × 3 dist × 2 dom = 42 jobs
# Total: 15 seeds × 42 = 630 jobs (CPU)
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
cd "$PROJECT_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Unlimited MS_*/MatStudio queues first (no per-QOS cap; SvmA daemon already
# saturates SINGLE/SMALL/LONG/etc.). Capped pools come after as fallback.
CPU_QUEUES=("MS_Castep" "MS_Compass" "MS_Dftbplus" "MS_Dmol3" "MS_Forcite" "MatStudio" "MS_Amorphous" \
            "LONG-L" "LONG" "VM-LM" "DEF" "SINGLE" "SMALL" "LARGE" "XLARGE" "X2LARGE" "VM-CPU" "TINY")
CPU_IDX=0

ALL_SEEDS=(0 1 3 7 13 42 99 123 256 512 777 999 1234 1337 2024)
DISTANCES=(mmd dtw wasserstein)
DOMAINS=(in_domain out_domain)
# CPU is competitive with GPU for baseline/undersample (~24min vs ~25min on GPU)
# but ~15-30x slower for SMOTE/SMOTE_plain (6-12h CPU vs 25min GPU). Keep SMOTE
# variants on GPU via submit_lstm_mixed_seeds.sh; only baseline + undersample CPU.
declare -a COND_VARIANTS=(
    "baseline:" "undersample:0.1" "undersample:0.5"
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
    find results/outputs/evaluation/Lstm -name "eval_results_Lstm_mixed_${1}.json" 2>/dev/null \
        | grep -v "_invalidated" | grep -q .
}

submit_one() {
    local COND="$1" RATIO="$2" DIST="$3" DOM="$4" SEED="$5"

    # Issue #15 (2026-05-10): Lstm event-based labels yield natural minority
    # rate ~27%, so target_ratio=0.1 ("make minority 10% of total") requires
    # imblearn to *remove* samples from the minority class, which raises
    # ValueError("specified ratio required to remove samples ...") for both
    # SMOTE-plain and undersample_rus. The exception is swallowed by
    # train.py and the job exits 0 with no model saved — a permanent silent
    # failure regardless of how many times we resubmit. Skip these combos
    # entirely; they are reported as N/A in the results table.
    if [[ "$RATIO" == "0.1" && ("$COND" == "smote_plain" || "$COND" == "undersample") ]]; then
        return 1
    fi

    local DL=$(short_dist "$DIST"); local DM=$(short_dom "$DOM"); local CS=$(short_cond "$COND")
    local RTAG=""
    [[ -n "$RATIO" ]] && RTAG="_r${RATIO}"
    local JOB_NAME="Lm_${CS}_${DL}${DM}_mx${RTAG}_s${SEED}"

    local TAG
    case "$COND" in
        baseline)    TAG="prior_Lstm_baseline_knn_${DIST}_${DOM}_mixed_split2_s${SEED}" ;;
        smote)       TAG="prior_Lstm_imbalv3_knn_${DIST}_${DOM}_mixed_split2_subjectwise_ratio${RATIO}_s${SEED}" ;;
        smote_plain) TAG="prior_Lstm_smote_plain_knn_${DIST}_${DOM}_mixed_split2_ratio${RATIO}_s${SEED}" ;;
        undersample) TAG="prior_Lstm_undersample_rus_knn_${DIST}_${DOM}_mixed_split2_ratio${RATIO}_s${SEED}" ;;
    esac

    if existing_completed "$TAG"; then return 1; fi
    if in_queue "$JOB_NAME"; then return 1; fi

    local QUEUE="${CPU_QUEUES[$((CPU_IDX % ${#CPU_QUEUES[@]}))]}"
    ((CPU_IDX++))

    local CMD="qsub -N $JOB_NAME -l select=1:ncpus=8:mem=32gb -l walltime=24:00:00 -q $QUEUE"
    CMD="$CMD -v MODEL=Lstm,CONDITION=$COND,MODE=mixed,DISTANCE=$DIST,DOMAIN=$DOM"
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

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

# 2026-05-10: VM-CPU is full per its tight QOS (vm-cpu maxsubmitpu=10), and
# MS_Castep/Compass/Dftbplus/Forcite/MatStudio are saturated by ongoing baseline
# /undersample submissions. Front-load partitions with verified open capacity:
# DEF (33), SINGLE (31), SMALL (19), VM-LM (15), MS_Amorphous (free), LARGE (4)
# come first. MS_Forcite/MatStudio/etc. retained as overflow.
# Material-science queues (MS_*, MatStudio) intentionally excluded —
# admin (uid 25935) repeatedly scancels jobs landing there. 2026-05-11.
CPU_QUEUES=("DEF" "SINGLE" "SMALL" "VM-LM" "LARGE" \
            "DEF" "SINGLE" "SMALL" "VM-LM" \
            "XLARGE" "X2LARGE" "TINY")
CPU_IDX=0

ALL_SEEDS=(0 1 3 7 13 42 99 123 256 512 777 999 1234 1337 2024)
DISTANCES=(mmd dtw wasserstein)
DOMAINS=(in_domain out_domain)
# CPU is competitive with GPU for baseline/undersample (~24min vs ~25min on GPU)
# but ~15-30x slower for SMOTE/SMOTE_plain (6-12h CPU vs 25min GPU). Originally
# kept SMOTE on GPU only, but as of 2026-05-10 the GPU queues are saturated
# (Ls_*_mx jobs all PENDING for days) while VM-CPU is wide open (42 idle nodes,
# 1344 free cores). Add SMOTE variants here as a CPU hedge — VM-CPU/VM-LM are
# front-loaded above, and the Lm_ job-name prefix differs from the GPU's Ls_,
# so a CPU instance + GPU instance can race for the same cell. Whichever
# finishes first leaves a stale duplicate that the next audit will scancel.
# (smote_plain ratio=0.1 omitted per Issue #15 — silent imblearn ValueError.)
declare -a COND_VARIANTS=(
    "baseline:" "undersample:0.1" "undersample:0.5"
    "smote:0.1" "smote:0.5" "smote_plain:0.5"
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
ACTIVE_NAMES=$(squeue -h -u s2240011 -o '%j' 2>/dev/null | sort -u)
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

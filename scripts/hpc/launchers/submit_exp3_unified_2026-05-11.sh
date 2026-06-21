#!/bin/bash
# ============================================================
# Unified exp3 submitter (dt + mx) — 7-col TSV input
#
# Replaces:
#   - submit_exp3_final_remaining.sh  (dt only, 6-col, broken on 7-col)
#   - submit_exp3_mixed_15seeds.sh    (mx only, 6-col, used MS_* queues
#                                      via auto_resub wrappers)
#
# Input file format (TSV, 7 cols):
#   MODEL  COND  DIST  DOM  MODE  RATIO  SEED
#
#   MODEL : SvmA | SvmW | Lstm
#   COND  : baseline | smote | smote_plain | undersample_rus
#   DIST  : mmd | dtw | wasserstein
#   DOM   : in_domain | out_domain
#   MODE  : dt  (= domain_train, unified.sh)  |  mx (= mixed, split2.sh)
#   RATIO : ''  for baseline,  '0.1' or '0.5' otherwise
#   SEED  : integer
#
# Crucial: this submitter NEVER routes to Materials Studio partitions
# (MatStudio, MS_Castep, MS_Compass, MS_Dftbplus, MS_Dmol3, MS_Forcite).
# Only JAIST-published general-purpose partitions are used.
#
# Active-job dedup uses squeue (NOT qstat — qstat truncates Job_Name).
#
# Usage:
#   MISSING_LIST=/tmp/exp3_missing_2026-05-11.tsv \
#       bash scripts/hpc/launchers/submit_exp3_unified_2026-05-11.sh --dry-run
#   MISSING_LIST=/tmp/exp3_missing_2026-05-11.tsv \
#       bash scripts/hpc/launchers/submit_exp3_unified_2026-05-11.sh
# ============================================================
set -uo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Job scripts
JOB_DT_CPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
JOB_DT_GPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh"
JOB_MX_CPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
JOB_MX_GPU="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2_gpu.sh"

# Inputs
MISSING_LIST="${MISSING_LIST:?MISSING_LIST env-var required (7-col TSV)}"
DRY_RUN="${DRY_RUN:-false}"
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Allowed queues (JAIST general-purpose; NO Materials Studio partitions)
# 2026-05-11: removed LONG, LONG-L, VM-CPU, VM-LM (drain >3 days due to low
# throughput / saturated nodes). All remaining queues have UNLIMITED walltime,
# so 48h mx jobs run fine on LARGE/XLARGE/X2LARGE.
GPU_QUEUES=("GPU-1" "GPU-1A" "GPU-S" "GPU-L" "GPU-LA" "VM-GPU-L")
CPU_QUEUES=("DEF" "SINGLE" "SMALL" "LARGE" "XLARGE" "X2LARGE")
GPU_IDX=0
CPU_IDX=0

# Per-queue MaxSubmitPU (from `sacctmgr show qos`).
# Used for capacity-aware queue selection.
declare -A QMAX=(
    [SINGLE]=40 [SMALL]=30 [DEF]=40 [LONG]=15 [LONG-L]=10
    [LARGE]=15 [XLARGE]=7 [X2LARGE]=7 [VM-CPU]=10 [VM-LM]=60
    [GPU-1]=30 [GPU-1A]=20 [GPU-S]=15 [GPU-L]=5 [GPU-LA]=5 [VM-GPU-L]=3
)

# Pick a queue that currently has free submission capacity. Returns "" if none.
pick_queue_with_capacity() {
    local pool_name="$1"  # CPU_QUEUES or GPU_QUEUES
    local -n pool="$pool_name"
    local q best="" best_free=-1
    declare -A counts
    while IFS= read -r line; do
        local p="${line%% *}"
        counts[$p]=$((${counts[$p]:-0} + 1))
    done < <(squeue -h -u "$USER" -o '%P' 2>/dev/null)
    for q in "${pool[@]}"; do
        local used=${counts[$q]:-0}
        local maxv=${QMAX[$q]:-0}
        local free=$(( maxv - used ))
        if (( free > best_free )); then
            best_free=$free
            best=$q
        fi
    done
    if (( best_free <= 0 )); then
        echo ""
    else
        echo "$best"
    fi
}

N_TRIALS="${N_TRIALS:-100}"
RANKING="${RANKING:-knn}"
ACTIVE_THRESHOLD="${ACTIVE_THRESHOLD:-280}"
THROTTLE_SLEEP="${THROTTLE_SLEEP:-60}"

active_count() { squeue -h -u "$USER" 2>/dev/null | wc -l; }

throttle_wait() {
    $DRY_RUN && return 0
    local n
    n=$(active_count)
    while (( n >= ACTIVE_THRESHOLD )); do
        echo "[THROTTLE] active=$n >= $ACTIVE_THRESHOLD; sleeping ${THROTTLE_SLEEP}s..."
        sleep "$THROTTLE_SLEEP"
        n=$(active_count)
    done
}

# ---- Active-name dedup using squeue (qstat truncates names) ----
echo "[INFO] Loading active job names via squeue..."
ACTIVE_NAMES=$(squeue -h -u "$USER" -o '%j' 2>/dev/null | sort -u)
echo "[INFO] Active job names loaded: $(echo "$ACTIVE_NAMES" | grep -c .)"

in_queue() {
    local name="$1"
    grep -qx "$name" <<<"$ACTIVE_NAMES"
}

# ---- Helpers ----
short_dist() { case "$1" in dtw) echo d;; mmd) echo m;; wasserstein) echo w;; esac; }
short_dom()  { case "$1" in in_domain) echo i;; out_domain) echo o;; esac; }

# 2-letter cond prefix used in JOB_NAME (kept distinct so dt 'sm' (smote_plain)
# does NOT collide with mx 'sm' (smote_plain) nor with the imbalv3 'iv').
short_cond() {
    case "$1" in
        baseline)        echo ba;;
        smote)           echo iv;;   # subject-wise SMOTE (imbalv3)
        smote_plain)     echo sm;;
        undersample_rus) echo un;;
        *)               echo "??";;
    esac
}

# Mapping: missing-list COND -> CONDITION env value expected by job script
cond_env() {
    case "$1" in
        baseline)        echo baseline;;
        smote)           echo smote;;
        smote_plain)     echo smote_plain;;
        undersample_rus) echo undersample;;
    esac
}

# Distinct 2-letter prefix per model (Sa, Sw, Ls) — must match dedup names
model_prefix() {
    case "$1" in
        SvmA) echo Sa;;
        SvmW) echo Sw;;
        Lstm) echo Ls;;
        *)    echo "${1:0:2}";;
    esac
}

# Resource & queue selection.
# NOTE: Lstm runs on CPU job scripts (unified.sh / split2.sh both accept Lstm)
# and CPU runs are empirically faster than GPU on this workload due to
# data-loading / small-batch overhead. GPU is used only as a small fallback
# (LSTM_GPU_RATIO out of every batch of submissions).
get_resources_dt() {
    local model="$1" use_gpu="$2"
    case "$model" in
        SvmW) echo "ncpus=8:mem=32gb 48:00:00" ;;
        SvmA) echo "ncpus=8:mem=32gb 48:00:00" ;;
        Lstm)
            if [[ "$use_gpu" == "true" ]]; then
                echo "ncpus=4:ngpus=1:mem=32gb 24:00:00"
            else
                echo "ncpus=8:mem=32gb 24:00:00"
            fi
            ;;
    esac
}

get_resources_mx() {
    local model="$1" cond="$2" use_gpu="$3"
    local is_smote=false
    [[ "$cond" == "smote" || "$cond" == "smote_plain" ]] && is_smote=true
    case "$model" in
        SvmW) $is_smote && echo "ncpus=8:mem=24gb 24:00:00" || echo "ncpus=8:mem=24gb 16:00:00" ;;
        # 2026-05-13: bumped SvmA SMOTE mx walltime 48h→96h. Recent run showed
        # ~80% of these conditions hit TIMEOUT at 48h (PSO + SMOTE + 2-group
        # Mixed train pool). All current CPU partitions allow >48h walltime
        # (DEF observed running 80h+).
        SvmA) $is_smote && echo "ncpus=8:mem=48gb 96:00:00" || echo "ncpus=8:mem=48gb 30:00:00" ;;
        Lstm)
            if [[ "$use_gpu" == "true" ]]; then
                $is_smote && echo "ncpus=4:ngpus=1:mem=16gb 24:00:00" || echo "ncpus=4:ngpus=1:mem=16gb 20:00:00"
            else
                $is_smote && echo "ncpus=8:mem=24gb 24:00:00" || echo "ncpus=8:mem=24gb 20:00:00"
            fi
            ;;
    esac
}

# Fraction of Lstm jobs to send to GPU. Default = 0 (all CPU).
# Override with LSTM_GPU_FRACTION env, e.g. 0.2 sends ~20% of Lstm jobs to GPU.
LSTM_GPU_FRACTION="${LSTM_GPU_FRACTION:-0}"
LSTM_COUNTER=0

submit_one() {
    local MODEL="$1" COND="$2" DIST="$3" DOM="$4" MODE="$5" RATIO="$6" SEED="$7"

    # '-' placeholder denotes empty RATIO (baseline)
    [[ "$RATIO" == "-" ]] && RATIO=""

    local DL=$(short_dist "$DIST")
    local DM=$(short_dom  "$DOM")
    local CS=$(short_cond "$COND")
    local MP=$(model_prefix "$MODEL")
    local COND_VAL=$(cond_env "$COND")

    # Mode tag in job name: 'dt' or 'm' (matches existing convention from
    # submit_exp3_final_remaining.sh and submit_exp3_mixed_15seeds.sh)
    local MODE_TAG
    case "$MODE" in
        dt) MODE_TAG="dt" ;;
        mx) MODE_TAG="m"  ;;
        *) echo "[ERR] Unknown MODE=$MODE for $MODEL/$COND/$SEED"; return ;;
    esac

    local JOB_NAME
    if [[ "$COND" == "baseline" ]]; then
        JOB_NAME="${MP}_${CS}_${DL}${DM}_${MODE_TAG}_s${SEED}"
    else
        JOB_NAME="${MP}_${CS}_${DL}${DM}_${MODE_TAG}_r${RATIO}_s${SEED}"
    fi

    if in_queue "$JOB_NAME"; then
        echo "[SKIP] $JOB_NAME (already in queue)"
        return 1   # signals 'skipped'
    fi

    # Pick job script + resources
    local JOB_SCRIPT RES_LINE NCPUS_MEM WALLTIME QUEUE USE_GPU="false"

    # Decide GPU vs CPU for Lstm using simple deterministic ratio
    if [[ "$MODEL" == "Lstm" ]]; then
        ((LSTM_COUNTER++))
        # USE_GPU = (LSTM_COUNTER * LSTM_GPU_FRACTION) crosses an integer
        local THRESH
        THRESH=$(awk -v n="$LSTM_COUNTER" -v f="$LSTM_GPU_FRACTION" 'BEGIN{printf "%d",(n*f)}')
        local PREV_THRESH
        PREV_THRESH=$(awk -v n="$((LSTM_COUNTER-1))" -v f="$LSTM_GPU_FRACTION" 'BEGIN{printf "%d",(n*f)}')
        [[ "$THRESH" -gt "$PREV_THRESH" ]] && USE_GPU="true"
    fi

    if [[ "$MODE" == "dt" ]]; then
        if [[ "$MODEL" == "Lstm" && "$USE_GPU" == "true" ]]; then
            JOB_SCRIPT="$JOB_DT_GPU"
        else
            JOB_SCRIPT="$JOB_DT_CPU"
        fi
        RES_LINE=$(get_resources_dt "$MODEL" "$USE_GPU")
    else
        if [[ "$MODEL" == "Lstm" && "$USE_GPU" == "true" ]]; then
            JOB_SCRIPT="$JOB_MX_GPU"
        else
            JOB_SCRIPT="$JOB_MX_CPU"
        fi
        RES_LINE=$(get_resources_mx "$MODEL" "$COND" "$USE_GPU")
    fi
    NCPUS_MEM=$(echo "$RES_LINE" | cut -d' ' -f1)
    WALLTIME=$(echo "$RES_LINE" | cut -d' ' -f2)

    if [[ "$MODEL" == "Lstm" && "$USE_GPU" == "true" ]]; then
        QUEUE=$(pick_queue_with_capacity GPU_QUEUES)
        if [[ -z "$QUEUE" ]]; then
            QUEUE="${GPU_QUEUES[$((GPU_IDX % ${#GPU_QUEUES[@]}))]}"
            ((GPU_IDX++))
        fi
    else
        QUEUE=$(pick_queue_with_capacity CPU_QUEUES)
        if [[ -z "$QUEUE" ]]; then
            # All CPU queues full → wait for capacity
            if ! $DRY_RUN; then
                while [[ -z "$QUEUE" ]]; do
                    echo "[WAIT] all CPU queues at MaxSubmitPU; sleeping ${THROTTLE_SLEEP}s..."
                    sleep "$THROTTLE_SLEEP"
                    QUEUE=$(pick_queue_with_capacity CPU_QUEUES)
                done
            else
                # Dry-run: fall back to round-robin so we still print a plan
                QUEUE="${CPU_QUEUES[$((CPU_IDX % ${#CPU_QUEUES[@]}))]}"
                ((CPU_IDX++))
            fi
        fi
    fi

    # Safety: hard refuse Materials Studio partitions
    case "$QUEUE" in
        MatStudio|MS_*)
            echo "[ABORT] $JOB_NAME -> queue $QUEUE is forbidden (Materials Studio)"
            return 1 ;;
    esac

    local VARS="MODEL=$MODEL,CONDITION=$COND_VAL,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true"
    [[ "$MODE" == "mx" ]] && VARS="$VARS,MODE=mixed"
    [[ -n "$RATIO" ]] && VARS="$VARS,RATIO=$RATIO"

    local CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v $VARS $JOB_SCRIPT"

    if $DRY_RUN; then
        echo "[DRY] $JOB_NAME [$QUEUE wall=$WALLTIME] script=$(basename "$JOB_SCRIPT") cond=$COND_VAL ratio=${RATIO:-—}"
        return 0
    else
        throttle_wait
        local OUT
        OUT=$(eval "$CMD" 2>&1)
        if [[ $? -eq 0 ]]; then
            echo "[SUB] $JOB_NAME [$QUEUE] -> $OUT"
            sleep 0.3
            return 0
        else
            echo "[ERR] $JOB_NAME [$QUEUE] -> $OUT"
            return 1
        fi
    fi
}

# ---- Main loop ----
[[ ! -f "$MISSING_LIST" ]] && { echo "[FATAL] $MISSING_LIST not found"; exit 1; }
TOTAL=$(wc -l < "$MISSING_LIST")
echo "[INFO] DRY_RUN=$DRY_RUN"
echo "[INFO] MISSING_LIST=$MISSING_LIST  ($TOTAL rows)"
echo ""

N_SUBMITTED=0
N_SKIPPED=0
N_BAD=0

while IFS=$'\t' read -r MODEL COND DIST DOM MODE RATIO SEED; do
    [[ -z "${MODEL:-}" || "$MODEL" == "#"* ]] && continue
    if [[ -z "${SEED:-}" ]]; then
        echo "[ERR] malformed row (need 7 cols): MODEL=$MODEL COND=$COND DIST=$DIST DOM=$DOM MODE=$MODE RATIO=$RATIO SEED=$SEED"
        ((N_BAD++))
        continue
    fi
    if submit_one "$MODEL" "$COND" "$DIST" "$DOM" "$MODE" "$RATIO" "$SEED"; then
        ((N_SUBMITTED++))
    else
        ((N_SKIPPED++))
    fi
done < "$MISSING_LIST"

echo ""
echo "[DONE] Submitted=$N_SUBMITTED  Skipped/Err=$N_SKIPPED  Malformed=$N_BAD  Total=$TOTAL"

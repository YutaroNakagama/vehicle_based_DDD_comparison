#!/bin/bash
# ============================================================
# 再実行デーモン — oversampling バグ修正後のジョブ再投入
# ============================================================
# lstm_train() / SvmA_train() で oversampling が適用されていなかった
# バグの修正後、smote_plain / undersample_rus 条件のジョブを再投入する。
#
# 対象:
#   - Lstm: smote_plain, undersample (全3モード × 24 configs = 144 jobs)
#   - SvmA: smote_plain, undersample (全3モード × 24 configs = 144 jobs)
#   合計: 最大 288 jobs
#
# Usage:
#   nohup bash scripts/hpc/launchers/rerun_oversampling_fix.sh &
#   # ログ: /tmp/rerun_oversampling_fix.log
# ============================================================

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

JOB_SCRIPT="$PROJECT_ROOT/scripts/hpc/jobs/train/pbs_prior_research_split2.sh"
LOG="/tmp/rerun_oversampling_fix.log"
SUBMITTED_KEYS="/tmp/rerun_oversampling_fix_keys.txt"
INVALIDATED_LOG="/tmp/rerun_oversampling_fix_invalidated.txt"
POLL_INTERVAL=300  # 5 minutes
N_TRIALS=100
RANKING="knn"

# ---- エラートラップ ----
trap 'echo "[$(date +%H:%M)] TRAP: daemon exiting (line $LINENO, exit=$?)" >> "$LOG"' EXIT
trap 'echo "[$(date +%H:%M)] TRAP: received signal, exiting" >> "$LOG"; exit 1' INT TERM HUP

# ---- キュー制限 ----
declare -A QUEUE_MAX=( [SINGLE]=40 [DEFAULT]=40 [SMALL]=30 [LONG]=15 )
declare -A QUEUE_CURRENT=()
CPU_QUEUES=("SINGLE" "DEFAULT" "SMALL" "LONG")

touch "$SUBMITTED_KEYS"
touch "$INVALIDATED_LOG"

# ---- リソース定義 ----
get_resources() {
    local model="$1"
    local mode="$2"

    case "$model" in
        Lstm)
            # Lstm: ~15-30 min per domain_train, ~30-60 min for mixed
            if [[ "$mode" == "mixed" ]]; then
                echo "ncpus=8:mem=48gb 04:00:00"
            else
                echo "ncpus=8:mem=48gb 03:00:00"
            fi
            ;;
        SvmA)
            # SvmA: ~6-7 hours for mixed, ~3-4 for domain_train
            if [[ "$mode" == "mixed" ]]; then
                echo "ncpus=8:mem=48gb 48:00:00"
            else
                echo "ncpus=8:mem=48gb 30:00:00"
            fi
            ;;
    esac
}

# ---- 旧評価結果の無効化 ----
invalidate_old_eval() {
    local model="$1" tag_pattern="$2" key="$3"

    # 既に無効化済みならスキップ
    if grep -qF "$key" "$INVALIDATED_LOG" 2>/dev/null; then
        return 0
    fi

    local eval_dir="results/outputs/evaluation/$model"
    local invalid_dir="${eval_dir}/_invalidated_oversampling_bug"
    
    # Find matching eval files
    local files
    files=$(find "$eval_dir" -name "${tag_pattern}*.csv" -o -name "${tag_pattern}*.json" 2>/dev/null | grep -v _invalidated || true)

    if [[ -n "$files" ]]; then
        mkdir -p "$invalid_dir"
        local count=0
        while IFS= read -r f; do
            # Move to invalidated directory preserving job structure
            local rel_path="${f#$eval_dir/}"
            local dest_dir="$invalid_dir/$(dirname "$rel_path")"
            mkdir -p "$dest_dir"
            mv "$f" "$dest_dir/" 2>/dev/null || true
            ((count++)) || true
        done <<< "$files"
        echo "[INVAL] $key: moved $count files to $invalid_dir" >> "$LOG"
    fi

    echo "$key" >> "$INVALIDATED_LOG"
}

# ---- 新しい評価結果が存在するか確認 (旧結果無効化後) ----
has_new_eval_result() {
    local model="$1" cond="$2" dist="$3" dom="$4" mode="$5" seed="$6" ratio="$7"
    local eval_dir="results/outputs/evaluation/$model"
    
    # condition → tag mapping
    local tag
    case "$cond" in
        smote_plain) tag="smote_plain" ;;
        undersample) tag="undersample_rus" ;;
    esac
    
    local pattern="eval_results_${model}_${mode}_prior_${model}_${tag}_knn_${dist}_${dom}_${mode}_split2_*ratio${ratio}_s${seed}"
    
    # Check if matching file exists (excluding invalidated)
    find "$eval_dir" -name "${pattern}*.json" 2>/dev/null | grep -v _invalidated | grep -q .
}

# ---- キュー状態確認 ----
get_queue_counts() {
    local qstat_output
    qstat_output=$(qstat -u s2240011 2>/dev/null | tail -n +6 || true)
    
    for q in "${CPU_QUEUES[@]}"; do
        QUEUE_CURRENT[$q]=$(echo "$qstat_output" | awk -v q="$q" '$3==q' | wc -l || echo 0)
    done
}

find_available_queue() {
    for q in "${CPU_QUEUES[@]}"; do
        local current="${QUEUE_CURRENT[$q]:-0}"
        local max="${QUEUE_MAX[$q]:-0}"
        if (( current < max )); then
            echo "$q"
            return 0
        fi
    done
    return 1
}

# ---- 全実験条件を列挙 ----
ALL_JOBS=()
DISTANCES=("mmd" "dtw" "wasserstein")
DOMAINS=("in_domain" "out_domain")
SEEDS=(42 123)
RATIOS=(0.1 0.5)
MODES=("source_only" "target_only" "mixed")
# 修正が必要な条件のみ (imbalv3 は pipeline レベルで適用済みなので不要)
CONDITIONS=("smote_plain" "undersample")
# Lstm を先に (高速), SvmA を後に
MODELS=("Lstm" "SvmA")

for MODEL in "${MODELS[@]}"; do
    for MODE in "${MODES[@]}"; do
        for DIST in "${DISTANCES[@]}"; do
            for DOM in "${DOMAINS[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    for COND in "${CONDITIONS[@]}"; do
                        for RATIO in "${RATIOS[@]}"; do
                            ALL_JOBS+=("$MODEL|$COND|$DIST|$DOM|$MODE|$SEED|$RATIO")
                        done
                    done
                done
            done
        done
    done
done

echo "[$(date +%H:%M)] Daemon started. Total jobs: ${#ALL_JOBS[@]}" >> "$LOG"
echo "[$(date +%H:%M)] Models: ${MODELS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Conditions: ${CONDITIONS[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Modes: ${MODES[*]}" >> "$LOG"
echo "[$(date +%H:%M)] Polling every ${POLL_INTERVAL}s" >> "$LOG"

# ---- Step 1: 旧結果の無効化 (初回のみ) ----
echo "[$(date +%H:%M)] Invalidating old eval results..." >> "$LOG"
for job_spec in "${ALL_JOBS[@]}"; do
    IFS='|' read -r MODEL COND DIST DOM MODE SEED RATIO <<< "$job_spec"
    
    local_tag=""
    case "$COND" in
        smote_plain) local_tag="smote_plain" ;;
        undersample) local_tag="undersample_rus" ;;
    esac
    
    KEY="${MODEL}:${COND}:${DIST}:${DOM}:${MODE}:r${RATIO}:s${SEED}"
    TAG_PATTERN="eval_results_${MODEL}_${MODE}_prior_${MODEL}_${local_tag}_knn_${DIST}_${DOM}_${MODE}_split2_*ratio${RATIO}_s${SEED}"
    
    invalidate_old_eval "$MODEL" "$TAG_PATTERN" "$KEY"
done
echo "[$(date +%H:%M)] Invalidation complete." >> "$LOG"

# ---- Step 2: メインループ ----
while true; do
    get_queue_counts || true
    
    SUBMITTED_THIS_ROUND=0
    REMAINING=0
    
    for job_spec in "${ALL_JOBS[@]}"; do
        IFS='|' read -r MODEL COND DIST DOM MODE SEED RATIO <<< "$job_spec"
        
        KEY="${MODEL}:${COND}:${DIST}:${DOM}:${MODE}:r${RATIO}:s${SEED}"
        
        # Skip if new eval result already exists
        if has_new_eval_result "$MODEL" "$COND" "$DIST" "$DOM" "$MODE" "$SEED" "$RATIO"; then
            continue
        fi
        
        # Skip if already submitted
        if grep -qF "$KEY" "$SUBMITTED_KEYS" 2>/dev/null; then
            ((REMAINING++)) || true
            continue
        fi
        
        # Find available queue
        QUEUE=""
        QUEUE=$(find_available_queue) || true
        if [[ -z "$QUEUE" ]]; then
            ((REMAINING++)) || true
            continue
        fi
        
        # Get resources
        RES=$(get_resources "$MODEL" "$MODE")
        NCPUS_MEM=$(echo "$RES" | cut -d' ' -f1)
        WALLTIME=$(echo "$RES" | cut -d' ' -f2)
        
        # Generate job name (compact)
        # Model prefix: Ls (Lstm), Sv (SvmA)
        MODEL_SHORT="${MODEL:0:2}"
        COND_SHORT="${COND:0:2}"
        MODE_SHORT="${MODE:0:2}"
        DIST_SHORT="${DIST:0:1}"
        DOM_SHORT="${DOM:0:1}"
        JOB_NAME="${MODEL_SHORT}_${COND_SHORT}_${DIST_SHORT}${DOM_SHORT}_${MODE_SHORT}_r${RATIO}_s${SEED}"
        
        # Build qsub command
        VARS="MODEL=$MODEL,CONDITION=$COND,MODE=$MODE,DISTANCE=$DIST,DOMAIN=$DOM,SEED=$SEED,N_TRIALS=$N_TRIALS,RANKING=$RANKING,RUN_EVAL=true,RATIO=$RATIO"
        
        CMD="qsub -N $JOB_NAME -l select=1:$NCPUS_MEM -l walltime=$WALLTIME -q $QUEUE -v $VARS $JOB_SCRIPT"
        
        JOB_ID=$(eval "$CMD" 2>&1) || {
            echo "  [ERR] Failed: $KEY ($CMD)" >> "$LOG"
            ((REMAINING++)) || true
            continue
        }
        
        # Record submission
        echo "$KEY:$JOB_ID" >> "$SUBMITTED_KEYS"
        ((SUBMITTED_THIS_ROUND++)) || true
        
        # Update queue count
        QUEUE_CURRENT[$QUEUE]=$(( ${QUEUE_CURRENT[$QUEUE]:-0} + 1 ))
        
        echo "  [SUB] $MODEL | $COND | $MODE | $DIST | $DOM | r=${RATIO} | s$SEED | $QUEUE → $JOB_ID" >> "$LOG"
        
        sleep 0.3
    done
    
    TOTAL_QUEUED=$(qstat -u s2240011 2>/dev/null | tail -n +6 | wc -l)
    TOTAL_SUBMITTED=$(wc -l < "$SUBMITTED_KEYS")
    echo "[POLL] $(date +%H:%M) | queued=$TOTAL_QUEUED | submitted=$TOTAL_SUBMITTED | new=$SUBMITTED_THIS_ROUND | remaining=$REMAINING" >> "$LOG"
    
    if [[ "$SUBMITTED_THIS_ROUND" -eq 0 && "$REMAINING" -eq 0 ]]; then
        echo "[DONE] All oversampling-fix re-run jobs submitted or completed. Exiting." >> "$LOG"
        break
    fi
    
    if [[ "$SUBMITTED_THIS_ROUND" -eq 0 ]]; then
        echo "  (all queues full or waiting for results, sleeping...)" >> "$LOG"
    fi
    
    sleep "$POLL_INTERVAL"
done

echo "[$(date +%H:%M)] Daemon finished." >> "$LOG"

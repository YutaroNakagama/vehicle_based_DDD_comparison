#!/bin/bash
# Repair 15 missing Lstm SW-SMOTE within-eval JSONs for the priority 12-seed scope.
#
# Part 1 (9 jobs): model+cross exists → evaluate.py --eval_type within only
# Part 2 (6 jobs): CUDA OOM → full retrain + within + cross
#
# Run from WSL2 Ubuntu:
#   bash /mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison/scripts/bash/repair_lstm_swsmote_within.sh

set -euo pipefail

REPO=/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison
PYTHON=/home/ynakagama/.venv_tf_gpu/bin/python
RANKINGS=$REPO/results/analysis/exp2_domain_shift/distance/rankings/split2/knn
LOG_DIR=$REPO/logs/exp3_lstm_repair
mkdir -p "$LOG_DIR"

cd "$REPO"

export PYTHONPATH=$REPO
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

opposite_dom() {
    [[ "$1" == "in_domain" ]] && echo "out_domain" || echo "in_domain"
}

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: eval-only (model+cross exist, within missing) — 9 jobs
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PART 1: within-eval only (9 jobs) ==="

eval_within_only() {
    local dist=$1 dom=$2 ratio=$3 seed=$4 jobid=$5
    local tag="prior_Lstm_imbalv3_knn_${dist}_${dom}_domain_train_split2_subjectwise_ratio${ratio}_s${seed}"
    local target_file="${RANKINGS}/${dist}_${dom}.txt"
    local log="$LOG_DIR/repair_within_${dist}_${dom}_r${ratio}_s${seed}.log"

    echo "[$(date '+%H:%M:%S')] EVAL-WITHIN: ${dist}_${dom}_r${ratio}_s${seed}"
    $PYTHON scripts/python/evaluation/evaluate.py \
        --model Lstm \
        --tag "$tag" \
        --mode domain_train \
        --target_file "$target_file" \
        --eval_type within \
        --jobid "$jobid" \
        >"$log" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE (rc=$?): ${dist}_${dom}_r${ratio}_s${seed}"
}

eval_within_only dtw         out_domain 0.5 0    177909381087211
eval_within_only wasserstein in_domain  0.3 7    177904667963011
eval_within_only wasserstein out_domain 0.3 7    177905533112011
eval_within_only wasserstein out_domain 0.5 99   177906235732411
eval_within_only mmd         in_domain  0.5 256  177903210056711
eval_within_only dtw         in_domain  0.5 777  177908140877811
eval_within_only dtw         in_domain  0.5 1000 177908178799011
eval_within_only dtw         in_domain  0.3 1337 177907445457211
eval_within_only wasserstein out_domain 0.3 2024 177905966363911

echo "=== PART 1 complete ==="

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: full retrain + within + cross — 6 jobs (sequential, 1 GPU job at a time)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PART 2: full retrain (6 jobs, sequential) ==="

retrain_and_eval() {
    local dist=$1 dom=$2 ratio=$3 seed=$4
    local cross_dom
    cross_dom=$(opposite_dom "$dom")
    local tag="prior_Lstm_imbalv3_knn_${dist}_${dom}_domain_train_split2_subjectwise_ratio${ratio}_s${seed}"
    local jobid
    jobid="$(date +%s%3N)$$"
    local log="$LOG_DIR/repair_retrain_${dist}_${dom}_r${ratio}_s${seed}.log"

    export PBS_JOBID=$jobid

    echo "[$(date '+%H:%M:%S')] RETRAIN: ${dist}_${dom}_r${ratio}_s${seed} (jobid=$jobid)"
    {
        echo "# RETRAIN jobid=$jobid tag=$tag"
        echo "# Started $(date)"
        echo ""

        echo "=== [TRAIN] ==="
        $PYTHON scripts/python/train/train.py \
            --model Lstm \
            --mode domain_train \
            --seed "$seed" \
            --target_file "${RANKINGS}/${dist}_${dom}.txt" \
            --tag "$tag" \
            --time_stratify_labels \
            --use_oversampling \
            --oversample_method smote \
            --target_ratio "$ratio" \
            --subject_wise_oversampling
        echo "TRAIN rc=$?"

        echo "=== [EVAL within] ==="
        $PYTHON scripts/python/evaluation/evaluate.py \
            --model Lstm \
            --tag "$tag" \
            --mode domain_train \
            --target_file "${RANKINGS}/${dist}_${dom}.txt" \
            --eval_type within \
            --jobid "$jobid"
        echo "EVAL-WITHIN rc=$?"

        echo "=== [EVAL cross] ==="
        $PYTHON scripts/python/evaluation/evaluate.py \
            --model Lstm \
            --tag "$tag" \
            --mode domain_train \
            --target_file "${RANKINGS}/${dist}_${cross_dom}.txt" \
            --eval_type cross \
            --jobid "$jobid"
        echo "EVAL-CROSS rc=$?"

        echo "# Finished $(date)"
    } >>"$log" 2>&1

    echo "[$(date '+%H:%M:%S')] DONE: ${dist}_${dom}_r${ratio}_s${seed}"
}

retrain_and_eval mmd         out_domain 0.3 512
retrain_and_eval wasserstein in_domain  0.3 777
retrain_and_eval dtw         out_domain 0.5 777
retrain_and_eval wasserstein out_domain 0.3 1000
retrain_and_eval dtw         in_domain  0.5 1337
retrain_and_eval dtw         out_domain 0.5 1337

echo "=== PART 2 complete ==="
echo "=== All 15 repair jobs done. Restarting other launcher... ==="

# Restart the other launcher
nohup $PYTHON "$REPO/scripts/python/train/local_exp3_lstm_wsl2_other_launcher.py" \
    >>/home/ynakagama/launcher_other.log 2>&1 &
echo $! > /home/ynakagama/launcher_other.pid
echo "Other launcher restarted (PID=$(cat /home/ynakagama/launcher_other.pid))"

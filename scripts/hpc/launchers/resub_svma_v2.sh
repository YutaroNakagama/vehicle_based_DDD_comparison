#!/bin/bash
# =============================================================
# Daemon: Wait for current SvmA jobs, regenerate missing list, submit
# =============================================================
set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

PBS_SCRIPT="scripts/hpc/jobs/train/pbs_array_svma.sh"
LOG="scripts/hpc/logs/train/resub_svma_v2_$(date +%Y%m%d_%H%M%S).log"
POLL_INTERVAL=300  # 5 minutes

# Current batch 1 job IDs to wait for
WAIT_JOBS=("14965886" "14965887" "14965888" "14965889")

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------- Phase 1: wait for current SvmA jobs ----------
log "Phase 1: Waiting for current SvmA jobs to finish..."
log "Jobs to wait for: ${WAIT_JOBS[*]}"

while true; do
    STILL=0
    QSTAT_OUT=$(qstat -u s2240011 2>/dev/null || true)
    for jid in "${WAIT_JOBS[@]}"; do
        if echo "$QSTAT_OUT" | grep -q "$jid"; then
            STILL=$((STILL + 1))
        fi
    done
    if [[ $STILL -eq 0 ]]; then
        log "All current SvmA jobs completed."
        break
    fi
    log "Still $STILL jobs active. Waiting ${POLL_INTERVAL}s..."
    sleep $POLL_INTERVAL
done

# ---------- Phase 2: regenerate missing task list ----------
log "Phase 2: Regenerating missing task list..."

TASK_FILE="scripts/hpc/logs/train/task_files/array_svma_domain_train_remaining_v2.txt"

python3 << 'PYEOF'
import os
distances = ['mmd', 'dtw', 'wasserstein']
domains = ['in_domain', 'out_domain']
seeds = [0, 1, 3, 7, 13, 42, 99, 123, 256, 512, 777, 999, 1234, 1337, 2024]
conditions = [
    ('baseline', 0.0),
    ('smote', 0.1), ('smote', 0.5),
    ('smote_plain', 0.1), ('smote_plain', 0.5),
    ('undersample', 0.1), ('undersample', 0.5),
]
model = 'SvmA'
def make_tag(cond, dist, dom, ratio, seed):
    r = 'knn'
    if cond == 'baseline':
        return f'domain_train_prior_{model}_baseline_{r}_{dist}_{dom}_domain_train_split2_s{seed}'
    elif cond == 'smote':
        return f'domain_train_prior_{model}_imbalv3_{r}_{dist}_{dom}_domain_train_split2_subjectwise_ratio{ratio}_s{seed}'
    elif cond == 'smote_plain':
        return f'domain_train_prior_{model}_smote_plain_{r}_{dist}_{dom}_domain_train_split2_ratio{ratio}_s{seed}'
    elif cond == 'undersample':
        return f'domain_train_prior_{model}_undersample_rus_{r}_{dist}_{dom}_domain_train_split2_ratio{ratio}_s{seed}'

done = set()
train_dir = f'results/outputs/training/{model}'
for root, dirs, files in os.walk(train_dir):
    for f in files:
        if f.endswith('.csv') and 'domain_train' in f:
            tag = f.replace(f'train_results_{model}_', '').replace('.csv', '')
            done.add(tag)

missing = []
for dist in distances:
    for dom in domains:
        for cond, ratio in conditions:
            for seed in seeds:
                tag = make_tag(cond, dist, dom, ratio, seed)
                if tag not in done:
                    missing.append(f'{model}|{cond}|domain_train|{dist}|{dom}|{ratio}|{seed}|100|knn|true|unified')

outfile = 'scripts/hpc/logs/train/task_files/array_svma_domain_train_remaining_v2.txt'
with open(outfile, 'w') as f:
    for line in missing:
        f.write(line + '\n')
print(f'{model}: Done={630 - len(missing)}, Missing={len(missing)}')
PYEOF

TOTAL=$(wc -l < "$TASK_FILE")
log "Regenerated task file: $TOTAL tasks remaining."

if [[ $TOTAL -eq 0 ]]; then
    log "No missing tasks! All 630 SvmA domain_train experiments complete."
    exit 0
fi

# ---------- Phase 3: submit across CPU queues ----------
log "Phase 3: Submitting $TOTAL tasks across CPU queues..."

# Distribute: SINGLE (40/125), DEFAULT (40/125), SMALL (30/125), LONG (15/125)
# Scale to actual total
S=$((TOTAL * 40 / 125))
D=$((TOTAL * 40 / 125))
SM=$((TOTAL * 30 / 125))
L=$((TOTAL - S - D - SM))

IDX=0
submit_range() {
    local QUEUE=$1 COUNT=$2
    if [[ $COUNT -le 0 || $IDX -ge $TOTAL ]]; then return; fi
    local END=$((IDX + COUNT - 1))
    if [[ $END -ge $TOTAL ]]; then END=$((TOTAL - 1)); fi
    local ACTUAL=$((END - IDX + 1))
    if [[ $ACTUAL -eq 1 ]]; then
        JOB=$(qsub -q "$QUEUE" -v "TASK_FILE=$TASK_FILE,PBS_ARRAY_INDEX=$IDX" "$PBS_SCRIPT" 2>&1) || true
    else
        JOB=$(qsub -J "$IDX-$END" -q "$QUEUE" -v "TASK_FILE=$TASK_FILE" "$PBS_SCRIPT" 2>&1) || true
    fi
    log "  $QUEUE: $IDX-$END ($ACTUAL tasks) -> $JOB"
    IDX=$((END + 1))
}

submit_range SINGLE $S
submit_range DEFAULT $D
submit_range SMALL $SM
submit_range LONG $L

log "=== All $TOTAL tasks submitted. Daemon finished. ==="

#!/usr/bin/env bash
# Verify completed SvmA cuML jobs: counts, sizes, rc codes, and content.
set -u
LOG=/home/ynakagama/svma_cuml_full.log
EVAL=/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison/results/outputs/evaluation/SvmA

echo "===== A. Within / Cross counts ====="
W=$(find "$EVAL" -name "*_within.json" 2>/dev/null | wc -l)
C=$(find "$EVAL" -name "*_cross.json"  2>/dev/null | wc -l)
echo "Within: $W"
echo "Cross : $C"
echo ""

echo "===== B. Launcher DONE / FAIL counts ====="
echo "DONE entries with rc1=0 rc2=0: $(grep ' DONE ' "$LOG" | grep -c 'rc1=0 rc2=0')"
echo "DONE entries (any rc):         $(grep -c ' DONE ' "$LOG")"
echo "FAIL / Train FAILED entries:   $(grep -cE 'FAIL|Train FAILED' "$LOG")"
echo "ERROR entries (non-benign):    $(grep ' ERROR ' "$LOG" | grep -vc 'Model object is None')"
echo ""

echo "===== C. Sample within JSON (s7 mmd_in_dom 0.5) ====="
WPATH=$(find "$EVAL" -name "*mmd_in_domain_domain_train_split2_subjectwise_ratio0.5_s7_within.json" 2>/dev/null | head -1)
if [ -n "$WPATH" ]; then
    echo "Path: $WPATH"
    echo "Size: $(stat -c %s "$WPATH") bytes"
    python3 -c "
import json
with open('$WPATH') as f: d = json.load(f)
print('Keys:', list(d.keys())[:15])
print('roc_auc:', d.get('roc_auc'))
print('auc_pr :', d.get('auc_pr'))
print('subject_list len:', len(d.get('subject_list', [])))
"
fi
echo ""

echo "===== D. Sample cross JSON (s7 mmd_in_dom 0.5) ====="
CPATH=$(find "$EVAL" -name "*mmd_in_domain_domain_train_split2_subjectwise_ratio0.5_s7_cross.json" 2>/dev/null | head -1)
if [ -n "$CPATH" ]; then
    echo "Path: $CPATH"
    echo "Size: $(stat -c %s "$CPATH") bytes"
    python3 -c "
import json
with open('$CPATH') as f: d = json.load(f)
print('Keys:', list(d.keys())[:15])
print('roc_auc:', d.get('roc_auc'))
print('auc_pr :', d.get('auc_pr'))
print('subject_list len:', len(d.get('subject_list', [])))
print('eval_type:', d.get('eval_type'))
"
fi
echo ""

echo "===== E. Last 5 DONE entries from log ====="
grep ' DONE ' "$LOG" | tail -5

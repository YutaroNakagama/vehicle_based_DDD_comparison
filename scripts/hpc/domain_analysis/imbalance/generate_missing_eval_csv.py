#!/usr/bin/env python3
import csv
import os
import re
from fnmatch import fnmatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
EVAL_ROOT = os.path.join(ROOT, 'results', 'domain_analysis', 'imbalance_v3')
OUT = os.path.join(os.path.dirname(__file__), 'missing_eval_list.csv')

# helper to find a model file and extract 8-digit jobid
def find_suggested_jobid(tag):
    if not tag:
        return ''
    m = tag.replace('imbalv3_', '')
    # prefer RF
    for base in ['models/RF', 'models/BalancedRF', 'models']:
        base_dir = os.path.join(ROOT, base)
        if not os.path.isdir(base_dir):
            continue
        for dirpath, dirs, files in os.walk(base_dir):
            for fn in files:
                if fnmatch(fn, f'*{m}*.pkl'):
                    # extract jobid from path
                    mo = re.search(r'/([0-9]{8})/', os.path.join(dirpath, fn))
                    if mo:
                        return mo.group(1)
    return ''

rows = []
for dirpath, dirs, files in os.walk(EVAL_ROOT):
    for fn in files:
        if fn != 'eval.log':
            continue
        path = os.path.join(dirpath, fn)
        try:
            with open(path, 'r', errors='ignore') as f:
                txt = f.read()
        except Exception:
            continue
        if 'imbalv3_' in txt and 'Results saved successfully' not in txt:
            # extract tag
            tag_m = re.search(r"\[TAG\]\s*(\S+)", txt)
            tag = tag_m.group(1) if tag_m else ''
            # path components: .../evaluation/<condition>/<ranking>/<metric>/<level>/<mode>/eval.log
            parts = path.split(os.sep)
            # find 'evaluation' index
            cond = ranking = metric = level = mode = ''
            if 'evaluation' in parts:
                i = parts.index('evaluation')
                try:
                    cond = parts[i+1]
                    ranking = parts[i+2]
                    metric = parts[i+3]
                    level = parts[i+4]
                    mode = parts[i+5]
                except IndexError:
                    pass
            # reason: first ERROR-like line
            reason_m = re.search(r"(ERROR.*|No model file found.*|Aborted.*)", txt)
            reason = reason_m.group(1).replace('\n',' ') if reason_m else ''
            suggested = find_suggested_jobid(tag)
            rows.append((cond, ranking, metric, level, mode, tag, path, suggested, reason))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['condition','ranking','metric','level','mode','tag','log_path','suggested_jobid','reason'])
    for r in rows:
        writer.writerow(r)

print(f'Wrote {len(rows)} entries to {OUT}')
print('Preview:')
with open(OUT) as f:
    for i,l in enumerate(f):
        if i<20:
            print(l.rstrip())
        else:
            break

#!/usr/bin/env python3
"""
Submit remaining Combined Experiment V1 jobs using additional empty queues.
Uses LARGE, XLARGE, LONG-L which are currently empty.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
import argparse

# Configuration
N_TRIALS = 75
SEED = 42

# Parameters
IMBAL_METHODS = {
    'smote': [0.1, 0.5, 1.0],
    'balanced_rf': [None],
    'smote_balanced_rf': [0.1, 0.5, 1.0],
}

RANKING_METHODS = ['knn', 'lof']
DISTANCE_METRICS = ['mmd', 'dtw', 'wasserstein']
DOMAINS = ['in_domain', 'mid_domain', 'out_domain']
MODES = ['source_only', 'target_only']

# Use empty queues for faster execution
QUEUES = ['LARGE', 'XLARGE', 'LONG-L', 'SEMINAR', 'DEFAULT', 'SMALL']
QUEUE_WALLTIME = {
    'LARGE': '12:00:00',
    'XLARGE': '12:00:00', 
    'LONG-L': '12:00:00',
    'SEMINAR': '12:00:00',
    'DEFAULT': '12:00:00',
    'SMALL': '12:00:00',
}

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "hpc"
LOG_DIR = PROJECT_ROOT / "logs" / "combined_v1"


def create_job_name(imbal_method, ratio, ranking_method, distance_metric, domain, mode):
    imbal_abbrev = {'smote': 'sm', 'balanced_rf': 'brf', 'smote_balanced_rf': 'smbrf'}
    rank_abbrev = {'knn': 'kn', 'lof': 'lf'}
    dist_abbrev = {'mmd': 'm', 'dtw': 'd', 'wasserstein': 'w'}
    domain_abbrev = {'in_domain': 'in', 'mid_domain': 'mid', 'out_domain': 'out'}
    mode_abbrev = {'source_only': 'src', 'target_only': 'tgt'}
    
    parts = ['cb', imbal_abbrev[imbal_method]]
    if ratio is not None:
        ratio_str = '01' if ratio == 0.1 else ('05' if ratio == 0.5 else '10')
        parts.append(f"r{ratio_str}")
    parts.extend([rank_abbrev[ranking_method], dist_abbrev[distance_metric], 
                  domain_abbrev[domain], mode_abbrev[mode]])
    return '_'.join(parts)


def create_tag(imbal_method, ratio, ranking_method, distance_metric, domain, mode):
    parts = ['combined_v1', imbal_method]
    if ratio is not None:
        parts.append(f"ratio{str(ratio).replace('.', '_')}")
    parts.extend([ranking_method, distance_metric, domain, mode, f"s{SEED}"])
    return '_'.join(parts)


def get_subject_list_path(ranking_method, distance_metric, domain):
    return f"results/domain_analysis/distance/subject-wise/ranks/ranks29/{ranking_method}/{distance_metric}_{domain}.txt"


def create_train_script(imbal_method, ratio, ranking_method, distance_metric, domain, mode, queue):
    job_name = create_job_name(imbal_method, ratio, ranking_method, distance_metric, domain, mode)
    tag = create_tag(imbal_method, ratio, ranking_method, distance_metric, domain, mode)
    walltime = QUEUE_WALLTIME[queue]
    subject_list = get_subject_list_path(ranking_method, distance_metric, domain)
    model = 'BalancedRF' if imbal_method in ['balanced_rf', 'smote_balanced_rf'] else 'RF'
    
    oversample_args = ""
    if imbal_method in ['smote', 'smote_balanced_rf']:
        oversample_args = f"    --use_oversampling \\\n    --oversample_method smote"
        if ratio is not None:
            oversample_args += f" \\\n    --target_ratio {ratio}"
    
    script = f'''#!/bin/bash
#PBS -N tr_{job_name}
#PBS -q {queue}
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -o {LOG_DIR}/

set -euo pipefail
cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="/home/s2240011/git/ddd/vehicle_based_DDD_comparison:${{PYTHONPATH:-}}"
export N_TRIALS_OVERRIDE={N_TRIALS}

echo "Tag: {tag} | Queue: {queue}"

python scripts/python/train.py \\
    --model {model} \\
    --mode {mode} \\
    --tag {tag} \\
    --target_file {subject_list} \\
    --seed {SEED}'''
    
    if oversample_args:
        script += f''' \\
{oversample_args}'''
    
    script += '\n\necho "Done at $(date)"'
    return script, job_name, tag


def create_eval_script(tag, job_name):
    return f'''#!/bin/bash
#PBS -N ev_{job_name}
#PBS -q TINY
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o {LOG_DIR}/

cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison
source ~/conda/etc/profile.d/conda.sh
conda activate python310
python scripts/python/evaluate.py --tag {tag}
'''


def get_submitted_tags(log_file):
    submitted = set()
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('combined_v1'):
                submitted.add(line.split(',')[0])
    return submitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', type=str, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    submitted_tags = get_submitted_tags(args.log_file)
    print(f"Already submitted: {len(submitted_tags)}")
    
    remaining = []
    for imbal_method, ratios in IMBAL_METHODS.items():
        for ratio in ratios:
            for rm in RANKING_METHODS:
                for dm in DISTANCE_METRICS:
                    for domain in DOMAINS:
                        for mode in MODES:
                            tag = create_tag(imbal_method, ratio, rm, dm, domain, mode)
                            if tag not in submitted_tags:
                                sl = PROJECT_ROOT / get_subject_list_path(rm, dm, domain)
                                if sl.exists():
                                    remaining.append({'imbal': imbal_method, 'ratio': ratio,
                                                    'rank': rm, 'dist': dm, 'domain': domain, 'mode': mode})
    
    print(f"Remaining: {len(remaining)}")
    
    if not remaining:
        print("All done!")
        return
    
    submitted = []
    for i, job in enumerate(remaining):
        queue = QUEUES[i % len(QUEUES)]
        train_script, job_name, tag = create_train_script(
            job['imbal'], job['ratio'], job['rank'], job['dist'], job['domain'], job['mode'], queue)
        eval_script = create_eval_script(tag, job_name)
        
        train_path = SCRIPTS_DIR / f"train_{job_name}.pbs"
        eval_path = SCRIPTS_DIR / f"eval_{job_name}.pbs"
        
        with open(train_path, 'w') as f:
            f.write(train_script)
        with open(eval_path, 'w') as f:
            f.write(eval_script)
        
        if args.dry_run:
            submitted.append({'tag': tag, 'queue': queue, 'train_id': 'DRY', 'eval_id': 'DRY'})
            continue
        
        result = subprocess.run(['qsub', str(train_path)], capture_output=True, text=True, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"Error: {result.stderr.strip()}")
            break
        train_id = result.stdout.strip()
        
        result = subprocess.run(['qsub', '-W', f'depend=afterok:{train_id}', str(eval_path)],
                               capture_output=True, text=True, cwd=PROJECT_ROOT)
        eval_id = result.stdout.strip() if result.returncode == 0 else None
        
        submitted.append({'tag': tag, 'queue': queue, 'train_id': train_id, 'eval_id': eval_id})
        
        if (i + 1) % 20 == 0:
            print(f"  Submitted {i + 1}/{len(remaining)}...")
    
    # Save log
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log = PROJECT_ROOT / "logs" / f"combined_v1_remaining_{ts}.txt"
    with open(log, 'w') as f:
        for j in submitted:
            f.write(f"{j['tag']},{j['queue']},{j['train_id']},{j['eval_id']}\n")
    
    print(f"\nSubmitted {len(submitted)} jobs")
    print(f"Log: {log}")


if __name__ == '__main__':
    main()

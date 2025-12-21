#!/usr/bin/env python3
"""
Combined Experiment V1: Imbalance + Ranking
不均衡対策 + ランキングベース被験者選択の組み合わせ実験

Parameters:
- 不均衡対策: smote, balanced_rf, smote_balanced_rf
- 比率: 0.1, 0.5, 1.0 (balanced_rfは比率なし)
- ランキング手法: knn, lof
- 距離手法: mmd, dtw, wasserstein
- ドメイン: in_domain, mid_domain, out_domain
- Mode: source_only, target_only
- Seed: 42 (Phase 1)

Total: 252 training jobs
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
import itertools
import argparse

# Configuration
N_TRIALS = 75
SEED = 42  # Phase 1: single seed

# Parameters
IMBAL_METHODS = {
    'smote': [0.1, 0.5, 1.0],
    'balanced_rf': [None],  # 比率なし
    'smote_balanced_rf': [0.1, 0.5, 1.0],
}

RANKING_METHODS = ['knn', 'lof']
DISTANCE_METRICS = ['mmd', 'dtw', 'wasserstein']
DOMAINS = ['in_domain', 'mid_domain', 'out_domain']
MODES = ['source_only', 'target_only']

# Queue configuration - 4つのキューに分散
QUEUES = ['SINGLE', 'SMALL', 'DEFAULT', 'SEMINAR']
QUEUE_WALLTIME = {
    'SINGLE': '06:00:00',
    'SMALL': '12:00:00',
    'DEFAULT': '24:00:00',
    'SEMINAR': '84:00:00',
}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "hpc"
LOG_DIR = PROJECT_ROOT / "logs" / "combined_v1"


def create_job_name(imbal_method, ratio, ranking_method, distance_metric, domain, mode):
    """Create a short job name (max 15 chars for PBS)"""
    # Abbreviations
    imbal_abbrev = {
        'smote': 'sm',
        'balanced_rf': 'brf',
        'smote_balanced_rf': 'smbrf',
    }
    rank_abbrev = {
        'knn': 'kn',
        'lof': 'lf',
    }
    dist_abbrev = {
        'mmd': 'm',
        'dtw': 'd',
        'wasserstein': 'w',
    }
    domain_abbrev = {
        'in_domain': 'in',
        'mid_domain': 'mid',
        'out_domain': 'out',
    }
    mode_abbrev = {
        'source_only': 'src',
        'target_only': 'tgt',
    }
    
    parts = [
        'cb',  # combined
        imbal_abbrev[imbal_method],
    ]
    
    if ratio is not None:
        # r01, r05, r10
        if ratio == 0.1:
            ratio_str = '01'
        elif ratio == 0.5:
            ratio_str = '05'
        else:
            ratio_str = '10'
        parts.append(f"r{ratio_str}")
    
    parts.extend([
        rank_abbrev[ranking_method],
        dist_abbrev[distance_metric],
        domain_abbrev[domain],
        mode_abbrev[mode],
    ])
    
    return '_'.join(parts)


def create_tag(imbal_method, ratio, ranking_method, distance_metric, domain, mode):
    """Create a descriptive tag for the experiment"""
    parts = ['combined_v1', imbal_method]
    
    if ratio is not None:
        parts.append(f"ratio{str(ratio).replace('.', '_')}")
    
    parts.extend([
        ranking_method,
        distance_metric,
        domain,
        mode,
        f"s{SEED}",
    ])
    
    return '_'.join(parts)


def get_subject_list_path(ranking_method, distance_metric, domain):
    """Get path to the ranked subject list"""
    # ランキングで生成された被験者リストのパス
    return f"results/domain_analysis/distance/subject-wise/ranks/ranks29/{ranking_method}/{distance_metric}_{domain}.txt"


def create_train_script(imbal_method, ratio, ranking_method, distance_metric, domain, mode, queue):
    """Create training PBS script"""
    job_name = create_job_name(imbal_method, ratio, ranking_method, distance_metric, domain, mode)
    tag = create_tag(imbal_method, ratio, ranking_method, distance_metric, domain, mode)
    walltime = QUEUE_WALLTIME[queue]
    subject_list = get_subject_list_path(ranking_method, distance_metric, domain)
    
    # Build model selection
    if imbal_method in ['balanced_rf', 'smote_balanced_rf']:
        model = 'BalancedRF'
    else:
        model = 'RF'
    
    # Build oversampling arguments
    oversample_args = ""
    if imbal_method in ['smote', 'smote_balanced_rf']:
        oversample_args = f"""    --use_oversampling \\
    --oversample_method smote"""
        if ratio is not None:
            oversample_args += f""" \\
    --target_ratio {ratio}"""
    
    script = f'''#!/bin/bash
#PBS -N tr_{job_name}
#PBS -q {queue}
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime={walltime}
#PBS -j oe
#PBS -o {LOG_DIR}/

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

# Environment setup
export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${{PROJECT_ROOT}}:${{PYTHONPATH:-}}"
export N_TRIALS_OVERRIDE={N_TRIALS}

echo "=========================================="
echo "Combined Experiment V1"
echo "=========================================="
echo "Tag: {tag}"
echo "Imbalance: {imbal_method} (ratio={ratio})"
echo "Ranking: {ranking_method}"
echo "Distance: {distance_metric}"
echo "Domain: {domain}"
echo "Mode: {mode}"
echo "Queue: {queue}"
echo "Model: {model}"
echo "Subject List: {subject_list}"
echo "N_TRIALS: {N_TRIALS}"
echo "=========================================="

# Verify subject list exists
if [[ ! -f "{subject_list}" ]]; then
    echo "[ERROR] Subject list not found: {subject_list}"
    exit 1
fi

echo "[INFO] Subject count: $(wc -l < "{subject_list}")"

python scripts/python/train.py \\
    --model {model} \\
    --mode {mode} \\
    --tag {tag} \\
    --target_file {subject_list} \\
    --seed {SEED}'''
    
    if oversample_args:
        script += f''' \\
{oversample_args}'''
    
    script += '''

echo "Training completed at $(date)"
'''
    return script, job_name, tag


def create_eval_script(tag, job_name, mode, model):
    """Create evaluation PBS script"""
    script = f'''#!/bin/bash
#PBS -N ev_{job_name}
#PBS -q TINY
#PBS -l select=1:ncpus=2:mem=8gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o {LOG_DIR}/

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${{PROJECT_ROOT}}:${{PYTHONPATH:-}}"

echo "Evaluating: {tag}"
echo "Mode: {mode}"
echo "Model: {model}"

python scripts/python/evaluate.py \\
    --tag {tag} \\
    --mode {mode} \\
    --model {model}

echo "Evaluation completed at $(date)"
'''
    return script


def submit_job(script_content, script_name, dry_run=False):
    """Submit a job to PBS"""
    script_path = SCRIPTS_DIR / script_name
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    if dry_run:
        return "DRY_RUN"
    
    result = subprocess.run(
        ['qsub', str(script_path)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    
    if result.returncode != 0:
        print(f"  Error submitting {script_name}: {result.stderr}")
        return None
    
    job_id = result.stdout.strip()
    return job_id


def submit_with_dependency(train_script, eval_script, train_name, eval_name, dry_run=False):
    """Submit training job and evaluation job with dependency"""
    # Submit training job
    train_job_id = submit_job(train_script, f"{train_name}.pbs", dry_run)
    
    if train_job_id is None:
        return None, None
    
    if dry_run:
        return train_job_id, "DRY_RUN"
    
    # Submit evaluation job with dependency
    eval_script_path = SCRIPTS_DIR / f"{eval_name}.pbs"
    with open(eval_script_path, 'w') as f:
        f.write(eval_script)
    
    result = subprocess.run(
        ['qsub', '-W', f'depend=afterok:{train_job_id}', str(eval_script_path)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT
    )
    
    if result.returncode != 0:
        print(f"  Error submitting eval job: {result.stderr}")
        return train_job_id, None
    
    eval_job_id = result.stdout.strip()
    return train_job_id, eval_job_id


def main():
    parser = argparse.ArgumentParser(description='Submit Combined Experiment V1')
    parser.add_argument('--dry-run', action='store_true', help='Print jobs without submitting')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of jobs')
    parser.add_argument('--verify-only', action='store_true', help='Only verify subject lists exist')
    args = parser.parse_args()
    
    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    jobs = []
    missing_files = []
    
    for imbal_method, ratios in IMBAL_METHODS.items():
        for ratio in ratios:
            for ranking_method in RANKING_METHODS:
                for distance_metric in DISTANCE_METRICS:
                    for domain in DOMAINS:
                        for mode in MODES:
                            subject_list = get_subject_list_path(ranking_method, distance_metric, domain)
                            full_path = PROJECT_ROOT / subject_list
                            
                            if not full_path.exists():
                                missing_files.append(subject_list)
                            else:
                                jobs.append({
                                    'imbal_method': imbal_method,
                                    'ratio': ratio,
                                    'ranking_method': ranking_method,
                                    'distance_metric': distance_metric,
                                    'domain': domain,
                                    'mode': mode,
                                })
    
    print(f"=" * 60)
    print(f"Combined Experiment V1")
    print(f"=" * 60)
    print(f"Total valid jobs: {len(jobs)}")
    
    if missing_files:
        print(f"\n[WARNING] Missing subject list files: {len(set(missing_files))}")
        for f in sorted(set(missing_files)):
            print(f"  - {f}")
    
    if args.verify_only:
        print("\n[VERIFY ONLY] No jobs submitted.")
        return
    
    if args.limit:
        jobs = jobs[:args.limit]
        print(f"Limited to: {len(jobs)} jobs")
    
    # Distribute jobs across queues
    queue_assignments = []
    for i, job in enumerate(jobs):
        queue = QUEUES[i % len(QUEUES)]
        queue_assignments.append((job, queue))
    
    # Print distribution
    queue_counts = {q: 0 for q in QUEUES}
    for _, queue in queue_assignments:
        queue_counts[queue] += 1
    
    print("\nQueue distribution:")
    for q, count in queue_counts.items():
        print(f"  {q}: {count} jobs")
    
    if args.dry_run:
        print("\n[DRY RUN MODE] No jobs will be submitted")
    
    # Submit jobs
    print(f"\n{'=' * 60}")
    print("Submitting jobs...")
    print(f"{'=' * 60}")
    
    submitted_jobs = []
    for i, (job, queue) in enumerate(queue_assignments):
        train_script, job_name, tag = create_train_script(
            job['imbal_method'],
            job['ratio'],
            job['ranking_method'],
            job['distance_metric'],
            job['domain'],
            job['mode'],
            queue,
        )
        
        # Determine model for evaluation
        if job['imbal_method'] in ['balanced_rf', 'smote_balanced_rf']:
            eval_model = 'BalancedRF'
        else:
            eval_model = 'RF'
        
        eval_script = create_eval_script(tag, job_name, job['mode'], eval_model)
        
        train_id, eval_id = submit_with_dependency(
            train_script, eval_script,
            f"train_{job_name}",
            f"eval_{job_name}",
            dry_run=args.dry_run,
        )
        
        if train_id:
            submitted_jobs.append({
                'tag': tag,
                'queue': queue,
                'train_id': train_id,
                'eval_id': eval_id,
            })
            
            if not args.dry_run and (i + 1) % 50 == 0:
                print(f"  Submitted {i + 1}/{len(queue_assignments)} jobs...")
    
    # Save job log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = PROJECT_ROOT / "logs" / f"combined_v1_jobs_{timestamp}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write(f"Combined Experiment V1 - {timestamp}\n")
        f.write(f"Total jobs: {len(submitted_jobs)}\n")
        f.write("=" * 60 + "\n")
        for job in submitted_jobs:
            f.write(f"{job['tag']},{job['queue']},{job['train_id']},{job['eval_id']}\n")
    
    print(f"\n{'=' * 60}")
    print(f"Submitted {len(submitted_jobs)} training jobs")
    print(f"Job log saved to: {log_file}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

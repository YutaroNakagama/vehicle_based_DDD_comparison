#!/usr/bin/env python3
"""
Submit remaining Combined Experiment V1 jobs that failed due to queue limits.
Skips already submitted jobs based on log file.
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path
import argparse

# Import from the main script
sys.path.insert(0, str(Path(__file__).parent))
from submit_combined_experiment_v1 import (
    IMBAL_METHODS, RANKING_METHODS, DISTANCE_METRICS, DOMAINS, MODES,
    QUEUES, QUEUE_WALLTIME, N_TRIALS, SEED, PROJECT_ROOT, SCRIPTS_DIR, LOG_DIR,
    create_job_name, create_tag, get_subject_list_path,
    create_train_script, create_eval_script, submit_with_dependency
)


def get_submitted_tags(log_file):
    """Read submitted tags from log file"""
    submitted = set()
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('combined_v1'):
                tag = line.split(',')[0]
                submitted.add(tag)
    return submitted


def main():
    parser = argparse.ArgumentParser(description='Submit remaining Combined V1 jobs')
    parser.add_argument('--log-file', type=str, required=True, 
                        help='Path to the log file from previous submission')
    parser.add_argument('--dry-run', action='store_true', help='Print jobs without submitting')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of jobs to submit')
    args = parser.parse_args()
    
    # Get already submitted tags
    submitted_tags = get_submitted_tags(args.log_file)
    print(f"Already submitted: {len(submitted_tags)} jobs")
    
    # Generate all jobs and filter out already submitted
    remaining_jobs = []
    
    for imbal_method, ratios in IMBAL_METHODS.items():
        for ratio in ratios:
            for ranking_method in RANKING_METHODS:
                for distance_metric in DISTANCE_METRICS:
                    for domain in DOMAINS:
                        for mode in MODES:
                            tag = create_tag(imbal_method, ratio, ranking_method, 
                                           distance_metric, domain, mode)
                            
                            if tag not in submitted_tags:
                                subject_list = get_subject_list_path(ranking_method, 
                                                                     distance_metric, domain)
                                full_path = PROJECT_ROOT / subject_list
                                
                                if full_path.exists():
                                    remaining_jobs.append({
                                        'imbal_method': imbal_method,
                                        'ratio': ratio,
                                        'ranking_method': ranking_method,
                                        'distance_metric': distance_metric,
                                        'domain': domain,
                                        'mode': mode,
                                    })
    
    print(f"Remaining jobs to submit: {len(remaining_jobs)}")
    
    if args.limit:
        remaining_jobs = remaining_jobs[:args.limit]
        print(f"Limited to: {len(remaining_jobs)} jobs")
    
    if not remaining_jobs:
        print("All jobs already submitted!")
        return
    
    # Distribute across queues
    queue_assignments = []
    for i, job in enumerate(remaining_jobs):
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
    print("Submitting remaining jobs...")
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
        eval_script = create_eval_script(tag, job_name)
        
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
            
            if not args.dry_run and (i + 1) % 20 == 0:
                print(f"  Submitted {i + 1}/{len(queue_assignments)} jobs...")
    
    # Save job log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = PROJECT_ROOT / "logs" / f"combined_v1_remaining_jobs_{timestamp}.txt"
    
    with open(log_file, 'w') as f:
        f.write(f"Combined Experiment V1 - Remaining Jobs - {timestamp}\n")
        f.write(f"Total jobs: {len(submitted_jobs)}\n")
        f.write("=" * 60 + "\n")
        for job in submitted_jobs:
            f.write(f"{job['tag']},{job['queue']},{job['train_id']},{job['eval_id']}\n")
    
    print(f"\n{'=' * 60}")
    print(f"Submitted {len(submitted_jobs)} remaining jobs")
    print(f"Job log saved to: {log_file}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

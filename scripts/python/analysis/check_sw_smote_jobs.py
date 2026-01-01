#!/usr/bin/env python3
"""Check completion status of SW-SMOTE full test jobs."""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path("/home/s2240011/git/ddd/vehicle_based_DDD_comparison")
LOG_DIR = PROJECT_ROOT / "logs" / "hpc"

# Job log files
JOB_LOG_FILES = [
    PROJECT_ROOT / "logs" / "sw_smote_fixed_jobs_20251225_234627.txt",
    PROJECT_ROOT / "logs" / "sw_smote_fixed_remaining_20251226_010543.txt",
    PROJECT_ROOT / "logs" / "sw_smote_rerun_20251229_205803.txt",  # Rerun failed jobs
    PROJECT_ROOT / "logs" / "sw_smote_rerun2_20251230_061344.txt",  # Rerun 2 (12h walltime)
]

def parse_job_logs():
    """Parse job submission logs to get job IDs and tags."""
    jobs = {}
    for log_file in JOB_LOG_FILES:
        if log_file.exists():
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('14'):  # Job ID starts with 14
                        parts = line.split()
                        if len(parts) >= 2:
                            job_id = parts[0]
                            tag = parts[1]
                            queue = parts[2] if len(parts) >= 3 else "UNKNOWN"
                            jobs[job_id] = {"tag": tag, "queue": queue}
    return jobs

def check_job_completion(job_id):
    """Check if a job completed successfully."""
    log_file = LOG_DIR / f"{job_id}.OU"
    if not log_file.exists():
        return False, "LOG_MISSING", None
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Check walltime exceeded
        if "PBS: job killed: walltime" in content:
            # Extract requested walltime
            match = re.search(r"walltime=(\d+:\d+:\d+)", content)
            walltime = match.group(1) if match else "unknown"
            return False, "WALLTIME_EXCEEDED", walltime
        
        if "TRAINING DONE" in content or "=== TRAINING" in content:
            return True, "SUCCESS", None
        elif "Error" in content or "Exception" in content or "Traceback" in content:
            return False, "ERROR", None
        else:
            return False, "INCOMPLETE", None

def main():
    jobs = parse_job_logs()
    print(f"Total jobs in submission logs: {len(jobs)}")
    print()
    
    success = []
    failed = []
    
    for job_id, info in jobs.items():
        completed, status, detail = check_job_completion(job_id)
        if completed:
            success.append((job_id, info["tag"], info["queue"]))
        else:
            failed.append((job_id, info["tag"], info["queue"], status, detail))
    
    print("=" * 70)
    print("SW-SMOTE Full Test Completion Summary")
    print("=" * 70)
    print(f"Total Jobs:    {len(jobs)}")
    print(f"Completed:     {len(success)}")
    print(f"Failed:        {len(failed)}")
    print("=" * 70)
    
    if failed:
        print()
        print("Failed/Incomplete Jobs:")
        print("-" * 70)
        
        # Group by failure reason
        by_reason = {}
        for job_id, tag, queue, status, detail in failed:
            if status not in by_reason:
                by_reason[status] = []
            by_reason[status].append((job_id, tag, queue, detail))
        
        for reason, jobs_list in by_reason.items():
            print(f"\n{reason} ({len(jobs_list)} jobs):")
            for job_id, tag, queue, detail in jobs_list:
                detail_str = f" (walltime={detail})" if detail else ""
                print(f"  {job_id}: {tag} [Queue: {queue}]{detail_str}")
    
    # Check expected vs actual
    print()
    print("=" * 70)
    print("Expected Test Cases Analysis")
    print("=" * 70)
    
    # Expected parameters
    ratios = ["0_1", "0_5", "1_0"]
    rankings = ["knn", "lof"]
    distances = ["dtw", "mmd", "wasserstein"]
    domains = ["in_domain", "mid_domain", "out_domain"]
    modes = ["pooled", "source_only", "target_only"]
    
    expected = 3 * 2 * 3 * 3 * 3  # 162
    print(f"Expected total: {expected}")
    
    # Find missing combinations
    existing_tags = set(info["tag"] for info in jobs.values())
    missing = []
    
    for ratio in ratios:
        for ranking in rankings:
            for distance in distances:
                for domain in domains:
                    for mode in modes:
                        expected_tag = f"swsmote_v2_r{ratio}_{ranking}_{distance}_{domain}_{mode}_s42"
                        if expected_tag not in existing_tags:
                            missing.append(expected_tag)
    
    if missing:
        print(f"Missing test cases (not submitted): {len(missing)}")
        for tag in missing:
            print(f"  - {tag}")
    else:
        print("All expected test cases were submitted!")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Submitted:          {len(jobs)}/162")
    print(f"  Completed:          {len(success)}")
    print(f"  Failed (walltime):  {len([f for f in failed if f[3] == 'WALLTIME_EXCEEDED'])}")
    print(f"  Other failures:     {len([f for f in failed if f[3] != 'WALLTIME_EXCEEDED'])}")
    print(f"  Not submitted:      {len(missing)}")
    print()
    
    # Action items
    need_rerun = len(failed) + len(missing)
    if need_rerun > 0:
        print(f"ACTION REQUIRED: {need_rerun} jobs need to be (re)submitted")
        print("  - Failed jobs need longer walltime (recommend 4-12 hours)")
        print("  - Missing jobs were never submitted")

if __name__ == "__main__":
    main()

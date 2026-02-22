#!/usr/bin/env python3
"""
Migrate queued CPU jobs from crowded queues to LARGE queue.

LARGE queue: ncpus_min=128, max_run=3/user, max_queued=15/user, walltime_max=168h

Usage:
    python migrate_to_large.py [--max-jobs N] [--dry-run]
"""
import subprocess
import re
import sys
import time
import argparse


def run_cmd(cmd):
    """Run a shell command and return stdout."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip(), result.returncode


def get_queued_cpu_jobs():
    """Get all queued CPU (non-GPU) job IDs."""
    out, _ = run_cmd("qstat -u s2240011 | tail -n +6 | grep ' Q ' | grep -v GPU | awk '{print $1}'")
    if not out:
        return []
    return [jid.strip() for jid in out.strip().split('\n') if jid.strip()]


def get_large_queue_count():
    """Count our jobs currently in LARGE queue."""
    out, _ = run_cmd("qstat -u s2240011 | tail -n +6 | awk '$3==\"LARGE\"' | wc -l")
    return int(out.strip()) if out.strip() else 0


def parse_job(jid):
    """Extract job parameters from qstat -xf output."""
    out, rc = run_cmd(f"qstat -xf {jid}")
    if rc != 0 or not out:
        return None

    # qstat continuation lines start with \t - join them to previous line
    clean = re.sub(r'\n\t', '', out)

    info = {}

    # Job_Name
    m = re.search(r'Job_Name\s*=\s*(\S+)', clean)
    info['name'] = m.group(1) if m else None

    # Current queue
    m = re.search(r'^\s*queue\s*=\s*(\S+)', clean, re.MULTILINE)
    info['queue'] = m.group(1) if m else None

    # Walltime
    m = re.search(r'Resource_List\.walltime\s*=\s*(\S+)', clean)
    info['walltime'] = m.group(1) if m else '48:00:00'

    # Memory
    m = re.search(r'Resource_List\.mem\s*=\s*(\S+)', clean)
    info['mem'] = m.group(1) if m else '32gb'

    # Extract Submit_arguments to get -v variables and script path
    m = re.search(r'Submit_arguments\s*=\s*(.+?)(?=\n\s*\w+\s*=)', clean, re.DOTALL)
    if m:
        submit_args = m.group(1).strip()

        # Extract -v variables
        vm = re.search(r'-v\s+(\S+)', submit_args)
        if vm:
            info['vars'] = vm.group(1).rstrip(',')
        
        # Extract PBS script path
        sm = re.search(r'(/home/\S+\.sh)', submit_args)
        if sm:
            info['script'] = sm.group(1)
    
    # Fallback: extract variables from Variable_List
    if 'vars' not in info or not info.get('vars'):
        vm = re.search(r'Variable_List\s*=\s*(.+?)(?=\n\s*comment)', clean, re.DOTALL)
        if vm:
            all_vars = vm.group(1).replace('\n', '').replace(' ', '')
            # Extract only our custom variables
            custom = []
            for pair in all_vars.split(','):
                key = pair.split('=')[0] if '=' in pair else ''
                if key in ('MODEL', 'CONDITION', 'DISTANCE', 'DOMAIN', 'RATIO', 
                          'SEED', 'N_TRIALS', 'RANKING', 'RUN_EVAL', 'MODE'):
                    custom.append(pair)
            info['vars'] = ','.join(custom)

    # Determine script if not found
    if 'script' not in info or not info.get('script'):
        # Check if MODE is in variables (mixed → split2, else → unified)
        if info.get('vars') and 'MODE=' in info['vars']:
            info['script'] = '/home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_split2.sh'
        else:
            info['script'] = '/home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh'

    return info


def main():
    parser = argparse.ArgumentParser(description='Migrate queued CPU jobs to LARGE queue')
    parser.add_argument('--max-jobs', type=int, default=15, help='Max jobs to migrate (default: 15)')
    parser.add_argument('--dry-run', action='store_true', help='Only show what would happen')
    args = parser.parse_args()

    print("=" * 60)
    print("Migrate queued CPU jobs → LARGE queue (ncpus=128)")
    print(f"Max jobs: {args.max_jobs}, Dry run: {args.dry_run}")
    print("=" * 60)

    # Get queued jobs
    queued = get_queued_cpu_jobs()
    print(f"\nTotal queued CPU jobs: {len(queued)}")

    # Check LARGE queue availability
    large_count = get_large_queue_count()
    large_avail = 15 - large_count
    print(f"Current LARGE queue (ours): {large_count}")
    print(f"Available LARGE slots: {large_avail}")

    if large_avail <= 0:
        print("\n[WARN] LARGE queue full (15 jobs). Cannot migrate.")
        return

    max_migrate = min(args.max_jobs, large_avail, len(queued))
    print(f"Will migrate up to: {max_migrate} jobs")

    migrated = 0
    failed = 0

    for jid in queued:
        if migrated >= max_migrate:
            print(f"\n[INFO] Reached limit ({max_migrate}). Done.")
            break

        print(f"\n--- {jid} ---")
        info = parse_job(jid)

        if not info or not info.get('name') or not info.get('vars') or not info.get('script'):
            print(f"  [SKIP] Could not parse job: {info}")
            failed += 1
            continue

        print(f"  Name:    {info['name']}")
        print(f"  From:    {info['queue']} → LARGE")
        print(f"  Mem:     {info['mem']}")
        print(f"  Wall:    {info['walltime']}")
        print(f"  Script:  {info['script'].split('/')[-1]}")
        print(f"  Vars:    {info['vars']}")

        # Build qsub command
        qsub_cmd = (
            f"qsub -N {info['name']} "
            f"-l select=1:ncpus=128:mem={info['mem']} "
            f"-l walltime={info['walltime']} "
            f"-q LARGE "
            f"-v {info['vars']} "
            f"{info['script']}"
        )
        print(f"  Cmd:     {qsub_cmd}")

        if args.dry_run:
            print("  [DRY-RUN] Would cancel and resubmit")
            migrated += 1
            continue

        # Cancel original job
        print(f"  Canceling {jid}...")
        _, rc = run_cmd(f"qdel {jid}")
        if rc != 0:
            print(f"  [SKIP] Failed to cancel (may have started running)")
            failed += 1
            continue
        time.sleep(0.3)

        # Submit to LARGE
        new_jid, rc = run_cmd(qsub_cmd)
        if rc != 0:
            print(f"  [ERROR] Submit failed: {new_jid}")
            failed += 1
            continue

        print(f"  → Submitted as {new_jid}")
        migrated += 1

    print(f"\n{'=' * 60}")
    print(f"Migration {'(dry-run) ' if args.dry_run else ''}complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Failed:   {failed}")
    print(f"  Remaining queued CPU: {len(queued) - migrated}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

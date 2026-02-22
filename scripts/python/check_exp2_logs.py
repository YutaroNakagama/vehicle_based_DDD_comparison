#!/usr/bin/env python3
"""Check HPC job logs for abnormal termination - RF/BalancedRF exp2 jobs."""
import os
import glob
import re
from collections import defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT, 'scripts/hpc/logs/domain')

def get_rf_job_ids():
    """Get all job IDs from RF and BalancedRF model directories."""
    ids = {}
    for model in ['RF', 'BalancedRF']:
        base = os.path.join(PROJECT, f'models/{model}')
        if os.path.exists(base):
            job_dirs = [d for d in os.listdir(base) if d.isdigit()]
            for jd in job_dirs:
                ids[jd] = model
    return ids

def check_log(logfile):
    """Check a log file for errors and extract status."""
    try:
        with open(logfile, 'r', errors='replace') as f:
            content = f.read()
    except:
        return 'UNREADABLE', {}
    
    info = {}
    
    # Extract resource usage
    walltime_match = re.search(r'walltime=(\d+:\d+:\d+)', content.split('Used Resource')[-1] if 'Used Resource' in content else '')
    if walltime_match:
        info['used_walltime'] = walltime_match.group(1)
    
    req_walltime = re.search(r'walltime=(\d+:\d+:\d+)', content.split('Requested Resource')[0] if 'Requested Resource' in content else '')
    
    # Check for known error patterns
    if 'walltime' in content.lower() and ('exceeded' in content.lower() or 'KILLED' in content):
        return 'WALLTIME_EXCEEDED', info
    
    if 'MemoryError' in content or 'OOM' in content:
        return 'OOM', info
    
    if 'SIGKILL' in content or 'SIGTERM' in content:
        return 'SIGNAL_KILLED', info
    
    # Check for Python tracebacks
    traceback_count = content.count('Traceback (most recent call last)')
    if traceback_count > 0:
        # Extract last traceback
        last_tb_idx = content.rfind('Traceback (most recent call last)')
        last_error = content[last_tb_idx:last_tb_idx+500]
        info['last_traceback'] = last_error[:200]
        
        # But check if eval output was still produced (recovered)
        if 'Evaluation complete' in content or 'eval_results' in content or 'Saved evaluation' in content:
            return 'ERROR_BUT_COMPLETED', info
        return 'TRACEBACK', info
    
    # Check for successful completion markers
    if 'Used Resource' in content:
        return 'COMPLETED', info
    
    if content.strip() == '':
        return 'EMPTY_LOG', info
    
    return 'UNKNOWN', info


def main():
    rf_ids = get_rf_job_ids()
    print(f"Total RF/BalancedRF job directories: {len(rf_ids)}")
    
    # Check logs
    status_counts = defaultdict(int)
    status_details = defaultdict(list)
    no_log = 0
    checked = 0
    
    for jid, model in sorted(rf_ids.items()):
        logfile = os.path.join(LOG_DIR, f'{jid}.spcc-adm1.OU')
        if not os.path.exists(logfile):
            no_log += 1
            continue
        
        checked += 1
        status, info = check_log(logfile)
        status_counts[status] += 1
        
        if status not in ('COMPLETED', 'ERROR_BUT_COMPLETED'):
            status_details[status].append((jid, model, info))
    
    print(f"\nLogs checked: {checked}")
    print(f"No log file: {no_log}")
    print(f"\nStatus breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status}: {count}")
    
    # Show problematic jobs
    for status in ['WALLTIME_EXCEEDED', 'OOM', 'SIGNAL_KILLED', 'TRACEBACK', 'EMPTY_LOG', 'UNKNOWN']:
        if status in status_details:
            print(f"\n--- {status} jobs ---")
            for jid, model, info in status_details[status][:10]:
                # Check if this job has eval output
                eval_path = os.path.join(PROJECT, f'results/outputs/evaluation/{model}', jid)
                has_eval = os.path.exists(eval_path) and bool(glob.glob(os.path.join(eval_path, '**/*split2*.json'), recursive=True))
                eval_str = "HAS_EVAL" if has_eval else "NO_EVAL"
                print(f"  Job {jid} ({model}): {eval_str} {info.get('used_walltime', '')}")
                if 'last_traceback' in info:
                    print(f"    {info['last_traceback'][:120]}")
    
    # Cross-check: jobs with model dir but no eval output
    print(f"\n{'='*60}")
    print("CROSS-CHECK: Jobs with model but NO eval (split2)")
    print(f"{'='*60}")
    missing_eval = []
    for jid, model in sorted(rf_ids.items()):
        model_dir = os.path.join(PROJECT, f'models/{model}', jid)
        eval_base = os.path.join(PROJECT, f'results/outputs/evaluation/{model}', jid)
        
        has_model = bool(glob.glob(os.path.join(model_dir, '**/*.pkl'), recursive=True))
        has_eval = os.path.exists(eval_base) and bool(glob.glob(os.path.join(eval_base, '**/*split2*.json'), recursive=True))
        
        if has_model and not has_eval:
            # Check if it's a split2 job (not exp1)
            pkl_files = glob.glob(os.path.join(model_dir, '**/*.pkl'), recursive=True)
            is_split2 = any('split2' in os.path.basename(f) for f in pkl_files)
            if is_split2:
                missing_eval.append((jid, model))
    
    print(f"  Split2 jobs with model but no eval: {len(missing_eval)}")
    for jid, model in missing_eval[:20]:
        logfile = os.path.join(LOG_DIR, f'{jid}.spcc-adm1.OU')
        if os.path.exists(logfile):
            status, info = check_log(logfile)
            print(f"    Job {jid} ({model}): log_status={status} {info.get('used_walltime', '')}")
        else:
            print(f"    Job {jid} ({model}): NO LOG")

    # Sample check: verify a few completed jobs have reasonable metrics
    print(f"\n{'='*60}")
    print("SAMPLE METRIC CHECK (5 random completed jobs)")
    print(f"{'='*60}")
    import json
    import random
    
    all_evals = []
    for model in ['RF', 'BalancedRF']:
        evals = glob.glob(os.path.join(PROJECT, f'results/outputs/evaluation/{model}/**/*split2*.json'), recursive=True)
        all_evals.extend(evals)
    
    sample = random.sample(all_evals, min(5, len(all_evals)))
    for f in sample:
        with open(f) as fh:
            d = json.load(fh)
        fname = os.path.basename(f)[:75]
        acc = d.get('accuracy', 'N/A')
        f1 = d.get('f1', d.get('f1_score', 'N/A'))
        auc = d.get('auc_pr', 'N/A')
        
        # Check confusion matrix
        cm = d.get('confusion_matrix', None)
        cm_shape = f"{len(cm)}x{len(cm[0])}" if cm and isinstance(cm, list) and len(cm) > 0 else 'N/A'
        
        print(f"  {fname}")
        print(f"    acc={acc}, f1={f1}, auc_pr={auc}, cm={cm_shape}")
        
        # Sanity: accuracy should be between 0 and 1
        if isinstance(acc, (int, float)) and (acc < 0 or acc > 1):
            print(f"    WARNING: accuracy out of range!")
        if isinstance(f1, (int, float)) and (f1 < 0 or f1 > 1):
            print(f"    WARNING: f1 out of range!")


if __name__ == '__main__':
    main()

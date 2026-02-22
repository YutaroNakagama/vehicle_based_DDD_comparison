#!/usr/bin/env python3
"""Health check for exp2 (RF domain shift) completed jobs."""
import json
import glob
import os
import sys
from collections import defaultdict

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_eval_jsons():
    """Check all split2 eval JSONs for RF and BalancedRF."""
    print("=" * 60)
    print("1. EVALUATION JSON HEALTH CHECK")
    print("=" * 60)

    results = {}
    for model in ['RF', 'BalancedRF']:
        base = os.path.join(PROJECT, f'results/outputs/evaluation/{model}')
        files = glob.glob(os.path.join(base, '**/*split2*.json'), recursive=True)
        
        valid = 0
        invalid = []
        empty = []
        truncated = []
        
        for f in files:
            sz = os.path.getsize(f)
            if sz == 0:
                empty.append(f)
                continue
            
            # Quick structural check
            with open(f, 'rb') as fh:
                first = fh.read(1)
                fh.seek(-2, 2)
                last = fh.read(2).strip()
            
            if first != b'{' or not last.endswith(b'}'):
                truncated.append((f, sz, first, last))
                continue
            
            # Full JSON parse for a sample
            try:
                with open(f) as fh:
                    d = json.load(fh)
                if not d:
                    empty.append(f)
                else:
                    valid += 1
            except json.JSONDecodeError as e:
                invalid.append((f, str(e)[:80]))
            except Exception as e:
                invalid.append((f, str(e)[:80]))
        
        results[model] = {
            'total': len(files),
            'valid': valid,
            'invalid': invalid,
            'empty': empty,
            'truncated': truncated
        }
        
        print(f"\n{model}:")
        print(f"  Total files: {len(files)}")
        print(f"  Valid: {valid}")
        print(f"  Empty: {len(empty)}")
        print(f"  Truncated: {len(truncated)}")
        print(f"  Invalid JSON: {len(invalid)}")
        
        for cat, items in [('Empty', empty), ('Truncated', truncated), ('Invalid', invalid)]:
            for item in items:
                if isinstance(item, tuple):
                    print(f"    {cat}: {item}")
                else:
                    print(f"    {cat}: {item}")
    
    return results


def check_eval_content_sample():
    """Spot-check eval JSON contents for expected keys."""
    print("\n" + "=" * 60)
    print("2. EVAL CONTENT STRUCTURE CHECK (sample)")
    print("=" * 60)
    
    expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix']
    alt_keys = ['test_accuracy', 'test_f1', 'balanced_accuracy']
    
    for model in ['RF', 'BalancedRF']:
        base = os.path.join(PROJECT, f'results/outputs/evaluation/{model}')
        files = glob.glob(os.path.join(base, '**/*split2*.json'), recursive=True)
        
        if not files:
            print(f"\n{model}: No files found")
            continue
        
        # Sample first 5 files
        sample = files[:5]
        print(f"\n{model}: Sampling {len(sample)} of {len(files)} files")
        
        for f in sample:
            with open(f) as fh:
                d = json.load(fh)
            keys = set(d.keys()) if isinstance(d, dict) else set()
            fname = os.path.basename(f)
            has_expected = any(k in keys for k in expected_keys + alt_keys)
            print(f"  {fname[:70]}")
            print(f"    Keys: {sorted(keys)[:10]}{'...' if len(keys)>10 else ''}")
            print(f"    Has metrics: {has_expected}")


def check_model_artifacts():
    """Check that model pkl/joblib files exist for completed jobs."""
    print("\n" + "=" * 60)
    print("3. MODEL ARTIFACT CHECK")
    print("=" * 60)
    
    for model in ['RF', 'BalancedRF']:
        base = os.path.join(PROJECT, f'models/{model}')
        if not os.path.exists(base):
            print(f"\n{model}: models directory not found")
            continue
        
        job_dirs = [d for d in os.listdir(base) if d.isdigit()]
        
        has_model = 0
        missing_model = []
        has_eval = 0
        
        for jd in job_dirs:
            jpath = os.path.join(base, jd)
            # Check for model files
            model_files = glob.glob(os.path.join(jpath, '**/*.pkl'), recursive=True) + \
                          glob.glob(os.path.join(jpath, '**/*.joblib'), recursive=True)
            if model_files:
                has_model += 1
            else:
                # Check subdirectories with [n] suffix
                subdirs = glob.glob(os.path.join(jpath, f'{jd}*'))
                any_model = False
                for sd in subdirs:
                    m = glob.glob(os.path.join(sd, '**/*.pkl'), recursive=True) + \
                        glob.glob(os.path.join(sd, '**/*.joblib'), recursive=True)
                    if m:
                        any_model = True
                        break
                if any_model:
                    has_model += 1
                else:
                    missing_model.append(jd)
            
            # Check for eval
            eval_files = glob.glob(os.path.join(jpath, '**/*split2*.json'), recursive=True)
            if eval_files:
                has_eval += 1
        
        print(f"\n{model}:")
        print(f"  Job directories: {len(job_dirs)}")
        print(f"  With model files: {has_model}")
        print(f"  With eval files: {has_eval}")
        print(f"  Missing model files: {len(missing_model)}")
        if missing_model[:10]:
            print(f"    Examples: {missing_model[:10]}")


def check_hpc_job_logs():
    """Check HPC job output/error files for abnormal termination."""
    print("\n" + "=" * 60)
    print("4. HPC JOB LOG CHECK (error patterns)")
    print("=" * 60)
    
    error_patterns = [
        'Traceback', 'Error', 'KILLED', 'OOM', 'Segmentation fault',
        'walltime', 'exceeded', 'SIGTERM', 'SIGKILL', 'MemoryError',
        'killed', 'Terminated'
    ]
    
    for model in ['RF', 'BalancedRF']:
        base = os.path.join(PROJECT, f'models/{model}')
        if not os.path.exists(base):
            continue
        
        job_dirs = [d for d in os.listdir(base) if d.isdigit()]
        
        error_jobs = []
        normal_jobs = 0
        no_logs = 0
        walltime_kills = 0
        
        for jd in sorted(job_dirs):
            jpath = os.path.join(base, jd)
            # Find .e (stderr) files
            err_files = glob.glob(os.path.join(jpath, '*.e*')) + \
                        glob.glob(os.path.join(jpath, f'{jd}*/*.e*'))
            
            if not err_files:
                no_logs += 1
                continue
            
            job_has_error = False
            for ef in err_files:
                try:
                    with open(ef, 'r', errors='replace') as fh:
                        content = fh.read()
                    
                    if 'walltime' in content.lower() or 'exceeded' in content.lower():
                        walltime_kills += 1
                        job_has_error = True
                        continue
                    
                    if 'Traceback' in content or 'KILLED' in content or 'MemoryError' in content:
                        # Check if it also has eval output (recovered)
                        eval_files = glob.glob(os.path.join(jpath, '**/*split2*.json'), recursive=True)
                        if not eval_files:
                            error_jobs.append((jd, ef, 'Error without eval output'))
                            job_has_error = True
                except:
                    pass
            
            if not job_has_error:
                normal_jobs += 1
        
        print(f"\n{model}:")
        print(f"  Total job dirs: {len(job_dirs)}")
        print(f"  Normal (no errors): {normal_jobs}")
        print(f"  Walltime exceeded: {walltime_kills}")
        print(f"  Errors without output: {len(error_jobs)}")
        print(f"  No log files: {no_logs}")
        
        if error_jobs:
            print(f"  ERROR JOBS:")
            for jid, ef, reason in error_jobs[:10]:
                print(f"    Job {jid}: {reason}")


def check_unique_configs():
    """Check that all expected configs have eval outputs."""
    print("\n" + "=" * 60)
    print("5. CONFIGURATION COVERAGE CHECK")
    print("=" * 60)
    
    base_rf = os.path.join(PROJECT, 'results/outputs/evaluation/RF')
    files = glob.glob(os.path.join(base_rf, '**/*split2*.json'), recursive=True)
    basenames = sorted(set(os.path.basename(f) for f in files))
    
    # Parse basenames to extract conditions
    conditions = defaultdict(list)
    for bn in basenames:
        # eval_results_RF_{mode}_{condition}_{dist}_{domain}_{mode2}_split2_{seed}.json
        parts = bn.replace('eval_results_RF_', '').replace('_split2_', '|').replace('.json', '')
        if '|' in parts:
            config, seed = parts.split('|', 1)
        else:
            config = parts
            seed = '?'
        
        # Categorize by condition
        if 'smote_plain' in bn:
            conditions['smote_plain'].append(bn)
        elif 'swsmote' in bn or 'imbalv3' in bn:
            conditions['sw_smote'].append(bn)
        elif 'undersample_rus' in bn:
            conditions['undersample_rus'].append(bn)
        elif 'baseline' in bn:
            conditions['baseline'].append(bn)
        else:
            conditions['other'].append(bn)
    
    # BalancedRF
    base_brf = os.path.join(PROJECT, 'results/outputs/evaluation/BalancedRF')
    brf_files = glob.glob(os.path.join(base_brf, '**/*split2*.json'), recursive=True)
    brf_basenames = sorted(set(os.path.basename(f) for f in brf_files))
    conditions['balanced_rf'] = list(brf_basenames)
    
    print(f"\nUnique eval files by condition:")
    total = 0
    for cond in ['baseline', 'smote_plain', 'sw_smote', 'undersample_rus', 'balanced_rf', 'other']:
        count = len(conditions.get(cond, []))
        expected = {'baseline': 36, 'smote_plain': 72, 'sw_smote': 72, 'undersample_rus': 72, 'balanced_rf': 36, 'other': 0}
        exp = expected.get(cond, '?')
        status = '✓' if count >= exp else f'MISSING {exp - count}'
        print(f"  {cond}: {count}/{exp} {status}")
        total += count
    
    print(f"\nTotal unique: {total}/288")
    
    # Check for expected seeds
    print(f"\nSeed coverage check:")
    for seed in ['s42', 's123']:
        rf_seed = sum(1 for bn in basenames if seed in bn)
        brf_seed = sum(1 for bn in brf_basenames if seed in bn)
        print(f"  {seed}: RF={rf_seed}, BalancedRF={brf_seed}")
    
    # Check for expected distances
    print(f"\nDistance coverage:")
    for dist in ['knn_dtw', 'mmd', 'wasserstein']:
        rf_dist = sum(1 for bn in basenames if dist in bn)
        brf_dist = sum(1 for bn in brf_basenames if dist in bn)
        print(f"  {dist}: RF={rf_dist}, BalancedRF={brf_dist}")
    
    # Check for expected domains
    print(f"\nDomain coverage:")
    for dom in ['in_domain', 'out_domain']:
        rf_dom = sum(1 for bn in basenames if dom in bn)
        brf_dom = sum(1 for bn in brf_basenames if dom in bn)
        print(f"  {dom}: RF={rf_dom}, BalancedRF={brf_dom}")
    
    # Check for expected modes
    print(f"\nMode coverage:")
    for mode in ['source_only', 'target_only', 'mixed']:
        rf_mode = sum(1 for bn in basenames if f'_{mode}_' in bn)
        brf_mode = sum(1 for bn in brf_basenames if f'_{mode}_' in bn)
        print(f"  {mode}: RF={rf_mode}, BalancedRF={brf_mode}")


def check_csv_outputs():
    """Check CSV result files."""
    print("\n" + "=" * 60)
    print("6. CSV OUTPUT CHECK")
    print("=" * 60)
    
    csv_base = os.path.join(PROJECT, 'results/outputs')
    csvs = glob.glob(os.path.join(csv_base, '**/*split2*.csv'), recursive=True) + \
           glob.glob(os.path.join(csv_base, '**/*RF*domain*.csv'), recursive=True)
    
    if not csvs:
        print("No CSV files found for exp2")
        return
    
    for f in sorted(set(csvs))[:20]:
        sz = os.path.getsize(f)
        with open(f) as fh:
            lines = sum(1 for _ in fh)
        relpath = os.path.relpath(f, PROJECT)
        print(f"  {relpath}: {sz:,} bytes, {lines} lines")


if __name__ == '__main__':
    check_eval_jsons()
    check_eval_content_sample()
    check_model_artifacts()
    check_hpc_job_logs()
    check_unique_configs()
    check_csv_outputs()
    print("\n" + "=" * 60)
    print("HEALTH CHECK COMPLETE")
    print("=" * 60)

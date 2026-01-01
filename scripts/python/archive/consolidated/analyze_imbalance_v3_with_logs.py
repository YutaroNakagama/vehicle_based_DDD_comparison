#!/usr/bin/env python3
"""
analyze_imbalance_v3_with_logs.py
Analyze imbalance v3 results by extracting condition/ranking info from PBS logs.
"""

import os
import json
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/home/s2240011/git/ddd/vehicle_based_DDD_comparison")
PBS_LOG_DIR = PROJECT_ROOT / "scripts/hpc/log"
EVAL_BASE = PROJECT_ROOT / "results/evaluation"
OUTPUT_DIR = PROJECT_ROOT / "results/domain_analysis/imbalance_v3/analysis"

# Evaluation job IDs from the run
EVAL_JOBS = ["14600379", "14600382"]


def parse_pbs_logs():
    """Parse PBS logs to extract condition -> training_job mapping."""
    mapping = {}  # (eval_job[array_idx]) -> {condition, ranking, train_jobid}
    # Also create eval_dir_to_info mapping: train_jobid + eval_array_idx -> info
    eval_dir_mapping = {}
    
    for eval_job in EVAL_JOBS:
        for log_file in PBS_LOG_DIR.glob(f"{eval_job}*.OU"):
            # Extract array index from filename
            match = re.search(r'\[(\d+)\]', str(log_file))
            if not match:
                continue
            array_idx = int(match.group(1))
            
            condition = None
            ranking = None
            train_jobid = None
            
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        if '[INFO] Condition:' in line:
                            condition = line.split(':')[-1].strip()
                        elif '[INFO] Ranking:' in line:
                            ranking = line.split(':')[-1].strip()
                        elif '[INFO] Found Training Job:' in line:
                            train_jobid = line.split(':')[-1].strip()
                        
                        if condition and ranking and train_jobid:
                            break
            except Exception as e:
                print(f"[WARN] Error reading {log_file}: {e}")
                continue
            
            if condition and ranking and train_jobid:
                key = f"{eval_job}[{array_idx}]"
                mapping[key] = {
                    'condition': condition,
                    'ranking': ranking,
                    'train_jobid': train_jobid
                }
                # Create mapping for eval result directory
                # Eval results are stored as: results/evaluation/{model}/{train_jobid}/{train_jobid}[{array_idx}]/
                eval_dir_key = f"{train_jobid}[{array_idx}]"
                eval_dir_mapping[eval_dir_key] = {
                    'condition': condition,
                    'ranking': ranking,
                    'train_jobid': train_jobid,
                    'eval_array_idx': array_idx
                }
    
    return mapping, eval_dir_mapping


def parse_train_jobid_to_condition(mapping):
    """Invert mapping to get train_jobid -> (condition, ranking)."""
    train_to_cond = defaultdict(list)
    for key, info in mapping.items():
        train_to_cond[info['train_jobid']].append({
            'condition': info['condition'],
            'ranking': info['ranking'],
            'eval_key': key
        })
    return train_to_cond


def load_eval_results(eval_dir_mapping):
    """Load evaluation results and associate with conditions using eval_dir_mapping."""
    results = []
    
    for model_type in ['RF', 'BalancedRF']:
        eval_dir = EVAL_BASE / model_type
        if not eval_dir.exists():
            continue
        
        for job_dir in eval_dir.iterdir():
            if not job_dir.is_dir():
                continue
            
            job_id = job_dir.name
            
            for sub_dir in job_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                
                # Get the directory name which is like "14598153[19]"
                sub_dir_name = sub_dir.name
                
                # Look up in eval_dir_mapping
                if sub_dir_name in eval_dir_mapping:
                    info = eval_dir_mapping[sub_dir_name]
                    condition = info['condition']
                    ranking = info['ranking']
                else:
                    # Fallback - try to extract from sub_dir name format
                    condition = 'unknown'
                    ranking = 'unknown'
                
                # Find eval results
                for json_file in sub_dir.glob('eval_results*.json'):
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        # Determine mode from filename
                        if 'source_only' in json_file.name:
                            mode = 'source_only'
                        elif 'target_only' in json_file.name:
                            mode = 'target_only'
                        else:
                            mode = 'unknown'
                        
                        # Extract array index
                        match = re.search(r'\[(\d+)\]', sub_dir_name)
                        sub_idx = int(match.group(1)) if match else 0
                        
                        results.append({
                            'model_type': model_type,
                            'condition': condition,
                            'ranking': ranking,
                            'mode': mode,
                            'train_jobid': job_id,
                            'sub_idx': sub_idx,
                            'accuracy': data.get('accuracy'),
                            'precision': data.get('precision'),
                            'recall': data.get('recall'),
                            'f1': data.get('f1'),
                            'source': 'ranking'
                        })
                    except Exception as e:
                        print(f"[WARN] Error loading {json_file}: {e}")
    
    return results


def load_pooled_results():
    """Load pooled evaluation results."""
    pooled_dir = PROJECT_ROOT / "results/domain_analysis/imbalance_v3/evaluation"
    results = []
    
    for cond_dir in pooled_dir.iterdir():
        if not cond_dir.is_dir():
            continue
        
        condition = cond_dir.name
        pooled_path = cond_dir / "pooled"
        
        if not pooled_path.exists():
            continue
        
        for json_file in pooled_path.glob('eval_results*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                mode = 'source_only' if 'source_only' in json_file.name else 'target_only'
                
                results.append({
                    'condition': condition,
                    'mode': mode,
                    'ranking': 'pooled',
                    'accuracy': data.get('accuracy'),
                    'precision': data.get('precision'),
                    'recall': data.get('recall'),
                    'f1': data.get('f1'),
                    'source': 'pooled'
                })
            except Exception as e:
                print(f"[WARN] Error loading {json_file}: {e}")
    
    return results


def main():
    print("=" * 60)
    print("Imbalance V3 Analysis with PBS Logs")
    print("=" * 60)
    
    # Parse PBS logs for mapping
    print("\n[Step 1] Parsing PBS logs...")
    mapping, eval_dir_mapping = parse_pbs_logs()
    print(f"  Found {len(mapping)} evaluation jobs with mappings")
    print(f"  Created {len(eval_dir_mapping)} eval_dir mappings")
    
    # Create train_jobid -> condition mapping
    train_to_cond = parse_train_jobid_to_condition(mapping)
    print(f"  Found {len(train_to_cond)} training jobs")
    
    print("\n[Step 2] Training Job -> Condition Mapping:")
    for job_id, info_list in sorted(train_to_cond.items()):
        conditions = set(i['condition'] for i in info_list)
        rankings = set(i['ranking'] for i in info_list)
        print(f"  {job_id}: {list(conditions)} | Rankings: {list(rankings)}")
    
    # Load evaluation results using eval_dir_mapping
    print("\n[Step 3] Loading evaluation results...")
    ranking_results = load_eval_results(eval_dir_mapping)
    print(f"  Loaded {len(ranking_results)} ranking results")
    
    # Count known vs unknown rankings
    known = sum(1 for r in ranking_results if r['ranking'] != 'unknown')
    unknown = sum(1 for r in ranking_results if r['ranking'] == 'unknown')
    print(f"  Known ranking: {known}, Unknown: {unknown}")
    
    # Load pooled results
    print("\n[Step 4] Loading pooled results...")
    pooled_results = load_pooled_results()
    print(f"  Loaded {len(pooled_results)} pooled results")
    
    # Create summary dataframe for ranking results
    if ranking_results:
        df_ranking = pd.DataFrame(ranking_results)
        
        # Filter to only known rankings for ranking analysis
        df_known = df_ranking[df_ranking['ranking'] != 'unknown']
        
        print("\n" + "=" * 60)
        print("RANKING RESULTS SUMMARY (known rankings only)")
        print("=" * 60)
        
        if len(df_known) > 0:
            print("\n[Detailed Breakdown by Condition x Ranking]")
            summary = df_known.groupby(['condition', 'ranking', 'mode']).agg({
                'recall': ['mean', 'std'],
                'precision': 'mean',
                'f1': 'mean',
                'accuracy': 'mean'
            }).round(4)
            print(summary)
            
            # Best performing combinations - source_only
            print("\n" + "-" * 60)
            print("TOP 15 COMBINATIONS BY RECALL (source_only)")
            print("-" * 60)
            src_only = df_known[df_known['mode'] == 'source_only']
            if len(src_only) > 0:
                top_recall = src_only.groupby(['condition', 'ranking']).agg({
                    'recall': 'mean',
                    'precision': 'mean',
                    'f1': 'mean'
                }).round(4).sort_values('recall', ascending=False).head(15)
                print(top_recall)
            
            print("\n" + "-" * 60)
            print("TOP 15 COMBINATIONS BY F1 (source_only)")
            print("-" * 60)
            if len(src_only) > 0:
                top_f1 = src_only.groupby(['condition', 'ranking']).agg({
                    'recall': 'mean',
                    'precision': 'mean',
                    'f1': 'mean'
                }).round(4).sort_values('f1', ascending=False).head(15)
                print(top_f1)
        
        # Also show unknown results summary
        df_unknown = df_ranking[df_ranking['ranking'] == 'unknown']
        if len(df_unknown) > 0:
            print("\n" + "=" * 60)
            print("RESULTS WITH UNKNOWN RANKING (older eval jobs)")
            print("=" * 60)
            summary_unk = df_unknown.groupby(['condition', 'mode']).agg({
                'recall': 'mean',
                'precision': 'mean',
                'f1': 'mean'
            }).round(4)
            print(summary_unk)
    
    # Create summary for pooled results
    if pooled_results:
        df_pooled = pd.DataFrame(pooled_results)
        print("\n" + "=" * 60)
        print("POOLED RESULTS SUMMARY (sorted by Recall)")
        print("=" * 60)
        
        # Pivot for easier reading
        summary_pooled = df_pooled.groupby('condition').agg({
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'f1': 'mean'
        }).round(4).sort_values('recall', ascending=False)
        print(summary_pooled)
    
    # Save to output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save all results
    all_results = ranking_results + pooled_results
    if all_results:
        df_all = pd.DataFrame(all_results)
        output_file = OUTPUT_DIR / "all_results_summary.csv"
        df_all.to_csv(output_file, index=False)
        print(f"\n[Saved] Results to {output_file}")
    
    # Save mapping
    mapping_file = OUTPUT_DIR / "job_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"[Saved] Job mapping to {mapping_file}")
    
    # Save eval_dir_mapping
    eval_mapping_file = OUTPUT_DIR / "eval_dir_mapping.json"
    with open(eval_mapping_file, 'w') as f:
        json.dump(eval_dir_mapping, f, indent=2)
    print(f"[Saved] Eval dir mapping to {eval_mapping_file}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

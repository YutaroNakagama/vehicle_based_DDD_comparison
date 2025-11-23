#!/usr/bin/env python3
"""
Display confusion matrices in table format for all evaluation results.
"""
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

def load_confusion_matrix(json_path):
    """Load confusion matrix from evaluation JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    test_results = data.get('test', {})
    
    # Get confusion matrix values
    if 'confusion_matrix' in test_results:
        cm = test_results['confusion_matrix']
        tn, fp = cm[0]
        fn, tp = cm[1]
    else:
        tn = test_results.get('tn', 0)
        fp = test_results.get('fp', 0)
        fn = test_results.get('fn', 0)
        tp = test_results.get('tp', 0)
    
    # Get metrics
    precision = test_results.get('precision', 0)
    recall = test_results.get('recall', 0)
    f1 = test_results.get('f1', 0)
    
    return tn, fp, fn, tp, precision, recall, f1

def parse_filename(filename):
    """Parse evaluation result filename to extract metadata."""
    mode = None
    distance = None
    level = None
    
    if 'pooled' in filename:
        mode = 'pooled'
    elif 'source_only' in filename:
        mode = 'source_only'
    elif 'target_only' in filename:
        mode = 'target_only'
    
    if 'dtw' in filename:
        distance = 'dtw'
    elif 'mmd' in filename:
        distance = 'mmd'
    elif 'wasserstein' in filename:
        distance = 'wasserstein'
    
    if 'high' in filename:
        level = 'high'
    elif 'middle' in filename:
        level = 'middle'
    elif 'low' in filename:
        level = 'low'
    
    return mode, distance, level

def main():
    eval_dir = Path('results/evaluation/RF/14357179')
    
    if not eval_dir.exists():
        print(f"Error: {eval_dir} does not exist")
        return
    
    # Collect all evaluation files
    eval_files = list(eval_dir.rglob('eval_results_*.json'))
    print(f"Found {len(eval_files)} evaluation files\n")
    
    # Organize data
    data_dict = defaultdict(lambda: defaultdict(dict))
    
    for eval_file in eval_files:
        mode, distance, level = parse_filename(eval_file.name)
        
        if mode and distance and level:
            tn, fp, fn, tp, precision, recall, f1 = load_confusion_matrix(eval_file)
            data_dict[distance][level][mode] = {
                'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
                'Precision': precision, 'Recall': recall, 'F1': f1
            }
    
    # Display tables
    distances = ['dtw', 'mmd', 'wasserstein']
    levels = ['high', 'middle', 'low']
    modes = ['pooled', 'source_only', 'target_only']
    
    for distance in distances:
        if distance not in data_dict:
            continue
        
        print("=" * 100)
        print(f"Distance Metric: {distance.upper()}")
        print("=" * 100)
        
        for level in levels:
            if level not in data_dict[distance]:
                continue
            
            print(f"\n--- Level: {level.upper()} ---")
            
            rows = []
            for mode in modes:
                if mode in data_dict[distance][level]:
                    d = data_dict[distance][level][mode]
                    row = {
                        'Mode': mode,
                        'TN': d['TN'],
                        'FP': d['FP'],
                        'FN': d['FN'],
                        'TP': d['TP'],
                        'Precision': f"{d['Precision']:.4f}",
                        'Recall': f"{d['Recall']:.4f}",
                        'F1': f"{d['F1']:.4f}"
                    }
                    rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                print(df.to_string(index=False))
                
                # Calculate total samples
                total = rows[0]['TN'] + rows[0]['FP'] + rows[0]['FN'] + rows[0]['TP']
                drowsy = rows[0]['FN'] + rows[0]['TP']
                non_drowsy = rows[0]['TN'] + rows[0]['FP']
                print(f"\nTotal samples: {total} (Non-drowsy: {non_drowsy}, Drowsy: {drowsy})")
        
        print()
    
    # Summary table across all cases
    print("\n" + "=" * 120)
    print("SUMMARY: All Cases")
    print("=" * 120)
    
    summary_rows = []
    for distance in distances:
        if distance not in data_dict:
            continue
        for level in levels:
            if level not in data_dict[distance]:
                continue
            for mode in modes:
                if mode in data_dict[distance][level]:
                    d = data_dict[distance][level][mode]
                    summary_rows.append({
                        'Distance': distance,
                        'Level': level,
                        'Mode': mode,
                        'TN': d['TN'],
                        'FP': d['FP'],
                        'FN': d['FN'],
                        'TP': d['TP'],
                        'Precision': f"{d['Precision']:.4f}",
                        'Recall': f"{d['Recall']:.4f}",
                        'F1': f"{d['F1']:.4f}"
                    })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))
        
        # Save to CSV
        output_dir = Path('results/domain_analysis/summary/csv')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'confusion_matrices_all_cases.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved to: {output_file}")

if __name__ == '__main__':
    main()

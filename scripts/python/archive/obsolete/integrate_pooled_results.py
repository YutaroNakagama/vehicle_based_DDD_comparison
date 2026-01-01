#!/usr/bin/env python3
"""
integrate_pooled_results.py
===========================
Integrate existing pooled evaluation results into imbalance_v3 domain analysis.

Existing pooled results from ratio experiments can be directly used since
pooled mode doesn't depend on domain ranking.
"""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path("/home/s2240011/git/ddd/vehicle_based_DDD_comparison")

# Mapping: (method, ratio) -> (model_type, job_id, subdirectory)
POOLED_MAPPING = {
    ("smote", "0.1"): ("RF", "14593005", "14593005[1]"),
    ("smote", "0.5"): ("RF", "14593030", "14593030[1]"),
    ("smote", "1.0"): ("RF", "14593052", "14593052[1]"),
    ("smote_tomek", "0.1"): ("RF", "14592998", "14592998[1]"),
    ("smote_tomek", "0.5"): ("RF", "14593021", "14593021[1]"),
    ("smote_tomek", "1.0"): ("RF", "14593046", "14593046[1]"),
    ("smote_balanced_rf", "0.1"): ("BalancedRF", "14592992", "14592992[1]"),
    ("smote_balanced_rf", "0.5"): ("BalancedRF", "14584040", "14584040[1]"),
    ("smote_balanced_rf", "1.0"): ("BalancedRF", "14584066", "14584066[1]"),
    ("undersample_rus", "0.1"): ("RF", "14592994", "14592994[1]"),
    ("undersample_rus", "0.5"): ("RF", "14593017", "14593017[1]"),
    ("undersample_rus", "1.0"): ("RF", "14593042", "14593042[1]"),
    ("baseline", "none"): ("RF", "14593038", "14593038[1]"),
}

def get_method_suffix(method: str, ratio: str) -> str:
    """Generate method suffix for directory naming."""
    if ratio == "none":
        return method
    return f"{method}_{ratio}"

def find_eval_result(model_type: str, job_id: str, subdir: str) -> Path | None:
    """Find evaluation result JSON file."""
    eval_dir = PROJECT_ROOT / "results" / "evaluation" / model_type / job_id / subdir
    if eval_dir.exists():
        json_files = list(eval_dir.glob("eval_results_*.json"))
        if json_files:
            return json_files[0]
    return None

def integrate_pooled_results():
    """Integrate existing pooled results into imbalance_v3 structure."""
    output_base = PROJECT_ROOT / "results" / "domain_analysis" / "imbalance_v3" / "evaluation"
    output_base.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for (method, ratio), (model_type, job_id, subdir) in POOLED_MAPPING.items():
        method_suffix = get_method_suffix(method, ratio)
        output_dir = output_base / method_suffix / "pooled"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find existing evaluation result
        src_file = find_eval_result(model_type, job_id, subdir)
        
        if src_file:
            # Copy to new location with standardized name
            dst_file = output_dir / f"eval_results_{model_type}_pooled_{method_suffix}.json"
            shutil.copy2(src_file, dst_file)
            
            # Load and store result
            with open(src_file, 'r') as f:
                data = json.load(f)
            
            results.append({
                "method": method,
                "ratio": ratio,
                "model_type": model_type,
                "source_job": job_id,
                "source_file": str(src_file),
                "dest_file": str(dst_file),
                "recall": data.get("recall", "N/A"),
                "precision": data.get("precision", "N/A"),
                "f1": data.get("f1", "N/A"),
                "f2": data.get("f2", "N/A"),
                "auc_pr": data.get("auc_pr", "N/A"),
            })
            print(f"✅ {method_suffix}/pooled: Copied from {job_id}")
        else:
            print(f"❌ {method_suffix}/pooled: Source not found ({model_type}/{job_id})")
    
    # Save summary
    summary_file = output_base / "pooled_integration_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Integrated {len(results)} pooled results")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}")
    
    # Print results table
    print("\nPooled Results Summary:")
    print(f"{'Method':<25} {'Ratio':<8} {'Recall':<10} {'F2':<10} {'AUPRC':<10}")
    print("-" * 70)
    for r in results:
        method_suffix = get_method_suffix(r["method"], r["ratio"])
        recall = f"{r['recall']:.4f}" if isinstance(r['recall'], float) else r['recall']
        f2 = f"{r['f2']:.4f}" if isinstance(r['f2'], float) else r['f2']
        auprc = f"{r['auc_pr']:.4f}" if isinstance(r['auc_pr'], float) else r['auc_pr']
        print(f"{method_suffix:<25} {r['ratio']:<8} {recall:<10} {f2:<10} {auprc:<10}")

if __name__ == "__main__":
    integrate_pooled_results()

#!/usr/bin/env python3
"""
Batch evaluation script for KNN + Imbalance experiments.

Evaluates all trained models from a job array and generates a summary table.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


LEVELS = ["out_domain", "mid_domain", "in_domain"]
METHODS = ["baseline", "undersample_rus", "undersample_tomek", "smote_rus", "smote_tomek"]


def evaluate_single_job(
    model: str,
    mode: str,
    distance: str,
    ranking_method: str,
    level: str,
    method: str,
    job_id: str,
    job_idx: int,
) -> dict:
    """Evaluate a single job and return metrics."""
    
    group_file = (
        PROJECT_ROOT / "results" / "domain_analysis" / "distance" / "subject-wise" /
        distance / "groups" / "clustering_ranked" / f"{distance}_{ranking_method}_{level}.txt"
    )
    
    tag = f"rank_{distance}_{ranking_method}_{level}_{method}"
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "python" / "evaluate.py"),
        "--model", model,
        "--mode", mode,
        "--target_file", str(group_file),
        "--tag", tag,
        "--jobid", f"{job_id}[{job_idx}]",
        "--subject_wise_split",
        "--seed", "42",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        
        # Find the output JSON file
        eval_dir = PROJECT_ROOT / "results" / "evaluation" / model / job_id / f"{job_id}[{job_idx}]"
        json_files = list(eval_dir.glob("eval_results_*.json"))
        
        if json_files:
            with open(json_files[0], "r") as f:
                data = json.load(f)
            
            return {
                "level": level,
                "method": method,
                # Metrics matching summary_metrics_bar.png format
                "accuracy": data.get("acc_thr", None),  # Use threshold-based accuracy
                "recall": data.get("recall_thr", None),  # Use threshold-based recall
                "precision": data.get("prec_thr", None),  # Use threshold-based precision
                "f1": data.get("f1_thr", None),  # Use threshold-based F1
                "f2": data.get("f2_thr", None),  # Use threshold-based F2
                "auc": data.get("roc_auc", None),  # AUROC
                "auc_pr": data.get("auc_pr", None),  # AUPRC
                # Keep original names for backward compatibility
                "acc_thr": data.get("acc_thr", None),
                "recall_thr": data.get("recall_thr", None),
                "prec_thr": data.get("prec_thr", None),
                "f1_thr": data.get("f1_thr", None),
                "f2_thr": data.get("f2_thr", None),
                "specificity_thr": data.get("specificity_thr", None),
                "roc_auc": data.get("roc_auc", None),
                "auprc": data.get("auc_pr", None),
                "pos_rate": data.get("pred_pos_rate_at_0p5", 0.033),
                "status": "success",
            }
        else:
            return {
                "level": level,
                "method": method,
                "status": "no_output",
                "error": result.stderr[-500:] if result.stderr else "Unknown error",
            }
            
    except subprocess.TimeoutExpired:
        return {
            "level": level,
            "method": method,
            "status": "timeout",
        }
    except Exception as e:
        return {
            "level": level,
            "method": method,
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate KNN + Imbalance experiments")
    parser.add_argument("--job_id", type=str, required=True, help="Job array ID (e.g., 14471570)")
    parser.add_argument("--model", type=str, default="RF", help="Model name")
    parser.add_argument("--mode", type=str, default="source_only", help="Training mode")
    parser.add_argument("--distance", type=str, default="mmd", help="Distance metric")
    parser.add_argument("--ranking_method", type=str, default="knn", help="Ranking method")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
    args = parser.parse_args()
    
    results = []
    
    # Evaluate all 15 jobs (3 levels x 5 methods)
    for level_idx, level in enumerate(LEVELS):
        for method_idx, method in enumerate(METHODS):
            job_idx = level_idx * len(METHODS) + method_idx + 1
            
            print(f"Evaluating Job[{job_idx}]: {level} + {method}...", flush=True)
            
            result = evaluate_single_job(
                model=args.model,
                mode=args.mode,
                distance=args.distance,
                ranking_method=args.ranking_method,
                level=level,
                method=method,
                job_id=args.job_id,
                job_idx=job_idx,
            )
            results.append(result)
            
            if result.get("status") == "success":
                auc = result.get('auc')
                auc_pr = result.get('auc_pr')
                f1 = result.get('f1')
                f2 = result.get('f2')
                recall = result.get('recall')
                auc_str = f"{auc:.4f}" if auc is not None else "N/A"
                auc_pr_str = f"{auc_pr:.4f}" if auc_pr is not None else "N/A"
                f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
                f2_str = f"{f2:.4f}" if f2 is not None else "N/A"
                recall_str = f"{recall:.4f}" if recall is not None else "N/A"
                print(f"  ✓ AUC: {auc_str}, AUPRC: {auc_pr_str}, F1: {f1_str}, F2: {f2_str}, Recall: {recall_str}")
            else:
                print(f"  ✗ {result.get('status')}: {result.get('error', '')[:100]}")
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = (
            PROJECT_ROOT / "results" / "domain_analysis" / "knn_imbalance" /
            f"summary_{args.job_id}_{args.mode}_{args.distance}.csv"
        )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved summary to: {output_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    
    success_df = df[df["status"] == "success"]
    if not success_df.empty:
        pivot_f2 = success_df.pivot(index="level", columns="method", values="f2")
        pivot_f1 = success_df.pivot(index="level", columns="method", values="f1")
        pivot_recall = success_df.pivot(index="level", columns="method", values="recall")
        pivot_auc = success_df.pivot(index="level", columns="method", values="auc")
        pivot_auc_pr = success_df.pivot(index="level", columns="method", values="auc_pr")
        
        print("\nF2 Score:")
        print(pivot_f2.to_string())
        
        print("\nF1 Score:")
        print(pivot_f1.to_string())
        
        print("\nRecall:")
        print(pivot_recall.to_string())
        
        print("\nAUROC:")
        print(pivot_auc.to_string())
        
        print("\nAUPRC:")
        print(pivot_auc_pr.to_string())
    else:
        print("No successful evaluations.")
    
    return df


if __name__ == "__main__":
    main()

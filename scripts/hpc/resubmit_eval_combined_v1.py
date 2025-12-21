#!/usr/bin/env python3
"""
Combined V1: 評価ジョブ再投入スクリプト
訓練完了後のモデルに対して評価ジョブを投入する

問題: 元の評価スクリプトに --mode と --model が欠けていた
修正: このスクリプトで正しい評価ジョブを再投入
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "combined_v1"

# 訓練完了ログからタグを抽出
def get_completed_training_tags():
    """訓練完了したジョブのタグを取得"""
    completed = []
    
    log_files = list(LOG_DIR.glob("*.OU"))
    
    for log_file in log_files:
        try:
            content = log_file.read_text()
            if "Training completed" in content:
                # タグを抽出
                for line in content.split('\n'):
                    if line.startswith("Tag:"):
                        tag = line.split("Tag:")[1].strip()
                        # モードとモデルも抽出
                        mode = None
                        model = None
                        for l in content.split('\n'):
                            if l.startswith("Mode:"):
                                mode = l.split("Mode:")[1].strip()
                            if l.startswith("Model:"):
                                model = l.split("Model:")[1].strip()
                        if tag and mode and model:
                            completed.append({
                                'tag': tag,
                                'mode': mode,
                                'model': model,
                                'log_file': log_file.name,
                            })
                        break
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    return completed


def create_eval_script(tag, mode, model):
    """評価ジョブスクリプト作成"""
    job_name = f"ev_{tag[:12]}"
    
    script = f'''#!/bin/bash
#PBS -N {job_name}
#PBS -q TINY
#PBS -l select=1:ncpus=2:mem=4gb
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

echo "=========================================="
echo "Evaluation (Re-submitted)"
echo "Tag: {tag}"
echo "Mode: {mode}"
echo "Model: {model}"
echo "=========================================="

python scripts/python/evaluate.py \\
    --tag {tag} \\
    --mode {mode} \\
    --model {model}

echo "Evaluation completed at $(date)"
'''
    return script


def submit_eval_job(script_content):
    """評価ジョブを投入"""
    result = subprocess.run(
        ['qsub'],
        input=script_content,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print(f"Error: {result.stderr}")
        return None


def main():
    print("=" * 60)
    print("Combined V1: 評価ジョブ再投入")
    print("=" * 60)
    
    # 訓練完了したジョブを取得
    completed = get_completed_training_tags()
    print(f"訓練完了ジョブ: {len(completed)}件")
    
    if not completed:
        print("訓練完了ジョブがありません。")
        return
    
    # 評価ジョブを投入
    submitted = 0
    for job in completed:
        tag = job['tag']
        mode = job['mode']
        model = job['model']
        
        script = create_eval_script(tag, mode, model)
        job_id = submit_eval_job(script)
        
        if job_id:
            print(f"  ✓ {tag[:40]}... -> {job_id}")
            submitted += 1
        else:
            print(f"  ✗ {tag[:40]}... FAILED")
    
    print(f"\n[完了] {submitted}件の評価ジョブを投入しました")


if __name__ == "__main__":
    main()

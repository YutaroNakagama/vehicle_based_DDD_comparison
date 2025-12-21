#!/usr/bin/env python3
"""
Imbal V2 Fixed: LONGキューからSEMINARキューへ再投入
LONGキューはユーザー最大2ジョブ制限のため、SEMINARキューに再投入して並列化

対象: undersample系 (undersample_tomek, undersample_enn) 実験
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "imbal_v2_fixed"
LOG_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS = 75

# 実験リスト (experiment_list.txtから抽出)
EXPERIMENTS = [
    # undersample_tomek
    ("undersample_tomek", 0.5, 123, "RF", "undersample_tomek"),
    ("undersample_tomek", 0.5, 456, "RF", "undersample_tomek"),
    ("undersample_tomek", 1.0, 42, "RF", "undersample_tomek"),
    ("undersample_tomek", 1.0, 123, "RF", "undersample_tomek"),
    ("undersample_tomek", 1.0, 456, "RF", "undersample_tomek"),
    # undersample_enn
    ("undersample_enn", 0.1, 42, "RF", "undersample_enn"),
    ("undersample_enn", 0.1, 123, "RF", "undersample_enn"),
    ("undersample_enn", 0.1, 456, "RF", "undersample_enn"),
    ("undersample_enn", 0.5, 42, "RF", "undersample_enn"),
    ("undersample_enn", 0.5, 123, "RF", "undersample_enn"),
    ("undersample_enn", 0.5, 456, "RF", "undersample_enn"),
    ("undersample_enn", 1.0, 42, "RF", "undersample_enn"),
    ("undersample_enn", 1.0, 123, "RF", "undersample_enn"),
    ("undersample_enn", 1.0, 456, "RF", "undersample_enn"),
]


def create_train_script(method, ratio, seed, model, oversample_method):
    """訓練ジョブスクリプト作成"""
    tag = f"imbalv2_{method}_ratio{ratio}_seed{seed}"
    ratio_str = str(ratio).replace('.', '_')
    job_name = f"iv2_{method[:4]}_r{ratio_str}_s{seed}"[:15]
    
    script = f'''#!/bin/bash
#PBS -N {job_name}
#PBS -q SEMINAR
#PBS -l select=1:ncpus=4:mem=4gb
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o {LOG_DIR}/

set -euo pipefail

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$PROJECT_ROOT"

export PATH=~/conda/bin:$PATH
source ~/conda/etc/profile.d/conda.sh
conda activate python310
export PYTHONPATH="${{PROJECT_ROOT}}:${{PYTHONPATH:-}}"
export N_TRIALS_OVERRIDE={N_TRIALS}

echo "=========================================="
echo "Imbalance V2 Fixed (SEMINAR queue)"
echo "Method: {method}"
echo "Ratio: {ratio}"
echo "Seed: {seed}"
echo "Model: {model}"
echo "=========================================="

python scripts/python/train.py \\
    --model {model} \\
    --mode pooled \\
    --tag {tag} \\
    --seed {seed} \\
    --time_stratify_labels \\
    --use_oversampling \\
    --oversample_method {oversample_method} \\
    --target_ratio {ratio}

echo "Training completed at $(date)"
'''
    return script, job_name, tag


def create_eval_script(tag, job_name, model="RF"):
    """評価ジョブスクリプト作成"""
    script = f'''#!/bin/bash
#PBS -N ev_{job_name[:12]}
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
echo "Evaluation: {tag}"
echo "Mode: pooled"
echo "Model: {model}"
echo "=========================================="

python scripts/python/evaluate.py \\
    --tag "{tag}" \\
    --mode pooled \\
    --model {model}

echo "Evaluation completed at $(date)"
'''
    return script


def submit_job(script_content, job_name):
    """PBS ジョブ投入"""
    result = subprocess.run(
        ['qsub'],
        input=script_content,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        job_id = result.stdout.strip()
        return job_id
    else:
        print(f"[ERROR] Failed to submit {job_name}: {result.stderr}")
        return None


def main():
    print("=" * 60)
    print("Imbal V2 Fixed: LONGキュー → SEMINARキュー再投入")
    print(f"対象実験数: {len(EXPERIMENTS)}")
    print("=" * 60)
    
    # 既存のLONGキュージョブをキャンセル
    print("\n[STEP 1] 既存のLONGキュージョブをキャンセル...")
    result = subprocess.run(
        "qstat -u $USER 2>/dev/null | grep 'iv2_' | awk '{print $1}'",
        shell=True, capture_output=True, text=True
    )
    job_ids = result.stdout.strip().split('\n')
    for job_id in job_ids:
        if job_id:
            subprocess.run(['qdel', job_id], capture_output=True)
            print(f"  Cancelled: {job_id}")
    
    # 評価ジョブもキャンセル
    result = subprocess.run(
        "qstat -u $USER 2>/dev/null | grep 'ev_iv2' | awk '{print $1}'",
        shell=True, capture_output=True, text=True
    )
    eval_job_ids = result.stdout.strip().split('\n')
    for job_id in eval_job_ids:
        if job_id:
            subprocess.run(['qdel', job_id], capture_output=True)
            print(f"  Cancelled eval: {job_id}")
    
    print("\n[STEP 2] SEMINARキューに再投入...")
    submitted = []
    
    for method, ratio, seed, model, oversample_method in EXPERIMENTS:
        # 訓練ジョブ
        train_script, job_name, tag = create_train_script(
            method, ratio, seed, model, oversample_method
        )
        train_id = submit_job(train_script, job_name)
        
        if train_id:
            # 評価ジョブ（訓練完了後に実行）
            eval_script = create_eval_script(tag, job_name)
            eval_result = subprocess.run(
                ['qsub', '-W', f'depend=afterok:{train_id}'],
                input=eval_script,
                capture_output=True,
                text=True
            )
            eval_id = eval_result.stdout.strip() if eval_result.returncode == 0 else None
            
            print(f"  ✓ {job_name}: train={train_id}, eval={eval_id}")
            submitted.append((tag, train_id, eval_id))
    
    print(f"\n[完了] {len(submitted)} 実験を SEMINAR キューに投入しました")
    print("推定完了時間: ~2-3時間")


if __name__ == "__main__":
    main()

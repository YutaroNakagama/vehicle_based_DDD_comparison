#!/usr/bin/env python3
"""
Combined V1: モデルは存在するが評価が未完了のジョブに対して評価を投入
"""

import subprocess
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

def get_tag_info_from_model(model_dir: Path) -> dict:
    """モデルディレクトリからタグ情報を抽出"""
    # モデルファイルを探す (サブディレクトリ含む)
    for pkl_file in model_dir.rglob("*.pkl"):
        name = pkl_file.stem
        # scaler_RF_target_only_combined_v1_... or RF_target_only_combined_v1_...
        # パターン: (scaler_|selected_features_)?{MODEL}_{MODE}_{TAG}_{JOBID}_{TRIAL}
        
        if "RF_" in name and "combined_v1" in name:
            # RF_target_only_combined_v1_smote_ratio0_1_lof_mmd_mid_domain_target_only_s42_14621245_1.pkl
            # 最初の RF_ または BalancedRF_ を見つける
            rf_start = name.find("BalancedRF_")
            if rf_start == -1:
                rf_start = name.find("RF_")
            if rf_start == -1:
                continue
            
            # RF_以降を取得
            name_from_rf = name[rf_start:]
            parts = name_from_rf.split("_")
            
            if len(parts) >= 3:
                model_type = parts[0]  # RF or BalancedRF
                mode = parts[1] + "_" + parts[2]  # source_only or target_only
                
                # タグは combined_v1 から始まる
                tag_start = name_from_rf.find("combined_v1")
                if tag_start != -1:
                    # タグはジョブIDの前まで (_14621XXX_の手前まで)
                    # _s42_ で終わるのを探す
                    s42_pos = name_from_rf.find("_s42")
                    if s42_pos != -1:
                        tag = name_from_rf[tag_start:s42_pos + 4]
                        return {"tag": tag, "mode": mode, "model": model_type}
    return None


def submit_eval_job(jobid: str, tag: str, mode: str, model: str, queue: str) -> str:
    """評価ジョブを投入"""
    pbs_content = f'''#!/bin/bash
#PBS -N ev_{jobid[-4:]}
#PBS -q {queue}
#PBS -l select=1:ncpus=2:mem=4gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o $HOME/git/ddd/vehicle_based_DDD_comparison/logs/combined_v1/

cd $HOME/git/ddd/vehicle_based_DDD_comparison
source ~/.bashrc
conda activate ddd3.9

echo "[EVAL] Start: $(date)"
echo "Tag: {tag}"
echo "Mode: {mode}"
echo "Model: {model}"
echo "JobID: {jobid}"

python bin/evaluate.py --tag {tag} --mode {mode} --model {model} --jobid {jobid}

echo "[EVAL] End: $(date)"
'''
    
    result = subprocess.run(
        ['qsub'],
        input=pbs_content,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        return None


def main():
    print("=" * 60)
    print("Combined V1: 未評価モデルへの評価ジョブ投入")
    print("=" * 60)
    
    # キュー割り当て
    queues = ["TINY", "DEFAULT", "SINGLE", "SMALL"]
    queue_idx = 0
    
    submitted = 0
    skipped = 0
    failed = 0
    
    # 未評価リストを読み込む
    missing_file = Path("/tmp/missing_eval.txt")
    if not missing_file.exists():
        print("Error: /tmp/missing_eval.txt not found")
        return
    
    job_ids = missing_file.read_text().strip().split('\n')
    print(f"未評価モデル数: {len(job_ids)}")
    
    for jobid in job_ids:
        # モデルディレクトリを探す
        rf_dir = PROJECT_ROOT / "models" / "RF" / jobid
        bf_dir = PROJECT_ROOT / "models" / "BalancedRF" / jobid
        
        if rf_dir.exists():
            model_dir = rf_dir
            model_type = "RF"
        elif bf_dir.exists():
            model_dir = bf_dir
            model_type = "BalancedRF"
        else:
            print(f"  Skip {jobid}: model directory not found")
            skipped += 1
            continue
        
        # すでに評価が完了しているか確認
        eval_rf = PROJECT_ROOT / "results" / "evaluation" / "RF" / jobid
        eval_bf = PROJECT_ROOT / "results" / "evaluation" / "BalancedRF" / jobid
        if eval_rf.exists() or eval_bf.exists():
            print(f"  Skip {jobid}: already evaluated")
            skipped += 1
            continue
        
        # タグ情報を抽出
        info = get_tag_info_from_model(model_dir)
        if not info:
            print(f"  Skip {jobid}: cannot extract tag info")
            skipped += 1
            continue
        
        tag = info["tag"]
        mode = info["mode"]
        model = info["model"]
        
        # 評価ジョブを投入
        queue = queues[queue_idx % len(queues)]
        queue_idx += 1
        
        eval_jobid = submit_eval_job(jobid, tag, mode, model, queue)
        if eval_jobid:
            print(f"  ✓ {jobid} ({mode}, {model}) -> {eval_jobid} ({queue})")
            submitted += 1
        else:
            print(f"  ✗ {jobid}: submission failed")
            failed += 1
    
    print()
    print(f"[完了] {submitted}件投入, {skipped}件スキップ, {failed}件失敗")


if __name__ == "__main__":
    main()

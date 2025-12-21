#!/usr/bin/env python3
"""
Imbal V2 Fixed: 評価ジョブ再投入 (V2)
訓練完了済みジョブの評価を再投入（--mode pooled --model RF --jobid 追加）
"""

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "imbal_v2_fixed"


def get_completed_training():
    """訓練完了したジョブのタグ、モデル、ジョブIDを取得"""
    completed = []
    
    for log_file in LOG_DIR.glob("*.OU"):
        # ログファイル名からジョブIDを抽出 (14621641.spcc-adm1.OU)
        job_id_match = re.match(r'^(\d+)', log_file.name)
        if not job_id_match:
            continue
        job_id = job_id_match.group(1)
        
        try:
            content = log_file.read_text()
            
            # 訓練完了を確認
            if "Training completed" not in content and "[DONE]" not in content:
                continue
            
            # 評価ジョブのログはスキップ (Evaluation の文字列が含まれる)
            if "Evaluation (Re-submitted)" in content or "[EVAL] Start" in content:
                continue
            
            # タグを [START] 行から正確に抽出
            # [START] model=RF | mode=pooled | tag=imbalv2_undersample_tomek_ratio0.5_seed123
            tag = None
            model = "RF"
            
            for line in content.split('\n'):
                # [START] 行を探す
                if "[START]" in line and "tag=" in line:
                    # tag=... | suffix=... または tag=...(行末)
                    tag_match = re.search(r'tag=([^\s\|]+)', line)
                    if tag_match:
                        tag = tag_match.group(1)
                    
                    # モデル抽出
                    if "model=BalancedRF" in line:
                        model = "BalancedRF"
                    elif "model=EasyEnsemble" in line:
                        model = "EasyEnsemble"
                    else:
                        model = "RF"
                    break
            
            if tag:
                completed.append({
                    'tag': tag,
                    'model': model,
                    'jobid': job_id,
                    'log_file': log_file.name,
                })
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    return completed


def create_eval_script(tag, model, jobid):
    """評価ジョブスクリプト作成（jobid付き）"""
    job_name = f"ev_iv2_{tag[:8]}"
    
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
echo "Evaluation (Re-submitted V2)"
echo "Tag: {tag}"
echo "Mode: pooled"
echo "Model: {model}"
echo "JobID: {jobid}"
echo "=========================================="

python scripts/python/evaluate.py \
    --tag "{tag}" \
    --mode pooled \
    --model {model} \
    --jobid {jobid}

echo "Evaluation completed at $(date)"
'''
    return script


def submit_job(script_content):
    """ジョブを投入"""
    result = subprocess.run(
        ['qsub'],
        input=script_content,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def main():
    print("=" * 60)
    print("Imbal V2 Fixed: 評価ジョブ再投入 (jobid修正版 V2)")
    print("=" * 60)
    
    # 訓練完了ジョブを取得
    completed = get_completed_training()
    print(f"訓練完了ジョブ: {len(completed)}件")
    
    if not completed:
        print("訓練完了ジョブがありません。")
        return
    
    # 抽出した情報を表示
    print("\n[DEBUG] 抽出した訓練ジョブ情報:")
    for job in completed:
        print(f"  - tag={job['tag']}, model={job['model']}, jobid={job['jobid']}")
    
    # 失敗した評価ジョブをキャンセル
    print("\n[STEP 1] 失敗した評価ジョブをキャンセル...")
    subprocess.run(
        "qstat -u $USER 2>/dev/null | grep 'ev_iv2' | awk '{print $1}' | xargs -I {} qdel {} 2>/dev/null",
        shell=True, capture_output=True
    )
    
    # 評価ジョブを投入
    print("\n[STEP 2] 評価ジョブを投入...")
    submitted = 0
    for job in completed:
        tag = job['tag']
        model = job['model']
        jobid = job['jobid']
        
        script = create_eval_script(tag, model, jobid)
        new_job_id = submit_job(script)
        
        if new_job_id:
            print(f"  ✓ tag={tag}, jobid={jobid} ({model}) -> {new_job_id}")
            submitted += 1
        else:
            print(f"  ✗ tag={tag}, jobid={jobid} FAILED")
    
    print(f"\n[完了] {submitted}件の評価ジョブを投入しました")


if __name__ == "__main__":
    main()

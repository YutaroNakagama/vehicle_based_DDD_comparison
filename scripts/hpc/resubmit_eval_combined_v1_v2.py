#!/usr/bin/env python3
"""
Combined V1: 評価ジョブ再投入スクリプト V2
訓練完了後のモデルに対して評価ジョブを投入する（jobid修正版）

修正: --jobid を追加して正確なモデルを参照
"""

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "combined_v1"


def get_completed_training_tags():
    """訓練完了したジョブのタグ、モード、モデル、ジョブIDを取得"""
    completed = []
    
    for log_file in LOG_DIR.glob("*.OU"):
        # ログファイル名からジョブIDを抽出
        job_id_match = re.match(r'^(\d+)', log_file.name)
        if not job_id_match:
            continue
        job_id = job_id_match.group(1)
        
        try:
            content = log_file.read_text()
            
            # 訓練完了を確認
            if "Training completed" not in content:
                continue
            
            # 評価ジョブのログはスキップ
            if "Evaluation (Re-submitted)" in content or "[EVAL] Start" in content:
                continue
            
            # タグ、モード、モデルを抽出
            tag = None
            mode = None
            model = None
            
            for line in content.split('\n'):
                if line.startswith("Tag:"):
                    tag = line.split("Tag:")[1].strip()
                elif line.startswith("Mode:"):
                    mode = line.split("Mode:")[1].strip()
                elif line.startswith("Model:"):
                    model = line.split("Model:")[1].strip()
            
            if tag and mode and model:
                completed.append({
                    'tag': tag,
                    'mode': mode,
                    'model': model,
                    'jobid': job_id,
                    'log_file': log_file.name,
                })
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    return completed


def create_eval_script(tag, mode, model, jobid):
    """評価ジョブスクリプト作成（jobid付き）"""
    job_name = f"ev_c1_{tag[:8]}"
    
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
echo "Mode: {mode}"
echo "Model: {model}"
echo "JobID: {jobid}"
echo "=========================================="

python scripts/python/evaluate.py \
    --tag "{tag}" \
    --mode {mode} \
    --model {model} \
    --jobid {jobid}

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
    return None


def get_already_evaluated():
    """既に評価完了したジョブIDを取得"""
    evaluated = set()
    for log_file in LOG_DIR.glob("*.OU"):
        try:
            content = log_file.read_text()
            if "[EVAL DONE]" in content:
                # JobID: を探す
                for line in content.split('\n'):
                    if line.startswith("JobID:"):
                        jobid = line.split("JobID:")[1].strip()
                        evaluated.add(jobid)
                        break
        except:
            pass
    return evaluated


def main():
    print("=" * 60)
    print("Combined V1: 評価ジョブ再投入 (jobid修正版 V2)")
    print("=" * 60)
    
    # 訓練完了したジョブを取得
    completed = get_completed_training_tags()
    print(f"訓練完了ジョブ: {len(completed)}件")
    
    if not completed:
        print("訓練完了ジョブがありません。")
        return
    
    # デバッグ情報表示
    print("\n[DEBUG] 抽出した訓練ジョブ情報:")
    for job in completed[:5]:  # 最初の5件のみ表示
        print(f"  - tag={job['tag'][:40]}..., mode={job['mode']}, model={job['model']}, jobid={job['jobid']}")
    if len(completed) > 5:
        print(f"  ... 他 {len(completed) - 5}件")
    
    # 既存の評価完了を確認
    evaluated = get_already_evaluated()
    print(f"\n既に評価完了: {len(evaluated)}件")
    
    # 失敗した評価ジョブをキャンセル
    print("\n[STEP 1] キュー中の評価ジョブをキャンセル...")
    subprocess.run(
        "qstat -u $USER 2>/dev/null | grep 'ev_c1\\|ev_com' | awk '{print $1}' | xargs -I {} qdel {} 2>/dev/null",
        shell=True, capture_output=True
    )
    
    # 評価ジョブを投入
    print("\n[STEP 2] 評価ジョブを投入...")
    submitted = 0
    skipped = 0
    
    for job in completed:
        tag = job['tag']
        mode = job['mode']
        model = job['model']
        jobid = job['jobid']
        
        # 既に評価完了しているならスキップ
        if jobid in evaluated:
            skipped += 1
            continue
        
        script = create_eval_script(tag, mode, model, jobid)
        new_job_id = submit_eval_job(script)
        
        if new_job_id:
            print(f"  ✓ jobid={jobid}, mode={mode}, model={model} -> {new_job_id}")
            submitted += 1
        else:
            print(f"  ✗ jobid={jobid} FAILED")
    
    print(f"\n[完了] {submitted}件投入, {skipped}件スキップ（既に評価完了）")


if __name__ == "__main__":
    main()

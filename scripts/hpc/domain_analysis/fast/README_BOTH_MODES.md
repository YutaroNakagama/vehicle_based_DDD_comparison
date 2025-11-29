# Both-Modes Pipeline Management

## 概要

source_only と target_only の両モードで全ランキング手法を実行するための完全自動化パイプライン。

## ファイル構成

```
scripts/hpc/domain_analysis/fast/
├── pbs_train_rank_fast_both_modes.sh      # トレーニングジョブ（18タスク = 9ケース × 2モード）
├── pbs_eval_rank_fast_both_modes.sh       # 評価ジョブ（18タスク = 9ケース × 2モード）
├── launch_both_modes_managed.sh           # 完全自動管理ランチャー
├── resume_both_modes.sh                   # リカバリー・再開用ランチャー
└── .state/                                # ジョブ状態管理ディレクトリ
    ├── both_modes_jobs.txt               # ジョブID記録
    └── both_modes_launch.log             # 実行ログ
```

## 使用方法

### 1. 新規実行（完全自動）

```bash
# バックグラウンドで実行（推奨）
nohup bash scripts/hpc/domain_analysis/fast/launch_both_modes_managed.sh --auto-analyze \
  > launch_output.log 2>&1 &

# フォアグラウンドで実行
bash scripts/hpc/domain_analysis/fast/launch_both_modes_managed.sh --auto-analyze
```

**機能:**
- キューの制限を自動監視
- 3つの手法を順次投入（mean_distance → centroid_umap → lof）
- トレーニング完了後に自動で評価ジョブを投入
- 全完了後に自動で分析・可視化を実行（`--auto-analyze`指定時）

### 2. 実行中パイプラインへの接続・再開

既にジョブが実行中の場合や、中断から再開する場合：

```bash
# バックグラウンドで実行
nohup bash scripts/hpc/domain_analysis/fast/resume_both_modes.sh --auto-analyze \
  > resume_output.log 2>&1 &

# フォアグラウンドで実行
bash scripts/hpc/domain_analysis/fast/resume_both_modes.sh --auto-analyze
```

**機能:**
- 実行中のジョブを自動検出
- 不足しているジョブを補完投入
- 完了済みジョブはスキップ

### 3. 状態確認

```bash
# 進行状況を確認
bash scripts/hpc/domain_analysis/fast/launch_both_modes_managed.sh --status

# ログファイルを確認
tail -f scripts/hpc/domain_analysis/fast/.state/both_modes_launch.log

# キューの状態を確認
qstat -u $USER
```

### 4. 手動での段階的実行

キューの制限が厳しい場合：

```bash
# Step 1: mean_distance のみ実行
qsub -v RANKING_METHOD=mean_distance scripts/hpc/domain_analysis/fast/pbs_train_rank_fast_both_modes.sh
# 完了したら評価を投入
qsub -v RANKING_METHOD=mean_distance scripts/hpc/domain_analysis/fast/pbs_eval_rank_fast_both_modes.sh

# Step 2: centroid_umap を実行
qsub -v RANKING_METHOD=centroid_umap scripts/hpc/domain_analysis/fast/pbs_train_rank_fast_both_modes.sh
qsub -v RANKING_METHOD=centroid_umap scripts/hpc/domain_analysis/fast/pbs_eval_rank_fast_both_modes.sh

# Step 3: lof を実行
qsub -v RANKING_METHOD=lof scripts/hpc/domain_analysis/fast/pbs_train_rank_fast_both_modes.sh
qsub -v RANKING_METHOD=lof scripts/hpc/domain_analysis/fast/pbs_eval_rank_fast_both_modes.sh

# Step 4: 分析・可視化
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
python3 scripts/python/domain_analysis/collect_evaluation_metrics_ranked.py
python3 scripts/python/domain_analysis/visualize_summary_metrics_ranked.py
```

## ジョブ構成

### トレーニングジョブ（各18タスク）

| Task ID | Mode         | Distance    | Level  |
|---------|--------------|-------------|--------|
| 1-9     | source_only  | mmd/dtw/ws  | h/m/l  |
| 10-18   | target_only  | mmd/dtw/ws  | h/m/l  |

### 合計ケース数

- **3 methods** (mean_distance, centroid_umap, lof)
- **× 3 distances** (mmd, dtw, wasserstein)
- **× 3 levels** (high, middle, low)
- **× 2 modes** (source_only, target_only)
- **= 54 ケース**

## 推定実行時間

- トレーニング: ~2分/タスク × 18タスク × 3メソッド = 約1.5-2時間
- 評価: ~30秒/タスク × 18タスク × 3メソッド = 約30分
- **合計: 約2-2.5時間**

## 出力結果

```
results/
├── evaluation/RF/              # 評価結果（各モデルのメトリクス）
│   ├── 14464696/              # mean_distance (source_only)
│   ├── 14464697/              # centroid_umap (source_only)
│   └── ...
└── domain_analysis/
    └── summary/
        ├── csv/
        │   └── summary_ranked_test.csv   # 全結果サマリー
        └── png/
            ├── comparison/               # 手法間比較
            │   ├── summary_metrics_ranked_bar.png
            │   ├── summary_f1_ranked_bar.png
            │   └── ...
            ├── mean_distance/            # 手法別詳細
            ├── centroid_umap/
            └── lof/
```

## トラブルシューティング

### キュー制限エラー

```
qsub: would exceed queue generic's per-user limit
```

**解決策:**
1. 既存ジョブの完了を待つ
2. 手動で段階的に実行（上記参照）
3. `.state/both_modes_jobs.txt` に既存ジョブIDを記録して `resume_both_modes.sh` を使用

### ジョブが見つからない

```bash
# 状態ファイルを確認
cat scripts/hpc/domain_analysis/fast/.state/both_modes_jobs.txt

# 手動でジョブIDを登録
echo "TRAIN_mean_distance=<JOBID>" >> scripts/hpc/domain_analysis/fast/.state/both_modes_jobs.txt
```

### バックグラウンド実行の確認

```bash
# プロセスを確認
ps aux | grep launch_both_modes

# ログをリアルタイム監視
tail -f scripts/hpc/domain_analysis/fast/.state/both_modes_launch.log
```

## 現在の状況（2025-11-29 17:10時点）

実行中のジョブ:
- 14464696[]: mean_distance トレーニング（18タスク）
- 14464697[]: centroid_umap トレーニング（18タスク）

次のステップ:
1. これらのジョブ完了を待つ（約30-40分）
2. 手動で評価とlofトレーニングを投入、または
3. `resume_both_modes.sh` が自動で続行

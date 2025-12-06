# logs/

HPC (PBS) ジョブのログファイルを保存するディレクトリ。

## Directory Structure

```
logs/
├── <experiment_name>/
│   ├── err/    # 標準エラー出力 (*.ER)
│   └── out/    # 標準出力 (*.OU)
└── archive/    # 古い実験のログ
```

## File Naming Convention

PBS ジョブログは以下の形式で保存される:
- `<jobID>[<arrayIndex>].spcc-adm1.OU` - 標準出力
- `<jobID>[<arrayIndex>].spcc-adm1.ER` - エラー出力

## Current Experiments

| Directory | Description | Status |
|-----------|-------------|--------|
| `knn_imbalance/` | 初期KNN不均衡実験 | [archive] 完了 |
| `knn_imbalance_full/` | フル版KNN不均衡実験 | [archive] 完了 |

## Notes

- ログファイルは `.gitignore` により Git 管理対象外
- 新しい実験は `scripts/hpc/` のPBSスクリプトで `-o` / `-e` オプションで出力先を指定
- 古いログは定期的に `archive/` へ移動または削除

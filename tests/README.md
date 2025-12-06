# Tests

このディレクトリには、プロジェクトのテストコードが含まれています。

## テストの実行

### 全テストの実行
```bash
pytest tests/
```

### カバレッジ付きで実行
```bash
pytest tests/ --cov=src --cov-report=html
```

### 特定のテストファイルのみ実行
```bash
pytest tests/test_config.py
```

### 特定のテスト関数のみ実行
```bash
pytest tests/test_config.py::test_paths_exist
```

### 詳細な出力で実行
```bash
pytest tests/ -v
```

### マーカーでフィルタリング
```bash
# 高速なテストのみ
pytest tests/ -m fast

# スモークテストのみ
pytest tests/ -m smoke

# 遅いテストを除外
pytest tests/ -m "not slow"
```

## テストの構成

- `conftest.py`: pytest共通fixtures（サンプルデータ生成、モック環境設定など）
- `test_config.py`: 設定ファイルとパスのテスト
- `test_imbalance_methods.py`: 不均衡データ処理手法のテスト（SMOTE、アンダーサンプリング等）
- `test_smoke_pipeline.py`: データ処理の軽量なエンドツーエンドテスト
- `test_smoke_training.py`: モデル学習の軽量なエンドツーエンドテスト
- `fixtures/`: テスト用ダミーデータ

## テストマーカーの説明

- `@pytest.mark.fast`: 数秒以内に完了する高速なテスト（デフォルト）
- `@pytest.mark.slow`: 1分以上かかる可能性のあるテスト
- `@pytest.mark.smoke`: パイプライン全体が動作することを確認する軽量テスト
- `@pytest.mark.integration`: 複数のコンポーネントを統合したテスト
- `@pytest.mark.requires_data`: 実際のデータセットが必要なテスト

## テストデータ

テスト用のダミーデータは `tests/fixtures/` ディレクトリに配置されています。
実データを使ったテストは、環境変数でデータパスを指定することで実行できます。

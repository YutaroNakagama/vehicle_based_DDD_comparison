cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison && python3 << 'EOF'
print("=" * 100)
print("訓練時と評価時の指標の違いと妥当性の分析")
print("=" * 100)

print("""
【現在の設定】

1. Optunaによるハイパーパラメータ最適化（訓練時）
   目的関数: Average Precision (AP / AUC-PR)
   - 3-fold Cross-Validation on 訓練データ
   - 各foldでAPを計算し、平均値を最大化
   
2. 閾値最適化（評価時）
   目的関数: F2スコア (β=2)
   - 検証データ上でF2を最大化する閾値を探索
   - テストデータには同じ閾値を適用

3. 最終評価（テスト時）
   報告指標: Accuracy, Precision, Recall, F1, AUC, Confusion Matrix
   - 最適化されたF2閾値を使用
   - 複数の指標を総合的に評価


【指標が異なる理由と妥当性】

■ なぜ訓練時と評価時で指標が異なるのか？

  理由1: 目的の違い
    - 訓練時（AP）: モデルの「ランキング能力」を評価
      → 確率予測の質を最大化（閾値に依存しない）
    - 評価時（F2）: 実用的な「判定性能」を評価
      → 具体的な閾値での分類性能を最適化

  理由2: 2段階最適化の利点
    ① Optunaフェーズ: 確率予測が良いモデルを選択（AP最大化）
    ② 閾値調整フェーズ: そのモデルを実用向けに調整（F2最大化）


■ これは問題ないか？

  【結論: 問題ない、むしろベストプラクティス】

  根拠1: 分離の原則
    - モデル選択（Optuna）と判定基準（閾値）を分離
    - モデルの本質的な能力（確率予測）と運用設定（閾値）を独立して最適化

  根拠2: APの利点
    - クラス不均衡に強い（ROC-AUCより現実的）
    - 閾値に依存しない評価（モデル自体の良さを測る）
    - Precision-Recallのトレードオフ全体を考慮

  根拠3: F2の利点（β=2）
    - Recall重視（居眠り検出では見逃しを減らす）
    - Precisionとのバランスも考慮（完全にRecallだけではない）
    - 安全運転支援というドメイン特性に適合

  根拠4: 実世界での標準的なアプローチ
    - Kaggle等でも一般的（AUC最適化 + 閾値調整）
    - 医療診断、異常検知など不均衡問題で広く採用


■ もし指標を統一したら？

  ケース1: OptunaもF2で最適化
    問題点:
      - F2は特定の閾値に依存（汎化性能の評価が不十分）
      - CVの各foldで閾値が変わる可能性
      - 確率予測の質が悪くても、たまたま良い閾値があれば高スコア

  ケース2: 評価もAPで統一
    問題点:
      - 実用時の判定基準（閾値）が決まらない
      - 「確率0.3以上でアラート」のような具体的な運用ができない
      - エンドユーザーにとって分かりにくい


■ 現在のアプローチの利点

  ✓ モデルの本質的な性能（AP）と実用性（F2）の両立
  ✓ 2段階最適化により、より良い解を見つけやすい
  ✓ 閾値変更による運用調整が容易
  ✓ 複数の評価指標で多面的に性能を把握


■ 補足: なぜF2なのか？

  居眠り運転検出の特性:
    - False Negative（見逃し）: 事故につながる可能性 → **絶対に避けたい**
    - False Positive（誤報）: 運転者に不快感を与える → できれば避けたいが許容可能

  F2スコア (β=2):
    - Recallの重みをPrecisionの2倍にする
    - つまり「見逃しを減らすことを優先しつつ、誤報も考慮」
    - F1 (β=1): Precision = Recall の等価バランス
    - F2 (β=2): Recall重視（Precisionの2倍の重み）
    - F0.5 (β=0.5): Precision重視


【推奨事項】

現在の設定は適切ですが、さらに改善するなら:

1. ドメイン専門家との協議
   - 「誤報何回なら、見逃し1回を許容できるか？」
   - → より具体的なβ値の決定

2. 複数の閾値候補を保存
   - F1最適化、F2最適化、Precision=0.10固定など
   - → 運用時に切り替え可能

3. キャリブレーション
   - 確率予測の信頼性を向上（現在はRFで実施済み）
   - → より良いAP、より良い閾値選択

4. コスト考慮の最適化
   - FNコスト >> FPコストを明示的にモデルに組み込む
   - → class_weight や sample_weight の調整（一部実施済み）


【結論】

訓練時（AP）と評価時（F2）で指標が異なることは:
  ✓ 理論的に妥当
  ✓ 実用的に有効
  ✓ ベストプラクティスに準拠
  ✓ ドメイン特性（安全最優先）に適合

→ 現在のアプローチは適切であり、変更の必要はありません。
""")

EOF
cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison && python3 << 'EOF'
import os
from pathlib import Path

print("=" * 100)
print("Recall重視 vs Precision重視の実験計画")
print("=" * 100)

print("""
【現状の確認】

現在のモデル設定（Recall重視）:
  - Optuna目的関数: Average Precision (AP)
  - 閾値最適化: F2スコア (β=2) - Recall重視
  - class_weight: balanced または {0:1.0, 1:10.0}
  - sample_weight: クラスバランス調整

最新の実験結果:
  - Job ID: 14354364
  - 保存場所: results/evaluation/RF/14354364/
  - モデル: models/RF/14354364/


【バックアップ戦略】

ステップ1: 現在の結果をバックアップ
ステップ2: コードを修正してPrecision重視に変更
ステップ3: 新しい実験を実行
ステップ4: 結果を比較分析


【具体的な実装方法】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
方法1: 実験タグ（experiment_tag）を使用（推奨）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

現在のコードは既に suffix パラメータをサポートしています。
これを利用して実験を区別します。

Recall重視（現在）: suffix=""（デフォルト）
Precision重視（新規）: suffix="_precision_focused"

保存先の例:
  Recall重視:
    - models/RF/{job_id}/model_RF_target_only.pkl
    - results/evaluation/RF/{job_id}/target_only/eval_results_mmd_mean_low.json
  
  Precision重視:
    - models/RF/{job_id}/model_RF_target_only_precision_focused.pkl
    - results/evaluation/RF/{job_id}/target_only_precision_focused/eval_results_mmd_mean_low.json


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
方法2: 別ディレクトリに保存
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recall重視: results/evaluation/RF_recall/{job_id}/
Precision重視: results/evaluation/RF_precision/{job_id}/


【Precision重視への変更内容】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
変更1: 閾値最適化をF0.5（Precision重視）に変更
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ファイル: src/models/architectures/common.py
行番号: 約700行目

現在（F2 = Recall重視）:
  best_threshold, best_f2 = find_optimal_threshold(y_val, m_val["_proba"], beta=2.0)

変更後（F0.5 = Precision重視）:
  best_threshold, best_f05 = find_optimal_threshold(y_val, m_val["_proba"], beta=0.5)

F-betaスコアの意味:
  - β < 1: Precision重視
  - β = 1: Precision = Recall（F1）
  - β > 1: Recall重視
  
  - β = 0.5: PrecisionをRecallの2倍重視
  - β = 2.0: RecallをPrecisionの2倍重視


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
変更2: class_weightの調整（オプション）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ファイル: src/models/architectures/common.py
行番号: 約595行目（RFの定義）

現在（Positive重視）:
  best_clf = RandomForestClassifier(**best_params, class_weight={0:1.0, 1:10.0}, n_jobs=1)

変更後（よりバランス）:
  best_clf = RandomForestClassifier(**best_params, class_weight={0:1.0, 1:3.0}, n_jobs=1)
  # または
  best_clf = RandomForestClassifier(**best_params, class_weight="balanced", n_jobs=1)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
変更3: Optuna目的関数の変更（オプション）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

現在はAP（Average Precision）を使用しており、これは既にPrecisionを考慮しています。
大きな変更は不要ですが、より明示的にPrecision@固定Recallなども検討可能。

# そのまま使用でOK（APは既にPrecision-Recallのバランスを評価）
ap = average_precision_score(y_va, p)


【実装手順】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
手順1: 現在の結果をバックアップ（念のため）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

コマンド:
  # 最新のJob IDを確認
  cat models/RF/latest_job.txt
  
  # バックアップディレクトリ作成
  mkdir -p results/evaluation/RF_backup_recall_focused
  mkdir -p models/RF_backup_recall_focused
  
  # 現在の結果をコピー
  cp -r results/evaluation/RF/14354364 results/evaluation/RF_backup_recall_focused/
  cp -r models/RF/14354364 models/RF_backup_recall_focused/
  
  # サマリーCSVもバックアップ
  cp results/domain_analysis/summary/csv/summary_40cases_*.csv \\
     results/domain_analysis/summary/csv/summary_recall_focused_backup.csv


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
手順2: コードを修正
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ファイル: src/models/architectures/common.py

修正箇所1: 閾値最適化（必須）
  行番号: 700行目付近
  
  変更前:
    best_threshold, best_f2 = find_optimal_threshold(y_val, m_val["_proba"], beta=2.0)
    logging.info(f"Optimal threshold for F2 (β=2): {best_threshold:.3f} (F2={best_f2:.3f})")
  
  変更後:
    best_threshold, best_f05 = find_optimal_threshold(y_val, m_val["_proba"], beta=0.5)
    logging.info(f"Optimal threshold for F0.5 (β=0.5): {best_threshold:.3f} (F0.5={best_f05:.3f})")

修正箇所2: F-betaスコア計算（必須）
  行番号: 707行目付近
  
  変更前:
    metrics["f2"] = float(fbeta_score(y_true, yhat, beta=2, zero_division=0))
  
  変更後:
    metrics["f05"] = float(fbeta_score(y_true, yhat, beta=0.5, zero_division=0))

修正箇所3: class_weight調整（オプション）
  行番号: 595行目付近
  
  変更前:
    best_clf = RandomForestClassifier(**best_params, class_weight={0:1.0, 1:10.0}, n_jobs=1)
  
  変更後:
    best_clf = RandomForestClassifier(**best_params, class_weight={0:1.0, 1:3.0}, n_jobs=1)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
手順3: 実験スクリプトの準備
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

新しいランチャーを作成（テスト用）:

ファイル: scripts/hpc/domain_analysis/test/launch_domain_analysis_precision_test.sh

内容:
  #!/bin/bash
  # Precision重視の実験用ランチャー
  
  # 環境変数でモード識別
  export EXPERIMENT_MODE="precision_focused"
  
  # 通常のランチャーを呼び出し
  bash scripts/hpc/domain_analysis/test/launch_domain_analysis_test.sh


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
手順4: 実験実行
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

テスト実行:
  cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison
  bash scripts/hpc/domain_analysis/test/launch_domain_analysis_test.sh

本番実行:
  bash scripts/hpc/domain_analysis/full/launch_domain_analysis.sh


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
手順5: 結果比較
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

比較スクリプト作成:

ファイル: scripts/python/domain_analysis/compare_recall_vs_precision.py

機能:
  - Recall重視とPrecision重視の結果を並べて比較
  - Confusion Matrixの差分を可視化
  - Precision-Recallトレードオフのプロット
  - 最適なバランスポイントの提案


【期待される結果の違い】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recall重視（現在: β=2, class_weight={0:1, 1:10}）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

予想:
  - Recall: 高い（90-95%）
  - Precision: 低い（5-10%）
  - FP: 多い（~2000件）
  - FN: 少ない（~10件）
  
実用例:
  - 1時間で誤報 20回、見逃し 0.1回
  - アラート疲れのリスク高い
  - 安全性は最高


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Precision重視（新規: β=0.5, class_weight={0:1, 1:3}）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

予想:
  - Recall: 中程度（70-80%）
  - Precision: 高め（15-25%）
  - FP: 少ない（~500件）
  - FN: 増加（~40件）
  
実用例:
  - 1時間で誤報 5回、見逃し 0.4回
  - アラートの信頼性向上
  - 安全性はやや低下


【どちらが良いか？】

用途別の推奨:

1. 安全運転支援システム（主目的）
   → Recall重視（現在の設定）
   理由: 見逃しは絶対に避けたい
   
2. 補助的な注意喚起
   → Precision重視
   理由: 信頼性が重要、誤報でドライバーが無視するようになるのを防ぐ
   
3. 段階的警告システム
   → 両方を組み合わせ
   - レベル1（軽度）: Precision重視（高い閾値）
   - レベル2（中度）: バランス型（F1最適化）
   - レベル3（重度）: Recall重視（低い閾値）


【実装の優先順位】

高優先度:
  ✓ 閾値最適化をF0.5に変更（手順2の修正箇所1-2）
  ✓ テスト実行して結果確認
  ✓ 比較分析

中優先度:
  △ class_weightの調整
  △ 複数の閾値候補を保存

低優先度:
  △ Optuna目的関数の変更（現在のAPで十分）


【次のステップ】

1. 今すぐ実行可能:
   - common.pyの修正（beta=0.5に変更）
   - テスト実験の実行
   
2. コードを修正して実験を開始しますか？
   それとも、まず修正内容の詳細を確認しますか？
""")

# 現在の結果ディレクトリを確認
print("\n" + "=" * 100)
print("現在の結果ディレクトリの確認")
print("=" * 100)

latest_job_file = Path("models/RF/latest_job.txt")
if latest_job_file.exists():
    with open(latest_job_file) as f:
        latest_job = f.read().strip()
    print(f"\n最新のJob ID: {latest_job}")
    
    eval_dir = Path(f"results/evaluation/RF/{latest_job}")
    if eval_dir.exists():
        json_files = list(eval_dir.rglob("*.json"))
        print(f"評価結果ファイル数: {len(json_files)}件")
    
    model_dir = Path(f"models/RF/{latest_job}")
    if model_dir.exists():
        pkl_files = list(model_dir.rglob("*.pkl"))
        print(f"モデルファイル数: {len(pkl_files)}件")
else:
    print("\nlatest_job.txt が見つかりません")

EOF
cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison && python3 << 'EOF'
print("=" * 100)
print("Precision重視実験の開始")
print("=" * 100)

print("""
【実行内容】

✅ ステップ1: バックアップ完了
   - 既存のRecall重視結果を保存
   - 場所: results/evaluation/RF_backup_recall_focused/14354364/

✅ ステップ2: コード修正完了
   変更1: class_weight {0:1.0, 1:10.0} → {0:1.0, 1:3.0}
   変更2: 閾値最適化 F2 (β=2.0) → F0.5 (β=0.5)
   変更3: メトリクス計算 f2 → f05

✅ ステップ3: テスト実験投入完了
   Train Job:   14357100[].spcc-adm1
   Eval Job:    14357101[].spcc-adm1
   Summary Job: 14357102.spcc-adm1


【投入されたジョブの内容】

1. Training (14357100)
   - 6ケース（2 ranks × 3 modes）
   - N_TRIALS=5（高速テスト）
   - Precision重視の設定で訓練
   - 推定時間: ~10-15分

2. Evaluation (14357101)
   - 訓練完了後に自動開始
   - 6ケースすべてを評価
   - 推定時間: ~5-10分

3. Summary/Visualization (14357102)
   - 評価完了後に自動開始
   - CSVサマリー作成
   - PNG可視化生成
   - 推定時間: ~1-2分


【期待される結果の違い】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Recall重視（既存: β=2, weight=10）vs Precision重視（新規: β=0.5, weight=3）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

指標          Recall重視    Precision重視    変化の方向
────────────────────────────────────────────────────────
Recall        高い(90%+)    中程度(70-80%)   ↓ 減少
Precision     低い(5-10%)   高め(15-25%)     ↑ 増加
F0.5          低い          高い             ↑ 増加
FP（誤報）    多い(~2000)   少ない(~500)     ↓ 大幅減少
FN（見逃し）  少ない(~10)   増加(~40)        ↑ 増加
閾値          低い(0.2-0.3) 高い(0.5-0.6)    ↑ 上昇
────────────────────────────────────────────────────────

実用的な影響:
  - 誤報頻度: 20回/時間 → 5回/時間（改善）
  - 見逃し率: 0.1回/時間 → 0.4回/時間（悪化）
  - アラート信頼性: 大幅向上
  - 安全性: やや低下（要許容度判断）


【進捗確認コマンド】

ジョブステータス確認:
  qstat -u s2240011

詳細ログ確認:
  # 訓練ログ
  tail -f models/RF/latest_job.txt  # Job IDを確認
  tail -f train_rf*.o14357100
  
  # 評価ログ
  tail -f eval_rf*.o14357101


【結果確認（完了後）】

結果の場所:
  - 評価JSON: results/evaluation/RF/14357100/
  - サマリーCSV: results/domain_analysis/summary/csv/
  - 可視化PNG: results/domain_analysis/summary/png/

比較コマンド:
  # Recall重視とPrecision重視の比較
  python3 scripts/python/domain_analysis/compare_experiments.py \\
    --recall_job 14354364 \\
    --precision_job 14357100


【次のステップ（テスト完了後）】

1. テスト結果の確認
   - Precision/Recallのトレードオフを確認
   - 許容可能な性能かを判断

2. 本番実験の実行
   - 満足できる結果なら full/ ランチャーで本番実行
   - N_TRIALS=20, 27ケース（9 ranks × 3 modes）

3. 結果比較と最終判断
   - Recall重視 vs Precision重視の詳細比較
   - 用途に応じた最適設定の決定


【推定完了時刻】

テスト開始: 現在
テスト完了: 約20-30分後

その後、結果を確認して次のステップを決定します。
""")

EOF
cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison && python3 << 'EOF'
print("=" * 100)
print("summary_metrics_bar.png 可視化コードの概要")
print("=" * 100)

print("""
【現在の可視化構造】

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. メインスクリプト
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ファイル: scripts/python/domain_analysis/visualize_summary_metrics.py

役割:
  - summary_40cases_test.csv を読み込み
  - pos_rate（正例率）を動的に取得してAUPRCベースラインを設定
  - モード検出: pooled, source_only, target_only
  - 距離メトリック検出: dtw, mmd, wasserstein
  - レベル検出: high, middle, low
  - 8つのメトリクスを可視化: auc, recall, specificity, precision, f1, f2, accuracy, auc_pr

現在の処理フロー:
  1. CSV読み込み
  2. pos_rateから動的ベースライン算出（デフォルト0.033）
  3. plot_grouped_bar_chart_raw() 呼び出し
  4. 図を保存（DPI=200）


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. 可視化関数（コア実装）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ファイル: src/utils/visualization/visualization.py
関数: plot_grouped_bar_chart_raw()

【レイアウト構造】

4行 × N列（N = メトリクス数、現在8個）

行1: DTW距離         [high, middle, low] - source_only vs target_only
行2: MMD距離         [high, middle, low] - source_only vs target_only  
行3: Wasserstein距離 [high, middle, low] - source_only vs target_only
行4: Pooled         [単一バー] - pooledモードのみ


【各サブプロット】

- 行1-3（距離メトリクス行）:
  * X軸: high, middle, low（3本）
  * 棒グラフ: source_only（緑系）, target_only（青系）
  * 2色で比較可視化
  * width=0.35で並べて配置

- 行4（Pooled行）:
  * X軸: "Pooled Baseline"（1本のみ）
  * 棒グラフ: pooled（オレンジ系）
  * 全被験者混合のベースライン性能
  * width=0.6で中央配置


【ベースライン表示】

- auc_pr列のみ:
  * 水平破線（gray, linestyle='--'）
  * 正例率（pos_rate）をベースラインとして表示
  * テキスト: "Baseline (0.033)" 形式
  * Y軸範囲を動的調整（margin=30%）


【色分け】

colors = ["#66cc99", "#6699cc", "#ff9966"]
  - colors[0]: Pooled用（緑系）
  - colors[1]: Source-only用（青系）
  - colors[2]: Target-only用（オレンジ系）


【凡例配置】

- 位置: 最上行・最右列のサブプロット
- 表示内容:
  * 行1-3: "Source-only", "Target-only"
  * 行4: "Pooled"


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. データ構造要件
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

入力CSVフォーマット（summary_40cases_test.csv）:

必須列:
  - mode: pooled / source_only / target_only
  - distance: dtw / mmd / wasserstein（pooledはNaN）
  - level: high / middle / low（pooledはNaN）
  - メトリクス列: auc, recall, specificity, precision, f1, f2, accuracy, auc_pr
  - pos_rate: 正例率（AUPRCベースライン用）

データ形式: Long format（unpivoted）
  - 各行 = 1ケース
  - mode列でモード識別
  - 距離メトリクス行: distance列で分類
  - Pooled行: distance=NaN, level=NaN


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. 強みと制約
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【強み】

✓ 柔軟なデータ構造
  - Long formatで処理（pivotなし）
  - mode列で動的に適応モード検出
  - 距離メトリクス・レベルを自動検出

✓ 動的ベースライン
  - pos_rateから自動算出
  - AUPRCの実用的な基準線

✓ 明確な視覚的階層
  - 4行で論理分割（3距離 + 1ベースライン）
  - 色分けで直感的な比較

✓ 保守性の高い設計
  - メトリクス追加が容易（metricsリストに追加）
  - タイトルマップで表示名カスタマイズ可能


【制約・改善余地】

⚠ Confusion Matrix情報が欠如
  - Recall/Precisionは表示されるが、TP/FP/TN/FNは不明
  - 誤報数・見逃し数が直接見えない
  - 実用判断に追加情報が必要

⚠ 統計的有意差が不明
  - エラーバーなし（標準偏差・信頼区間）
  - 複数ランクの平均を取っているが、ばらつきが不明
  - 差が有意かどうか判断困難

⚠ トレードオフ可視化が弱い
  - Recall vs Precision の関係が別々のサブプロット
  - F2スコアだけでは全体像が見えにくい
  - PR曲線やROC曲線がない

⚠ Pooled行の情報密度が低い
  - 単一バーだけで空白が多い
  - distance/levelがないため他と直接比較しにくい
  - 行全体を使う必要性に疑問

⚠ Y軸範囲の一貫性
  - ベースラインありメトリクスは動的範囲（margin 30%）
  - ベースラインなしメトリクスは固定範囲（0-1）
  - 比較時に注意が必要


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. 議論の方向性候補
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A. レイアウト改善
   - Pooled行を独立図にする？
   - 4行→3行にして情報密度を上げる？
   - Heatmap形式に変更？

B. メトリクス追加・変更
   - Confusion Matrix要素（TP/FP/TN/FN）を追加？
   - F2スコアを強調？
   - 誤報率（FPR）を追加？

C. 統計情報の追加
   - エラーバー（標準偏差）を追加？
   - 信頼区間を表示？
   - 有意差検定結果をアノテーション？

D. トレードオフ可視化
   - Recall-Precision散布図を追加？
   - PR曲線を別図で作成？
   - Cost-Benefit分析図？

E. Precision重視結果との比較
   - 2つの実験結果を並べて表示？
   - 差分ヒートマップ？
   - Before/After比較図？


【優先度の高い議論トピック】

1️⃣ Pooled行の扱い
   - 現状は情報密度が低く、空白が多い
   - 別図にする or レイアウト変更？

2️⃣ Confusion Matrix要素の可視化
   - 実用判断に必須の情報
   - FP数（誤報）、FN数（見逃し）を直接表示したい

3️⃣ Precision重視結果の比較方法
   - 今後2つの実験結果を比較する必要
   - 効率的な比較可視化方法は？
""")

print("=" * 100)
print("どの議論トピックに焦点を当てますか？")
print("=" * 100)
EOF
# Domain Analysis HPC Job Scripts

このディレクトリには、ドメイン分析のためのHPCジョブスクリプトが含まれています。

## ディレクトリ構造

```
scripts/hpc/domain_analysis/
├── full/                 # 本番用ジョブスクリプト
│   ├── pbs_compute_distance.sh    # 距離計算 (MMD, Wasserstein, DTW)
│   ├── pbs_ranking.sh             # ランキング生成
│   ├── pbs_train_rank.sh          # 学習 (9ランク × 3モード)
│   ├── pbs_eval_rank.sh           # 評価 (9ランク × 3モード)
│   ├── pbs_analysis.sh            # 結果集約と可視化
│   └── launch_domain_analysis.sh  # パイプライン起動スクリプト
├── test/                 # テスト用ジョブスクリプト
│   ├── pbs_compute_distance.sh           # 距離計算 (テスト: 8コア, 2h)
│   ├── pbs_ranking.sh                    # ランキング生成 (テスト)
│   ├── pbs_train_rank.sh                 # 学習 (テスト: 2ランク, 5トライアル)
│   ├── pbs_eval_rank.sh                  # 評価 (テスト)
│   ├── pbs_analysis.sh                   # 結果集約 (テスト)
│   ├── launch_domain_analysis_test.sh    # 部分パイプライン (train→eval→summary)
│   └── launch_domain_analysis_full_test.sh  # フルパイプライン (distance→ranking→train→eval→summary)
├── baseline/             # ベースライン実験用スクリプト
│   ├── launch_RF_baseline.sh       # RF ベースライン (本番)
│   └── launch_RF_baseline_test.sh  # RF ベースライン (テスト)
├── lib/                  # 共通ライブラリ
│   └── common.sh         # 環境設定、スレッド制限、ログ関数
└── archive/              # 旧スクリプト (アーカイブ)
```

## 使用方法

### 本番環境での実行

```bash
# フルパイプライン (距離計算済みの場合)
cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison
./scripts/hpc/domain_analysis/full/launch_domain_analysis.sh

# 個別ジョブの投入
qsub scripts/hpc/domain_analysis/full/pbs_train_rank.sh
```

### テスト環境での実行

```bash
# 部分パイプライン (train → eval → summary)
./scripts/hpc/domain_analysis/test/launch_domain_analysis_test.sh

# フルパイプライン (distance → ranking → train → eval → summary)
./scripts/hpc/domain_analysis/test/launch_domain_analysis_full_test.sh
```

### ベースライン実験

```bash
# 本番
./scripts/hpc/domain_analysis/baseline/launch_RF_baseline.sh

# テスト
./scripts/hpc/domain_analysis/baseline/launch_RF_baseline_test.sh
```

## リソース設定の比較

| スクリプト | 本番 | テスト |
|-----------|------|--------|
| **pbs_compute_distance.sh** | 16コア, 24h, SINGLE queue | 8コア, 2h, DEFAULT queue |
| **pbs_ranking.sh** | 4コア, 2h | 4コア, 0.5h |
| **pbs_train_rank.sh** | 8コア, 256GB, 48h, 9ランク | 4コア, 16GB, 1h, 2ランク |
| **pbs_eval_rank.sh** | 4コア, 32GB, 6h, 9ランク | 4コア, 32GB, 1h, 2ランク |
| **pbs_analysis.sh** | 4コア, 16GB, 1h | 4コア, 16GB, 1h |

### テストモードの特徴
- `N_TRIALS_OVERRIDE=5` (本番: 20)
- `KSS_SIMPLIFIED=1` (簡略化されたKSSラベル)
- データ分割: `0.4:0.3:0.3` (train:val:test)
- ランク数: 2 (本番: 9)

## 共通ライブラリの使用

各スクリプトで共通ライブラリを使用する場合:

```bash
source "$(dirname "$0")/../lib/common.sh"

# 環境設定
setup_environment "$PROJECT_ROOT"

# スレッド制限設定
setup_thread_limits

# ジョブ情報のログ
log_job_info

# ... 処理 ...

# 完了ログ
log_job_complete
```

## ジョブの監視

```bash
# ジョブステータスの確認
qstat -u $USER

# ログの確認
tail -f scripts/hpc/log/*.o*
```

## 注意事項

1. **パス参照**: すべてのランチャースクリプトは`SCRIPT_DIR`変数を使用し、相対パスで他のスクリプトを参照します
2. **依存関係**: ジョブチェーンは`qsub -W depend=afterok:${job_id}`で管理されます
3. **ログ**: すべてのジョブログは`scripts/hpc/log/`に保存されます
4. **ランキング**: 距離計算とランキングは事前に実行する必要があります。結果は`ranks29/{ranking_method}/`ディレクトリに保存されます
   - `mean_distance`: 平均距離ベース（デフォルト）
   - `centroid_mds`: MDS空間でのcentroidからの距離
   - `centroid_umap`: UMAP空間でのcentroidからの距離
   - `medoid`: Medoidからの距離
   - `lof`: Local Outlier Factor

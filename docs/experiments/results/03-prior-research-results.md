# 実験3（先行研究再現）の結果

> **改訂**: domain_train 統一版（252 ジョブ）に移行後のステータス。
> 旧 split2 版（source_only/target_only, 504 ジョブ）のデータは参考値として記載。

---

## 実験ステータス

**状態:** 実行中（2026-02-20 時点）

### ジョブ投入・完了状況（domain_train 統一版）

| モデル | 目標 | 完了(exit=0) | 実行中 | キュー内 | 未投入 | デーモン |
|---|---|---|---|---|---|---|
| SvmW | 84 | 84 | 0 | 0 | 0 | 完了 ✅ |
| SvmA | 84 | 0 | — | — | — | 稼働中（auto_resub_unified_v2.sh） |
| Lstm | 84 | 0 | — | — | — | 稼働中（auto_resub_unified_v2.sh） |

> SvmA / Lstm の詳細進捗は `qstat -u $USER` で確認。

### SvmW 完了検証（84件 exit=0）: 全件正常 ✅

検証項目:
- モデルファイル（`SvmW_*.pkl`）: 84/84 ✅
- Optuna study（`optuna_*.pkl`）: 84/84 ✅
- スケーラー（`scaler_*.pkl`）: 84/84 ✅
- 評価結果 JSON（`*_within.json` + `*_cross.json`）: 168/168 ✅（84 × 2 eval types）
- トレーニング結果 JSON: 84/84 ✅

> SvmW domain_train の平均実行時間: 約15分（旧 split2 版の約5時間から大幅短縮）

---

## 既知の問題

### 1. Lstm `seq_len` バグ（解決済み — 旧 split2 版）

- **原因:** commit 278697c で `seg_len` → `seq_len` のタイプミスが混入
- **影響:** 修正前に投入された Lstm ジョブは全て `NameError: name 'seq_len' is not defined` で学習失敗
- **修正:** commit 49cf96e で `seq_len` → `seg_len` に修正
- **対処:** domain_train 統一版では修正済みコードを使用

### 2. SMOTE 系条件の walltime 超過（対策済み）

- **原因:** SMOTE/smote_plain 条件は計算量が大きく、元の walltime では不足
- **修正:** walltime を増加（SvmW: 24h, SvmA: 48h, Lstm: 24h）

### 3. 評価結果ファイル名の上書きバグ（解決済み）

- **原因:** domain_train モードでは within と cross の2回評価が実行されるが、
  旧 `save_eval_results` は `eval_type` をファイル名に含めなかったため、
  cross 評価が within 評価の結果ファイルを上書きしていた
- **修正:** `src/utils/io/savers.py` に `eval_type` サフィックスを追加
  - `eval_results_*_within.json` / `eval_results_*_cross.json` として別ファイルに保存
- **対処:** SvmW の 83 件は手動リネーム + 1件再評価で修復済み

### 4. 旧 split2 版の訓練重複問題（domain_train で解決）

- **原因:** source_only と target_only で同一のモデルが2回訓練されていた
  （例: in_domain の target_only と out_domain の source_only は同じ in_domain データで訓練）
- **修正:** domain_train モードに統一し、各ドメインで1回のみ訓練、2回評価

---

## 旧 split2 版の参考データ（2026-02-17 時点）

<details>
<summary>旧 split2 版のステータス（参考）</summary>

### ジョブ投入・完了状況（旧 split2 版）

| モデル | 目標 | 完了(exit=0) | 備考 |
|---|---|---|---|
| SvmW | 168 | 92 | domain_train 統一版に移行 |
| SvmA | 168 | 0 | domain_train 統一版に移行 |
| Lstm | 168 | 4 | domain_train 統一版に移行 |

### 蓄積 eval 結果数（split2、旧ジョブ含む）

| モデル | eval 件数 | 目標 |
|---|---|---|
| SvmW | 274 | 168 |
| SvmA | 233 | 168 |
| Lstm | 323 | 168 |

> 目標を超える値は、旧設定・旧コードでのジョブ結果（コード修正前）を含むため。
> 最終分析では domain_train 統一版の結果のみを使用する。

</details>

---

## 関連ドキュメント

- [実験条件](../conditions/03-prior-research-conditions.md) — 条件マトリクスと HPC リソース設定
- [再現性ガイド](../reproducibility.md) — 実験の再現方法
- [先行研究モデル](../../architecture/prior_research.md) — モデルアーキテクチャ詳細

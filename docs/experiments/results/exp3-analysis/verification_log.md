# exp3 検証ログ (verification log)

*作成: 2026-06-27. [`verification_tasks.md`](verification_tasks.md) の T1–T6 に対する実施記録と結果。*
*根拠の本体は [`domain_imbalance_factor_analysis.md`](domain_imbalance_factor_analysis.md)。本ログはその主張を確定/修正するために実施した検証の一次記録。*

## 方針更新 (advisor, 2026-06-27)
exp3 の主軸を **before/after** から **Cross-domain vs Within-domain (in/out)** に変更。
RF/SvmW/SvmA/Lstm の4手法を **Within (target_only)** と **Cross (source_only)** の
in/out で比較し、RF の優位性(特に cross-domain 頑健性)を示す。

- 実装: [`scripts/python/train/c1_domain_launcher.py`](../../../../scripts/python/train/c1_domain_launcher.py),
  [`scripts/shell/c1_watchdog.sh`](../../../../scripts/shell/c1_watchdog.sh)。
- グリッド: 4モデル × {target_only, source_only} × {in_domain, out_domain} × seed{42,123,2025},
  SW-SMOTE 固定 / wasserstein / ratio 0.5。
- Within-in は B1 から流用(同一 target_timewise eval)。ただし **SvmA は T1 修正の影響で全4条件を再実行**。
- cross-domain (source_only) は exp2 互換タグ(`imbalv3_knn_..._split2_...`)で
  `rankings/split2/knn/<dist>_<oppositeDomain>.txt` をソース群に解決
  ([`target_resolution.py:172-223`](../../../../src/utils/io/target_resolution.py#L172))。

## 進捗サマリ

| T | 内容 | 状態 | 結論 |
|---|---|---|---|
| **T1** | SvmA を Arefnezhad 完全18特徴で再テスト | ✅ **完了** | **「信号なし」を強化**(特徴フィルタの産物ではない) |
| **T2** | SvmW 0.79 が honest signal か split 依存か | 🟡 実行中 | (下記) |
| T3 | RF の SMOTE 単独効果 (pooled+SMOTE) | ⬜ 未 | — |
| **T4** | SvmA の 分類器×特徴 deconfound | 🟡 半分完了 | RF-on-SvmA特徴 = 0.496 → **特徴が壁(分類器非依存)** |
| T5 | Lstm の domain 帰属 | 🟡 進行 | IV25 before(local)=0.512 ✅ / cross は c1 で測定中 |
| T6 | road-curve 除去の faithfulness | ⬜ 未 | — |

---

## T1. SvmA 完全特徴セット再テスト — ✅ 完了

### 修正 (コード)
[`SvmA.py`](../../../../src/models/architectures/SvmA.py#L65) の `SVMA_PAPER_FEATURE_SUFFIXES`
を不忠実な14種から **Arefnezhad 忠実18種** に修正(commit `b715535`)。
- **追加(8)**: SampleEntropy, KatzFractalDim, ShannonEntropy, SpectralFlux, FreqVar(=Frequency
  Variability), Quartile25/Median/Quartile75。
- **削除(4)**: Mean, Variance, Max, Min(Arefnezhad に無い)。
- 検証: フィルタは **36列** を選択(Steering 18 + SteeringSpeed 18、未マッチ0)。
- 注: `verification_tasks.md` は Frequency Variability を「未計算」としていたが、データに
  `Steering_FreqVar` / `SteeringSpeed_FreqVar` が存在し、18特徴すべてを忠実に構成できた。

### 再テスト (feature-signal probe)
[`scripts/python/analysis/exp3_feature_signal_probe.py`](../../../../scripts/python/analysis/exp3_feature_signal_probe.py)。
n=66,993(87被験者, pos 3.62%)、**被験者分離 split**(GroupShuffleSplit)。SvmA の KSS マッピング
(1–6=Alert, 8–9=Drowsy, 7除外)はパイプライン同一。

| 特徴セット | univariate max (>0.55) | RBF-SVM raw | RBF-SVM +SMOTE | **RF on same feats** |
|---|---|---|---|---|
| **忠実18 (36列)** | **0.515 (0)** | 0.494 | 0.507 | **0.496** |
| 旧14 (28列) | 0.515 (0) | 0.498 | 0.507 | 0.513 |

- per-feature 最高は `SteeringSpeed_ZeroCrossingRate` 0.515。**原論文の主力 `SampleEntropy` も 0.514(chance)**。
- 多変量はどの分類器でも 0.49–0.51。

### 結論
**忠実18特徴でも全指標が chance(<0.55)。** 原論文の主力特徴(Sample Entropy 等)を加えても信号は出ない。
- 旧 null(0.515/0.509)は **特徴フィルタの産物ではなかった** → 「SvmA の特徴は KSS ラベルに対し信号を持たない」を**強化**。
- **RF(強力な別分類器)を同じ忠実特徴で学習しても 0.496** → 壁は**分類器でなく特徴**(= T4 の RF-on-SvmA特徴 半分を同時に確定)。

### 解析doc への反映
[`domain_imbalance_factor_analysis.md`](domain_imbalance_factor_analysis.md) §2.3 / §9 の
「SvmA 信号なし(⚠️ 特徴セット不忠実)」を **「✅ 忠実18特徴で再確認済(T1)」** に格上げ可能。
- 残: c1 の SvmA(忠実特徴・ANFIS+PSO 選択込み・全4条件)完走後に B1 数値(0.539)を忠実版へ置換。

---

## T2. SvmW clean-split 検証 — 🟡 実行中

### probe
[`scripts/python/analysis/exp3_svmw_split_probe.py`](../../../../scripts/python/analysis/exp3_svmw_split_probe.py)。
8 GHM 帯域 + default KSS、**同一の SMOTE+RBF-SVM/RF** で split のみを変えて比較:
- A) 被験者内 temporal(被験者ごと Timestamp ソート, 先頭70%→train / 末尾30%→test) = B1 target_only 相当
- B) 被験者分離(GroupShuffleSplit, 3 seed 平均)

判定: **A ≫ B(B≈chance)→ 0.79 は split 依存**。A≈B かつ両者上昇 → 真の潜在信号。

### 結果
*(probe 実行中。完了後に追記。)*

---

## T4/T5 メモ
- **T4(半分)**: 上記 RF-on-SvmA忠実特徴 = 0.496 → SvmA の null は分類器・特徴選択に依らない(特徴の壁)。
  残りの SVM-on-RF特徴 ≈ 0.78(分類器は壁でない)は別途。
- **T5**: IV2025 before(local pooled)Lstm = 0.512(n=6、公表 0.52 と整合)。
  Lstm cross-domain は **c1 の source_only Lstm** で測定中(within ≫ cross なら「向上は domain 由来」を確定)。

## 残タスク
- T2 完了 → 結果追記。
- T3(pooled+SMOTE の RF 1セル)、T6(Aygun コース曲率の確認)。
- c1 全完走 → before/within/cross 集計表 + プロット(verification_tasks の図1–6)。
- 独立サブエージェントによる結論の敵対的再検証(c1 データ確定後)。

## 参照
- probe スクリプト: [`exp3_feature_signal_probe.py`](../../../../scripts/python/analysis/exp3_feature_signal_probe.py),
  [`exp3_svmw_split_probe.py`](../../../../scripts/python/analysis/exp3_svmw_split_probe.py)
- 結果 JSON: `results/analysis/exp3_verification/t1_feature_signal_probe.json`, `t2_svmw_split_probe.json`
- c1: [`c1_domain_launcher.py`](../../../../scripts/python/train/c1_domain_launcher.py), [`c1_watchdog.sh`](../../../../scripts/shell/c1_watchdog.sh)

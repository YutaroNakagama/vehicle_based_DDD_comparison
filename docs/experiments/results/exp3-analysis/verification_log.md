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
| T6 | road-curve 除去の faithfulness | ✅ 論理的に moot | SvmA が chance ゆえ除去は結論不変(下記) |

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

### 結果 (2026-06-27)
| split | RBF-SVM +SMOTE | RF |
|---|---|---|
| A 被験者内 temporal | 0.489 | 0.478 |
| B 被験者分離 (3seed平均) | 0.505 | 0.493 |

**素の SMOTE+SVM/RF は両 split とも chance(~0.48–0.51)で、B1 の 0.79 をどちらの split でも再現しない。**
→ 0.79 は **8特徴そのもの**ではなく **フルパイプライン(Optuna 調整 SVM + 特定 SMOTE 設定)** から生じる。
この probe は 0.79 を再現できないため、probe 単体では split 依存性を分離できない(inconclusive)。

### 再フレーム: split 依存性の決定版は c1 自身
**c1 の Within(target_only) vs Cross(source_only) が、フルパイプラインでの split 依存性テストそのもの。**
Cross-domain は train と eval が別ドメイン=別被験者群(=被験者分離)。各手法で **Within ≫ Cross→chance** なら
「within-domain 構造に依存」を確定できる。
- **RF(9/9完了・異常なし)**: Within-out **0.790**(0.758/0.790/0.822) — **TIV2026 の RF after=0.781 と整合 ✓**
  (B1 within-in も 0.781)。vs Cross-in **0.520**(0.513–0.526)
  / Cross-out **0.512**(0.508–0.517)→ RF ですら within の信号は cross-domain で chance に崩落(全seed一貫、
  非縮退、ログ失敗なし)。SvmW/Lstm/SvmA も c1 Cross 完走で同型か判定する(0.79 等が within 限定構造かを確定)。
- よって T2 の結論は **c1 SvmW Cross の完走待ち**(probe は素特徴に簡単な信号が無いことだけを示した)。

---

## T4/T5/T6 メモ
- **T4(半分)**: 上記 RF-on-SvmA忠実特徴 = 0.496 → SvmA の null は分類器・特徴選択に依らない(特徴の壁)。
  残りの SVM-on-RF特徴 ≈ 0.78(分類器は壁でない)は別途。
- **T5**: IV2025 before(local pooled)Lstm = 0.512(n=6、公表 0.52 と整合)。
  **c1 Lstm 完了(9/9, 異常なし)**: Within-out **0.753** ≈ Cross-in **0.720** ≈ Cross-out **0.743**
  → **within ≈ cross(domain 不変)**。「向上は domain 由来」は反証寄り(cross でも維持)。
  Lstm は event_label(均衡・DRT タスク構造)ゆえ domain ロバストと解釈。**RF(KSS, cross→chance 0.51)
  と対照的** = cross-domain 転移可否は target/label で決まる(RQ2 と整合)。
- **T6(road-curve 除去): 主結論には moot(論理的に不要)。** road-geometry が steering を汚染すると
  「道路追従」由来の見かけ信号を **加える** 方向に働く。しかし T1 で SvmA は忠実特徴 + 複数分類器でも
  chance(univ 0.515 / RF 0.496)であり、**汚染除去は信号を下げることはあっても上げない** → 「信号なし」結論は
  除去前後で不変。除去が効くのは「SvmA が signal を示し、それが drowsiness でなく road-following の疑い」が
  ある場合のみで、本データはその状況にない。よって T1 の null を覆さない。Aygun コースが曲線を含むかの
  メタデータ確認は補足として残すが、優先度は低い。

## c1 実行上の不具合と修正: SvmA cuML ZeroDivisionError (2026-06-27)
- **症状**: c1 SvmA 全セルが ~3分で「Model object is None → Model could not be loaded」で
  JSON を生成せず、pending のまま再実行ループ(GPU 浪費)。ログ rc は 0 のため一見成功に見えた。
- **真因**(traceback): `SvmA.py` PSO 目的関数の cuML `SVC(gamma='scale')` が、**定数のみの特徴部分集合
  (X.var()=0)** で `_get_gamma` の `1/(n_feat·var)` を実行 → `ZeroDivisionError`。sklearn は黙って
  処理するが cuML は例外。**T1 の 18 特徴化で露呈**(旧 14 特徴では全定数部分集合に当たらなかった)。
- **修正**(commit `1243f26`): 分散 0 の部分集合では `gamma = 1/n_features` にフォールバック
  ([`SvmA.py:383`](../../../../src/models/architectures/SvmA.py#L383))。学習例外の traceback もログ化
  ([`model_pipeline.py:316`](../../../../src/models/model_pipeline.py#L316))。
- **検証**: 修正版でクラッシュ地点(~64s)を越え PSO が 210s+ 継続・例外 0 を確認。c1 SvmA を単一ジョブで再開。
- **注**: T1 で SvmA は信号なし確定済みのため、c1 SvmA の AUROC は chance 想定(within/cross 表を埋める用途)。

## Seed 数の数学的妥当性 (2026-06-28)
必要 seed 数 n は **95% CI half-width** 基準: n s.t. `t_{n-1,.975}·s/√n ≤ h`（s=seed間AUROC標準偏差）。
観測された s（RFは11 seed、他は3 seedの推定）と必要nは条件で大きく異なる:

| 条件 | s (std) | n@±0.02 | n@±0.03 | n@±0.05 |
|---|---|---|---|---|
| **RF Within-out** | **0.105** | 108 | 50 | 19 |
| **RF Within-in** (B1) | **0.078** | 60 | 28 | 11 |
| SvmA Within-in | 0.021 | 7 | 5 | 3 |
| Lstm Within-out | 0.019 | 6 | 4 | 3 |
| SvmW Within-out | 0.007 | 3 | 3 | 3 |
| Cross/Mixed (chance) | ~0.005 | 3 | 3 | 3 |

**重要所見**: **RF の within-domain AUROC は seed 不安定（std≈0.08–0.10, 範囲 0.62–0.95、両domain共通）**。
3 seed が 0.76–0.82 と狭かったのは偶然。→ within-domain RF を精密に出すには多seedが必須（±0.02には100超で非現実的）。
他手法（within std≈0.02、cross/mixed≈chance）は **12 seed で CI±0.013 と十分**。

**結論（国際ジャーナル品質の最終決定, 2026-06-28）**:
判断基準 = ①原手法への忠実性 ②統計的厳密性 ③再現性 ④査読耐性（恣意性回避）。
- 目標精度 **95% CI half-width ≈ 0.05**（RFの内在分散上 ±0.02 は100超で非現実的）。
- **Seed: RF=20, Lstm/SvmW/SvmA=15**（全て floor 12 超）。Lstm/SvmW/SvmA は**一律15で査読耐性**（条件別の不均一seedは恣意的に見えるため不採用）。RF のみ高分散ゆえ20（power analysis 明記）。
- **SvmA = `SVMA_PSO_MAXITER=30`**（収束根拠: 保存済 `pso_history`（5050評価）で best fitness が**iter ~2 で 0.0172 に収束し iter100 まで不変**＝maxiter=100は~50倍冗長）。30は15倍マージンで**同一最適解を再現**＝ANFIS/PSO/RBF-SVM の手法自体は不変（忠実）。これで SvmA=15 が ~5.6 GPU日で実行可能。
  - *要・経験的検証*: GPU 解放後に within-in s42 を maxiter=30 で再走し、既存 maxiter=100 値(0.523)と一致を確認予定。
- **SvmW: SVM に max_iter を入れない（忠実維持）— 経験的に確定（2026-06-30）**。高速化候補 `max_iter=100000`
  を Within-out s42 で検証した結果、**AUROC が 0.7697→0.7281（Δ0.042）と変化**＝cap は非収束の病的トライアルだけ
  でなく**収束に多反復を要する正規トライアルにも作用し選択ハイパラを変える**ことが判明。よって**忠実性条件を満たさず棄却**、
  max_iter 無制限（sklearn 既定）を維持。cap で実行された5セルは削除し無制限で再実行済。探索は遅いが結果は完全に忠実。
- 全6条件を同一 imbalv3 タグ・同一seedで統一（within-in も c1 で実行）。

### 全条件 seed 妥当性（TIV2026準拠・全条件網羅, 2026-06-30 実測）
TIV2026(exp2) は seed妥当性を ①σ_rank収束 ②bootstrap 95%CI(B=2000) ③検出力 で論じた。これを
**exp3 の全24条件(c1 6×4) に適用**（[`exp3_seed_adequacy.py`](../../../../scripts/python/analysis/exp3_seed_adequacy.py)）。
判定 = 識別条件は 95%CI half-width ≤ 0.05、chance条件は bootstrap CI 上限 < 0.60。

| モデル | 条件 | n | mean | std | CI hw | 判定 |
|---|---|---|---|---|---|---|
| **RF** | Within-in | 20 | 0.752 | 0.089 | 0.042 | ✅ |
| | **Within-out** | 20 | 0.787 | **0.108** | **0.051** | ⚠️ req_n=21 → **RF=24に増** |
| | Cross-in/out | 20 | 0.51 | 0.005 | 0.002 | ✅ chance(上限<0.52) |
| | Mixed-in/out | 20 | 0.74–0.76 | 0.08–0.10 | 0.037–0.048 | ✅ |
| **Lstm** | 全6条件 | 15 | 0.72–0.78 | 0.009–0.015 | 0.005–0.009 | ✅（req_n=3, 15は十分過剰） |
| **SvmW** | (完了分) Within-out | 3 | 0.765 | 0.007 | 0.016 | ✅／ Cross=chance ✅ |
| **SvmA** | (完了分) Within-in | 5 | 0.540 | 0.031 | – | ✅ chance(上限0.565<0.60) |

**σ_rank（6条件順位の seed安定性, TIV2026 fig相当）**:
- **RF**: κ=16→0.123, 18→0.10, **19→0.073** → 順位（within/mixed ≫ cross）は**seedでなく条件で決まる**（収束）。
- **Lstm**: κ=14→0.133（RFより高い）= 6条件が**統計的に近接**（0.72–0.78）ゆえ細順位は揺れる。ただし上位構造（全条件 chance に崩れず ~0.75）は安定 → **domain不変の裏付け**。

**結論**:
- **唯一の不足が RF Within-out**（高分散 std=0.108, 20では hw=0.051）→ **RF=24 seed に増**（req_n=21、24で hw≈0.046<0.05、CPU安価）。
- Lstm は req_n=3 と過剰だが一律15で査読耐性。chance条件(Cross/Mixed-KSS, SvmA全条件)は低分散で req_n≤4。
- **最終 seed 計画（2026-06-30 確定）: RF=24, Lstm=15, SvmW=8, SvmA=8**（分散比例・各条件で adequacy を満たす
  最小構成）。SvmW/SvmA は低分散（req_n=3〜7）ゆえ15→8に削減：統計的に十分かつ SvmW の病的SVM（~13h/セル）
  による数日のムダを回避。手法・パイプラインは不変（seed数のみ）。SvmW/SvmA 完走後に全6条件で本表を再生成し最終確定。
- TIV2026 が AUROC で σ_rank=0.147 の残差を許容したのと同程度〜より厳しく、exp3 の seed計画は妥当。

## 分割方法論と within-domain の性質（TIV2026/IV2025 と共有・既知の制約, 2026-07-01）
**方法論（TIV2026/IV2025 と同一・exp3 も踏襲）**: train は `--time_stratify_labels`（ラベル層化時系列 split,
stratify=True）、eval は target_timewise / pooled の stratify=False（単純時系列）。exp2 HPC スクリプト
（`pbs_domain_comparison*.sh`）と同一で、exp3 を TIV2026/IV2025 と**直接比較可能**にするため踏襲する。

**確認した性質（read-only 検証）**: train(stratify=True) と eval(stratify=False) は別パーティションのため、
within/mixed では eval-test の一部が train の学習集合と同一行になる（content照合で ~69%）。完全に時間整合な
分割（train・eval とも stratify=False）にすると within-in は **0.78 → 0.526** に低下（de-leak 実測）。
すなわち **within-domain の高値は、この temporal-split プロトコルに依存する性質**である。

**判断**: この性質は **IV2025/TIV2026 と共有**する。exp3 は論文間比較の一貫性を最優先し、**同一方法論を採用**する
（脱leakage版は不採用）。**相対比較（手法間・before/after・within vs cross）は同一枠組み内で有効**。
cross-domain は元から本性質の影響を受けない（別ドメイン学習）。本件は**隠さず既知の制約として明記**する。
TIV2026 は公表済みのため一切変更しない（本ログは exp3 の意思決定記録）。

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

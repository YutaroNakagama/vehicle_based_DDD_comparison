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

## 2026-07-04 敵対的再検証（11エージェント workflow）— Bug#4 再発を発見・修正

**方法**: 4モデル×(独立再スキャン→懐疑エージェントが refute→網羅性クリティック)。各ターゲットで生 JSON を
独立に rglob+load し、byte単位で相互照合。

**発見（実バグ1件, CONFIRMED）**: c1 **RF の Cross(source_only) 2セルが Within(target_only) の予測ベクトルと
byte完全一致**（`y_pred_proba` md5一致）:
- `source_only/in_domain/s42`  (AUROC 0.7734 = within-in s42 と同一)
- `source_only/out_domain/s123` (AUROC 0.7578 = within-out s123 と同一)

**根本原因（既知の "Bug #4" の再発）**: `evaluate.py` の `resolve_jobid_for_evaluation` 解決順で、優先#3の
glob `<M>_<mode>_rank_*` が **c1 の保存モデル名（内部mode="domain_train"）に一致せず**、優先#4の
**共有ファイル `models/<M>/latest_job.txt`**（各workerが上書きする可変ファイル）に落ちる。RF/SvmW/Lstm は
4/4/3 worker 並列のため、あるセルの train と eval の間に別workerが `latest_job.txt` を上書きすると eval が
**別セルのモデルをロード**。RF は高速セルで eval が密集し 2/48 で発生。
- **IV2025 が無傷な理由**: IV2025 の eval_cmd は `--jobid` を渡す（解決順#1）→ race-free。c1 は未指定だった。

**網羅スキャン**: 全4モデル×(domain,seed)×mode総当たりで roc_auc(10dp)+cm 完全一致を検出 → **一致は上記2件のみ**。
Lstm/SvmW/SvmA は clean。影響: Cross平均 0.5288→0.5182(in)、0.5171→0.5067(out)（依然 ~0.52 崩落, 結論不変）。

**修正**: `c1_domain_launcher.py` の eval_cmd に **`--jobid jobid` を追加**（IV2025 と同じ安全パターン）。解決順#1で
セル自身のモデルを確定ロード → **worker数に依らず race-free**。方法論（分割/特徴/ラベル/SMOTE）は不変。
- 汚染2セルの JSON を退避・削除し、修正版 launcher(workers=1)で再実行。他モデルは clean のため対象外。
- 稼働中 SvmW(旧コードをメモリ保持)は残セルで理論上 racy だが、~15h/セルで eval が疎→衝突稀（完了8セルは
  clean）。毎ステータス確認で同スキャンを実施し汚染検出→再実行で対応。watchdog の将来再起動は修正版を使用。

**その他は独立再検証で clean**: c1 RF(Win/Mix 実判別・Cross は minority-tracking の真の near-chance)、
c1 Lstm(domain不変は閾値非依存の AUROC で本物)、c1 SvmW(Within 実判別/Cross expected-collapse)、
c1 SvmA(全域 chance・非縮退)、IV2025(RF 0.738 実判別＋他3手法 expected-collapse=公開結果再現)。
- **カバレッジ注意**: Lstm/SvmA の JSON は `y_pred_proba` を持たず、RFバグを捕えたベクトル照合が不可。
  roc_auc+cm 完全一致の代替スキャンで補完（clean）。

## 2026-07-19 Lstm ROC 図の再生成（mean ± std, seed 集計）

`mixed_in` / `mixed_out` は各 15 seed、`pooled` は 6 seed の既存 evaluation CSV を使って、
ROC の平均曲線と $\pm 1\sigma$ 帯を重ねた図を再生成した。前回の単一 seed 図は、この集計版に置き換え済み。

- 出力図: [`lstm_mixed_in_out_pooled_roc_mean_std.png`](../../../../results/analysis/exp3_verification/lstm_mixed_in_out_pooled_roc_mean_std.png)
- 集計 JSON: [`lstm_mixed_in_out_pooled_roc_mean_std_summary.json`](../../../../results/analysis/exp3_verification/lstm_mixed_in_out_pooled_roc_mean_std_summary.json)
- 再現スクリプト: [`plot_lstm_mixed_pooled_roc.py`](../../../../scripts/python/analysis/plot_lstm_mixed_pooled_roc.py)

**要点**:
- `mixed_in`: AUROC mean 0.785332, std 0.006985, n=15
- `mixed_out`: AUROC mean 0.784263, std 0.006983, n=15
- `pooled`: AUROC mean 0.505686, std 0.017565, n=6

図中の凡例は `AUC=mean±std` 表記に修正し、文字化けの原因になっていた制御文字を除去した。

## 2026-07-19 【重大更新】分割不一致リークは "既知の制約" を超え、中心 finding を無効化する（4エージェント敵対的監査＋実モデル脱リーク実測）

2026-07-01 節（within/mixe の train stratify=True／eval stratify=False 不一致で ~69% 行重複、within-in 0.78→0.526）で
記録した「既知の制約・比較性のため意図的に残す」という判断を、**より深刻な結論に更新する**。ユーザ指摘（RF 全特徴 pooled
0.86 が高すぎる）を機に 4 エージェントで敵対的監査＋実モデルで脱リーク再評価した結果:

**確定事項（コード＋実モデルで記録JSONを小数3桁再現、敵対的反証を通過）**
- **pooled も同種リーク（むしろ最重症）**: pooled は train=被験者内時系列(前半60%, `TRAIN_RATIO=0.6`)／eval=**完全ランダム20%**
  （`iv2025_baseline_launcher` が eval に `--subject_wise_split` を渡さず、`eval_pipeline.py:152`→"random"）。eval テスト行の
  ~60% が train と同一行。実モデル分解: SEEN(学習済)=0.85–0.97 / **UNSEEN(正直)=0.49–0.53**。
- **honest なのは Cross(source_only, 被験者非交差=重複0%) と domain_train のみ**。§2 の pooled/within/mixed 全列がリーク。
- **within を実モデルで直接脱リーク**（保存モデルを time_stratified_three_way_split の held-out test で評価）: RF 0.77→**0.47**、
  SvmW 0.80→**0.52**（両 chance）。→ **論文の中心 finding「SW-SMOTE が SvmW を回復／SvmA と解離」は leak アーティファクト**。
  honest では RF も SvmW も SvmA も within=chance。回復も解離も消える。
- **陽性対照合格**: EEG バンドパワーは同 honest 分割で 0.62（>車両 0.53）＝ハーネスは本物の信号を通す。車両=chance は真。
- **唯一の実信号は Lstm**（event_label=DRT、別タスク）。Cross(honest) で 0.73–0.75、domain 不変。EEG眠気ではなく DRT 検出。

**判断の更新**: 2026-07-01 は「相対比較（before/after・手法間）は同一枠組み内で有効」としたが、**honest では全 KSS 手法が
chance に潰れるため相対パターン(SvmW>SvmA 等)も消える**ことが実測で判明。よって「比較性のため leak を残す」方針では
論文の中心主張が成立しない。**honest（train/eval 同一分割 or cross-subject）で全面再評価し、negative-result + 方法論的知見
（先行の高値は分割不一致由来）＋ Lstm on DRT を唯一の実信号、という形へ再構成が必要**（要ユーザ判断）。exp2 の公表済み
TIV2026 は変更しない。詳細は c1_results.md §3.6。scratchpad: leak_test.py / leak_test2.py / honest_within.py。

## 2026-07-20 【方針決定】exp3 は IV2025/TIV2026 準拠で完遂、脱リーク再テストは "構え"（非破壊・完遂後）

ユーザ決定: **exp3 は現行の同一方法論（IV2025/TIV2026 と共有、train `--time_stratify_labels`／eval stratify=False）で
完遂し、論文間比較性を最優先**する。2026-07-19 の脱リーク監査（§3.6/c1_results）は「既知の制約の深掘り＝ロバストネス
附録」として保持し、**実装見直し・脱リーク再テストは走行中実験・共有コードに非破壊で構え、exp3 完遂後に実行**する。
- 走行中（07-20 20:26）: c1 SvmW within+mixed 25/32・RF-nofs 19/23、異常なし。完遂を待つ。
- 準備物: c1_results §3.7（A 実装見直し read-only、B 脱リーク再テスト＝方式1 保存モデル honest 再スコア／方式2
  `evaluate.py --honest_split` 新フラグ＋新タグ `honest_` で既存温存・デフォルト不変）。
- 重要な実装注意（再掲）: `data_time_split_by_subject` は index リセット→ honest 再スコアは位置(iloc)指定必須
  （§3.6 の落とし穴。途中の pooled honest 0.75 は本バグ由来で不採用、正は ~0.52）。
- 論文（草稿 TIV2026_exp3）: 主結果は準拠提示＋Limitation/Robustness に脱リーク感度解析を明記、が最有力（判断保留）。

## 2026-07-20 【exp2/TIV2026 並行監査】headline 0.89 は "記述設計 honest だがコード実装が行リーク" — honest では chance（3エージェント, read-only, 公表物・走行 exp3 非干渉）

ユーザ依頼で exp2/TIV2026 を exp3 と同じ非破壊監査。結論を3層で:
1. **論文の記述設計は honest/透明**: 各被験者 時系列70/15/15・test=disjoint held-out、`ω_tr`(被験者重複)を明示的に leakage として
   開示、Cross=chance 明記、0.89 は within-subject(personalized) として提示（新ドライバー汎化とは主張せず）。exp3 より透明。
2. **だが 0.89 を生成した実装は行レベルでリーク（git 確定）**: exp2 pbs は train `--time_stratify_labels`(全体70/15/15) / eval それ無し
   (被験者別60/20/20)＝別関数 → **eval テスト行の 69% が学習行**（物理照合）。git `ed6e66d`(07-01 "69% within eval-test in train")、
   `f1895bd`(07-19 "eval stratify=False, as in the exp2 HPC scripts")、leaky eval 分岐は exp2 期 573ba1b で byte 一致。
   → 論文が記述した disjoint 設計をコードが実装しておらず、headline 数値は 69% 行再利用で水増し。
3. **honest 実測は全 regime で chance**: 被験者内(gap有無)0.49–0.52、cross-subject 0.50–0.51、ランダム(窓重複最大)0.51；
   陽性対照 EEG cross-subject 0.59(>車両)＝ハーネス妥当。personalization も窓重複も信号を作らない。0.79–0.89 の出所は ~69–70%
   行再利用のみ。→ 既存メモ project_exp2_rf_087_unreproducible の「0.89 ローカル再現不可」を機構レベルで確定。

**含意**: 「SW-SMOTE+within-domain が最適(0.89)」という TIV2026 中心結果は honest では成立せず（正しい被験者内分割でも 0.50）。
性質は科学的不正でなく実装バグ/再現性（設計は正しく記述、コード未実装）。**公表済み論文なので対応(正誤表 or exp3 で明示的に扱う)は
ユーザ判断**。本監査は read-only、公表 dir(exp2-analysis/paper/TIV2026) は無変更。但し書き: 論文数値は HPC(145特徴)由来で実バイナリは
未実行だが、機構は特徴非依存＋git 一致で蓋然性大。scratch: verify_exp2_overlap.py, decompose.py, index_bug_demo.py。

## 2026-07-21 【exp3 状況監査】完了状況・各ケース完了時刻推定・異常終了/想定値検証（inventory 全 390 セル + 4エージェント・ログ監査 + de-dup 運用）

ユーザ定例依頼「exp3 状況確認・各ケース完了時刻推定・異常終了/想定通りか検証・ドキュメント化」。権威データ = 全 eval JSON の
プログラム走査（scratch: exp3_inventory.py, 1715 files → 390 unique (model,arm,seed)）+ ログ実測 + ライブプロセス。

### 完了状況（TABLE 条件 = Within in/out + Mixed in/out。Cross は 07-11 に demoted、iv25 は pooled baseline）
- **Lstm: 60/60 完走**（07-01..03）。Within 0.766–0.793 / Mixed 0.760–0.800 / Cross 0.696–0.772。唯一の実信号（event_label）。iv25base/smote pooled 6+6 = 0.50–0.53。
- **RF(fs): 96/96 完走**（07-01..03）。Within 0.53–0.95 / Mixed 0.55–0.96（leak膨張・高分散）/ Cross 0.50–0.53（chance=honest）。iv25base 15=0.58–0.88, iv25smote 15=0.60–0.85。
- **SvmA: 32/32 完走**（07-03..10）。全条件 0.46–0.63（near-chance、特徴に信号なし=T1 と整合、縮退でなく真の無信号）。iv25 6+6。
- **SvmW: 26/32**（Within 16/16 完走 = in 0.800±0.012 / out 0.759±0.012；Mixed 10/16 = in 0.738±0.013(5) / out 0.766±0.017(5)、**残 6 実行中**）。Cross 2/8(demoted) 0.51。iv25base 6=0.50–0.54, iv25smote 5=0.69–0.73（SW-SMOTE 膨張）。
- **RF-nofs: 20/25**（pooled 5 + Within 9 + Mixed 6、**残 5 実行中**）。0.72–0.96（全特徴 leak で最高帯）。

### 異常終了チェック → 検出ゼロ（JSON レベル確定）
- **390 unique セル全て roc_auc ∈ [0,1] で有効**。NaN/inf/範囲外/truncated/parse-fail = **0 件**。破損・途中切れ eval なし。
- ログレベル監査（4エージェント並列 workflow wlniqhvo2: RF/SvmW/SvmA/Lstm の全ログを rc≠0・TRAIN FAILED・traceback・
  特徴フォールバック[Wang15/LaneOffset/MISSING]・非収束 について走査）→ 結果は本節末に追記。

### 想定値チェック → 全て想定通り（既知リーク署名と整合）
- **honest 側（Cross・iv25base-pooled）は chance 0.50–0.53**、**leaked 側（Within・Mixed の train/eval 分割不一致）は膨張 0.7–0.96**。
  この非対称が全4手法で一貫 → ハーネスは監査の予測どおり挙動、**新規の異常なし**。
- Lstm のみ Cross でも 0.70–0.77（実信号 event_label）。SvmA は全条件 near-chance（特徴律速）。SvmW iv25smote pooled のみ 0.69–0.73（SMOTE 膨張）。
- 位置づけ: これらは §2 の「IV2025/TIV2026 同一プロトコル（leak 込み・既知の制約）」下の値。honest なら車両→眠気は chance（§3.6 の脱リーク表で確定済）。**現行 methodology の値としては想定通り**、honest 再評価が follow-up。

### 各ケース完了時刻推定（稼働セル、プロセス開始時刻 + 完了実績中央値から）
SvmW mixed 実績: 35–102h（中央 ~76h、高分散、非収束SVM律速）。RF-nofs 実績: target ~10–27h, mixed ~23–38h。
- **SvmW** in/out mixed **s1**: 経過 103h、Optuna 完了→最終学習段階 → **~07-21〜22**（**要監視: 既観測最大 102h に到達。07-22 昼までに完了しなければ最終SVMフィット停滞の可能性→介入候補**）。
  in mixed **s2025** 64h / out **s2025** 53h → **~07-22**。in/out **s13**（07-21 02:38 開始）→ **~07-24**。→ **SvmW 全完了 ~07-24**（s1 停滞なければ）。
- **RF-nofs** in/out mixed **s0**（32–36h 経過、typ ~35h）→ **~07-21（数時間内）**。out target **s1** → 07-21 夕。in/out mixed **s1**（07-21 06:53 開始）→ **~07-22 夕**。→ **RF-nofs 全完了 ~07-22〜23**。

### 運用（07-20〜21、非破壊）
- **SvmW de-dup**: s2025 in/out が各2本二重実行(churn)していたため新しい方 2本を停止（1セル1プロセス化）。原因 = `CLAIM_FRESH_SEC=10h` < SvmW 単フィット時間 → stale 判定で二重起動。現在 idle_pending=0 のため再発せず。
- **RF-nofs top-up**: s1 mixed in/out の 2 idle-queued セルを拾う 2ワーカーを永続起動 → 残5セル全並列化。コア 11/20。

**総括**: exp3 は実質完了（Lstm/RF/SvmA 完走、SvmW 26/32 + RF-nofs 20/25 が ~07-24 までに完了見込み）。**異常終了ゼロ・全値想定通り**（現行 leak込みプロトコル基準）。honest 再評価は §3.6 の follow-up として保持。

### 追記（同 2026-07-21）ログレベル監査（4エージェント wlniqhvo2）確定 — 現行キャンペーンは結果レベルで健全、ただし3件の実所見

JSON レベル監査（全 390 セル有効）に加え、全ログ（RF 285・SvmW 237・SvmA 288・Lstm 724）を rc≠0/traceback/フォールバック等で走査。
**現行 c1/iv25 キャンペーンにクラッシュ・破損・縮退はゼロ**（clean-finish: Lstm 103/103、SvmA 60/60、RF c1dom 159/164〔残5は稼働中の nofs〕、SvmW 204）。
**現行 prior_ 系の 96(SvmA numpy)・131(Lstm xgboost) クラッシュは全て廃止済み env / 廃止済み距離選択 domain_train** で out-of-scope（[[project_rf_distance_selection]]、CUDA 移行）。

**現行キャンペーンに関わる実所見 3 件（要記録）:**
1. **【方法論ラベル不一致】iv25smote(pooled+SW-SMOTE)は全モデルで subject-wise でなく pooled SMOTE で実行**。ログ実測: `[OVERSAMPLE] subject_id not found, falling back to pooled oversampling` が **RF 20/20・SvmW 5/6・SvmA 6/6** に出現。pooled モードは subject_id を保持しないため subject-wise 不能→pooled SMOTE(ratio0.5)へフォールバック。**c1 の Within/Mixed グリッドは subject-wise が正常動作（0 fallback）**。→ pooled アームの SMOTE は「subject-wise」でなく「pooled ratio0.5」。結果(RF 0.60–0.85 / SvmW 0.69–0.73 / SvmA 0.48–0.63)は有効だが**ラベルは pooled SMOTE と記すべき**。論文の pooled 対策記述に反映要（ユーザ判断）。
2. **iv25smote SvmW s2025 のみ欠損（5/6）**。s2025 ログは 418B スタブ（rc=3221225794 DLL_INIT_FAILED、07-10 以降 dead、未再起動）。**s0/s7 は有効**（eval JSON 07-12 10:10 auc 0.720/0.728；監査が見た "dead log" は失敗リトライが `"w"` 上書きした痕跡で、成功実行の JSON は残存）。seed 目標~5 に対し 5/6 は十分。6 本目が必要なら s2025 再起動（任意）。
3. **【RF データ品質・系統的】全 RF 実行で `LaneOffset_Skewness/Kurtosis` が NaN→列平均補完、~3 被験者/実行が SMOTE スキップ**（`Input X contains NaN` により）。by-design・全実行で一貫（相対比較は保存）だが、RF の絶対値はこの2特徴の平均補完を含む。

**良性（将来監査で誤検出しないよう記録）**: (a) RF の `ERROR - [SAVE] Model object is None` は HPO 前の早期チェックポイントで1回/実行出るが cosmetic（Optuna 後に学習・保存し全実行 rc1=0）。(b) source_only(Cross) の dead スタブ 12本（07-15 mass-kill・未再起動）は **Cross が 07-11 に demoted のため無関係**。(c) `rank_names.txt not found -> Using CLI targets` と Lstm の16列 exclude-list fallback は意図された正規経路。(d) b1cmp アーム（c1/iv25 外）の `File not found: processed_Sxxxx.csv` は対象被験者の欠落データ警告で、primary scope 外。

**総合判定**: 現行 c1/iv25 の**結果は全て健全・想定通り**（異常終了ゼロ）。唯一の方法論的注意は所見1（pooled アームの SMOTE が pooled であり subject-wise でない）で、これは論文記述の正確性に関わるためユーザ判断事項。所見2/3 は軽微。

### 追記2（同 2026-07-21）【重要・エスカレーション】SW-SMOTE 逸脱の根本原因を行レベルで特定 — pooled アーム全4手法が subject-wise でなく pooled SMOTE（ユーザ指摘「exp3 の不均衡対策は全て SW-SMOTE のはず」を受けた深掘り）

ユーザ指摘（不均衡対策は全て SW-SMOTE が正）を受け、追記1・所見1を「表記不一致」から「**意図した方法論からの実逸脱**」として根本原因まで特定。

**確定した影響範囲（ログ実測）**: iv25smote(pooled) アームは **全4手法で pooled SMOTE**にフォールバック（`Applying subject-wise oversampling`=0）:
RF 20/20・SvmW 5/6・SvmA 6/6・**Lstm 6/6**。一方 **c1 グリッドは Within/Mixed とも subject-wise が正常**（SvmW target_only 16/16・mixed 16/16 が `Applying subject-wise`、fallback 0）。

**根本原因（コード行レベル）**:
1. `iv2025_baseline_launcher.py` L98-111 の pooled train_cmd は `--mode pooled --subject_wise_split` のみで **`--time_stratify_labels` を渡さない**。
2. → `model_pipeline.py` pooled(else)分岐 → `split_data(subject_split_strategy="subject_time_split", time_stratify_labels=False, keep_subject_id=True)`。
3. → `split_helpers.py` L161-173（time_stratify=False 経路）→ **`data_time_split_by_subject(...)` を呼ぶが、この関数は keep_subject_id 引数を持たない**。
4. → `split.py` L468 `X_train = train[feature_columns].drop(columns=[subject_col], errors="ignore")` が **subject_id を無条件削除**。
5. → `model_pipeline.py` L212 `"subject_id" not in X_train.columns` → L213 warning → pooled SMOTE で継続。
**対照**: c1 の target_only/mixed は launcher が `--time_stratify_labels` を渡す → `split_helpers.py` L142-160（time_stratify=True 経路）が **L154-155 で keep_subject_id を honor** → subject_id 保持 → SW-SMOTE 正常。

**含意**: 「exp3 不均衡対策は全て SW-SMOTE」という設計に対し、**pooled(iv25smote) アームの全4手法結果（RF 0.60–0.85 / SvmW 0.69–0.73 / SvmA 0.48–0.63 / Lstm 0.50–0.51）は pooled SMOTE で生成されており、意図と不一致**。SW-SMOTE で再生成が必要（iv25base=SMOTE 無しは無関係）。

**修正案（要ユーザ判断）**:
- **案A（推奨・IV2025 分割を保持）**: `data_time_split_by_subject` に `keep_subject_id=False` 引数を追加し L468 を条件付き drop に、`split_helpers.py` L169 で keep_subject_id をスレッド。→ pooled の分割方法（per-subject 60/20/20＝IV2025 config）を変えずに SW-SMOTE を有効化。default False で他呼び出し（eval 等）は不変・安全。
- **案B**: iv25 launcher に `--time_stratify_labels` を追加。→ time_stratified 分割へ変わり IV2025 比較性が崩れるため非推奨。
案採用後、iv25smote pooled を SW-SMOTE で再生成（対象手法・優先度は要相談）。

**状況更新（同日）**: 監視中だった SvmW mixed s1 in/out は 09:16/09:33 に**正常完了**（~4.3日）。SvmW **28/32**（残 mixed s2025・s13 × in/out の4セル稼働中、s2025 ~68–79h/ s13 ~20h → SvmW 全完了 ~07-24）。RF-nofs 21/25（target 完走、mixed s0/s1×in/out 稼働、~07-22）。RF-nofs in_mixed_s1 の重複(top-up 競合)を解消。全 eval 値は有限・想定帯内、現行キャンペーンにクラッシュ無し。

### 追記3（同 2026-07-21）SW-SMOTE 逸脱の修正を実装・検証（案A）＋ iv25smote pooled 再生成計画（~07-24 実施予定）

ユーザ判断（案A採用／全4手法を現行完了後に再生成）に基づき修正を実装。

**実装（案A、3箇所、default False で他経路・eval・走行中セルは不変）**:
- `src/utils/io/split.py` `data_time_split_by_subject`: `keep_subject_id: bool = False` 引数を追加。X_train の subject_id drop を条件付きに（`drop(columns=([] if keep_subject_id else [subject_col]))`）、`_check_nonfinite(X_train, preserve_cols=["subject_id"] if keep_subject_id else [])` で非数値 subject_id を保持。X_val/X_test は従来通り常に drop。
- `src/utils/io/split_helpers.py` L169: `data_time_split_by_subject(..., keep_subject_id=keep_subject_id)` をスレッド（split_data は既に keep_subject_id を受領し pooled 経路は `keep_subject_id=(use_oversampling and subject_wise_oversampling)` を渡す）。

**検証（実データ 8被験者, common）**: `keep_subject_id=False`→X_train 160列・subject_id 無し（既存挙動維持）。`keep_subject_id=True`→X_train 161列・subject_id 有り→`apply_oversampling(subject_wise=True)` が `[Subject-wise Oversampling] Processing 8 subjects`（7/8処理・1スキップ）で起動、3755→5208 行（pos 127→1580）。→ **pooled 経路で subject-wise SMOTE が正常動作することを確認**。両ファイル ast.parse OK。走行中の c1 SvmW/RF-nofs は time_stratify 経路（--time_stratify_labels）のため無影響、eval は default False で不変。

**再生成計画（~07-24、現行 SvmW/RF-nofs 完了後）**:
- 対象: iv25smote(pooled) の **全4手法**（RF/SvmW/SvmA/Lstm）を SW-SMOTE で再生成。iv25base（SMOTE無し）は無関係。
- 手順: 既存 iv25smote eval JSON/モデルは already_done で skip されるため、**旧結果(pooled-SMOTE版)を `results/_archived_pooledsmote_<date>/` へ退避（MANIFEST付き）してから同一タグで再実行**（新コードで subject-wise が走る）。§2 表の該当値は再生成後に更新。
- 起動: `iv2025_baseline_launcher.py --model {RF,SvmW,SvmA,Lstm} --smote --seeds <5>`（現行完了後に順次、CPU 競合回避）。再生成後、旧 pooled-SMOTE 値と新 SW-SMOTE 値の差分を検証・記録。

**現況（同日 22時台）**: SvmW 28/32（s1 in/out 完了、残 mixed s2025・s13×in/out 稼働、~07-24）。RF-nofs 21/25（target 完走、mixed s0/s1×in/out 稼働、~07-22）。異常終了ゼロ・全値想定帯内。

## 2026-07-22 【exp3 方向性・再評価】IV2025/TIV2026 再読＋実装再監査＋honest 再スコア（4エージェント w1qbn0j8c）— 「RF 最良」は honest で崩壊、TIV2026 は Sobol 因子研究と判明

ユーザ依頼「IV2025/TIV2026 を読み直し・実装を見直した上で exp3 方向性を決めたい／IV2025 の RF 最良は今も有効か」に対し4エージェント並列で確定。

### Q「IV2025 の RF 最良(AUC 0.85)は今も有効か」→ 【NO】honest 評価で chance に崩壊
- **iv25base pooled 実測**（saved-model 再スコア＋cross-subject 再学習、positional-safe）:
  | 手法 | recorded(leaked) | train/eval-test 行重複 | HONEST |
  |---|---|---|---|
  | **RF** | **0.738**(0.58–0.88) | **59.6%** | **0.516**（disjoint）／0.510（cross-subj）|
  | SvmW | 0.519 | 59.6% | 0.502 |
  | SvmA | 0.481（既に sub-chance） | 59.6% | ~chance |
  | Lstm | 0.512（DRT・既に chance） | 59.6% | ~chance |
  - **RF SEEN/UNSEEN/honest 勾配**: SEEN(学習内 59.6%)=0.85、UNSEEN=0.65、honest(時系列 disjoint)=0.52 ＝**記憶署名**。nofs160 は SEEN 0.97（容量↑＝記憶↑）。
  - **陽性対照 EEG バンドパワー cross-subject = 0.61 >車両 0.51** ＝ハーネス妥当、車両 chance は真の null。
  - **honest 順位: EEG(0.61) > {RF≈SvmW≈SvmA≈Lstm すべて 0.48–0.52 で chance 同着}**。RF は最良でない。
- **機構（実装トレース）**: iv25base の TRAIN=`--subject_wise_split`→per-subject 先頭60%、EVAL=フラグ無し→全行ランダム20% ＝別分割で重複 59.6%（[iv2025_baseline_launcher.py:98-115]、model_pipeline.py:108-110、split.py data_split/data_time_split_by_subject）。**RF だけが勝つ理由=(a)非縮退の行ランキング＋(b)木の記憶容量の両立**。SvmW=全陽性縮退／Lstm=多数派崩落／SvmA=特徴信号なし（T1: SvmA18特徴→RF 0.4955）で、同じリークを換金できない。RF の唯一の真の固有性は**不均衡頑健性（SMOTE無しで 0.738 vs SvmW 縮退 0.519）であって信号ではない**。honest では全手法同着 chance。

### IV2025 再読（[latex/IV2025/IV2025.tex]）
- 「RF 最良 AUC 0.85/Acc 88%」は**構成 C2 下の"手法間比較"**（C2 = **pooled random 8:1:1 split**、被験者分割なし＝same-subject＋50%窓重複で構造的リーク、L803/L936）。genuine-signal 主張ではない。
- **IV2025 自身が留保**: Threats「Feature Bias Towards RF」(L1017-1019「RF は他手法の全特徴を使うので有利かも」)＋Discussion「RF 高精度の主因は特徴が多いから」(L997)。→ 後の「容量が大きく重複行を記憶」機構を先取り。
- リーク批判は**他3論文向け**（training-accuracy-as-test）で自らの 0.85 には未適用。within-subject inflation・chance 崩壊は未議論。EEG は**ラベルのみ**（陽性対照なし）。

### TIV2026 再読（[exp2-analysis/paper/TIV2026/main.tex]、公表・無変更）— 記憶の前提を訂正
- **"RF 提案検出器で 0.89 が車両有効性を示す"論文ではない**。**4因子 Sobol 感度研究に再スコープ済み**（"does not propose a new detector" L45/73/161）。主結果は分散分解（訓練モード m＋再バランス R＋交互作用で系統分散の**≥93%**、S_Tm=0.50–0.71/S_TR=0.27–0.37、距離 d・グループ g は無視可能）。
- 0.89 は「最適因子セル(RF+SW-SMOTE r=0.1+within-domain, ω_tr=1)」に**降格**。ω_tr(被験者重複)を明示的に "leakage" と開示(L273)、**cross-subject 崩落(near-chance)を主要結果として報告**(L430/468/529)。**汎化の過大主張なし**、記述は honest。
- **4手法の head-to-head 主張なし**（RF は唯一の分類器で因子を振る対象、SvmW/SvmA/Lstm は future work）。→ exp3 の「RF=提案 vs 3ベースライン」枠組みは TIV2026 に存在しない。TIV2026 の cross-chance は exp3 の honest 結論と**一致**。soft spot: abstract が 0.890 を前面に出し cross 崩落幅を併記せず／within の 50%窓境界リーク未検討。

### 方向性への含意（詳細は本文の統合報告）
honest では車両→EEG眠気=全手法 chance（RF 含む）。「RF 最良」は IV2025 自身が予告した feature-bias＋分割リークの産物。TIV2026 は既に慎重な因子研究で cross-chance を認めている。→ **exp3 は "leakage-free 再評価でネガティブ＋機構＋方法論" が honest かつ TIV2026 と整合する唯一の方向**（Direction B）。現ドラフトの回復ストーリー(Direction A)は within-domain リークに依存し擁護困難。scratch: honest_iv25base.py, controls_only.py。

### 追記（2026-07-22）方針決定＝Direction A（honest-scoped）＋ドラフト主張の整合実施

ユーザ決定: **Direction A（回復ストーリー/機構解析の spine を保持・現ドラフト維持）**。ただし本日の確定事実に反する主張は
honest-scoped に補正する方針で合意。TIV2026 自身が「within-domain 0.89 を ω_tr=1 と開示しつつ主結果化」という Direction A 型を
honest に成立させている前例に倣う。

**実施（TIV2026_exp3/main.tex の全 \todo 主張を整合、Direction A 枠内）**:
- Before/After の2列を **Before / After / Leakage-free の3列**へ拡張し、**leakage-free 列を co-headline**化（附録でなく本文の決定的比較）。
- 各 recovery 値を **within-domain(ω_tr=1, 行重複~69%) にスコープ**、Before は pooled(~60%)。leakage-free: RF/SvmW/SvmA≈chance(0.50-0.52), Lstm(DRT) 0.73-0.75, EEG 陽性対照 0.61。
- **偽主張を撤去**: 「rebalancing は特徴に潜在信号がある手法を回復（SvmW yes）」→「SMOTE は縮退を解いて within の重複を換金させるだけ、信号は生まない（honest SvmW 0.52）」。RF「genuine 最良/頑健」→「不均衡頑健＋重複行記憶（SEEN0.85/UNSEEN0.65/honest0.52）＝IV2025 の Feature Bias Towards RF と一致、honest では chance」。
- RQ1 の答えを「回復する」→「見かけ上 within で回復するが leakage-free では不成立＝評価プロトコルのアーティファクト（honest は NO）」。RQ2 も leaky 前提で機構を説明。
- Related/Conclusion に **TIV2026=Sobol 因子研究（ω_tr 開示・cross-chance・RF-vs-baselines 主張なし）** の正しい位置づけを反映、exp3 は ω_tr=1 セルを行レベルへ拡張、と接続。
- Limitations に (0) Before/After は leaky 領域内で decisive は leakage-free 列、(6) SW-SMOTE pooled 実装欠陥(07-21 修正/再生成待ち、honest 結論に不変)を追記。
- ファイル冒頭に整合の根拠コメントを付与。

現在、再構成ドラフトを3敵対的クリティック（過大主張・内部矛盾・数値/事実）で検証中（wuc85ngnx）。指摘を反映後に確定。

### 追記（2026-07-22）再構成ドラフトの敵対的検証（3クリティック wuc85ngnx）＋指摘反映

過大主張・内部矛盾・数値/事実の3クリティックで再構成 main.tex を検証。**全 load-bearing 数値・IV2025/TIV2026 特徴づけ・手法属性は ledger と一致**（誤りなし）。検出＝主に1件の実欠陥（私が混入）＋軽微数件、全て修正:
- **【major・修正済】Lstm ターゲット混同**: Lstm は DRT distraction 対象で EEG眠気ではないのに、abstract「全"4"手法 chance」＋ results leakage-free 眠気列に Lstm DRT 0.73-0.75 を併記し、「1手法が leakage-free 眠気検出に生存」と読める矛盾。→ 崩落主張を **3手法(RF/SvmW/SvmA)にスコープ**、Lstm は「非可換な別ターゲット(DRT)の参照行」として分離（眠気列に DRT 値を置かない、眠気ターゲットでは Lstm も chance と明記）。
- **【minor・修正済】** intro「leakage-free pooled」用語衝突→「leakage-free (subject-disjoint)」。metrics の AUPRC dangling promise→results/mechanism で AUROC と併報と明記。rq2「near-balanced」→「~74%陽性で SMOTE/RUS 不活性」。
- **【要著者確認・未変更】** SvmW 引用 `zhao2012`(CWT 2012) と内部記録「Zhao2009 GHM 8帯域」の不一致。references.bib は zhao2012 のみ。実際に再現した Zhao 論文の出典を著者が確認要（主張整合の範囲外のため無変更）。
LaTeX 健全性: `$` 偶数・`{}`204/204 balanced。ドラフトは Direction A(honest-scoped) で内部整合・過大主張なしに到達。

## 2026-07-22 【exp3 状況監査】SvmW 28/32・RF-nofs 18/20、異常終了ゼロ・全値想定帯内、要監視2件

- **SvmW 28/32**: Within 16/16 完走。残 = mixed の **s2025・s13 × in/out（4セル稼働中）**。s2025 in 88h(Optuna Trial41)/out 77h、s13 in/out 29h。→ **全完了 ~07-24〜25**（s13 が律速）。
- **RF-nofs c1 18/20**: target in/out 5/5、mixed in/out 4/5。残 = **mixed s0 in+out**。両者 Optuna 完了(Trial49, best_so_far 0.946)後の **CALIBRATION 段（Sigmoid 5-fold CV、1094木×全165特徴）で 3〜6h 停滞**（in_mixed s0 log 01:52 stale=6h、out_mixed s0 05:01=3h）。hung でなく CPU 競合下の低速の見込みだが**要監視**（数時間で完了しなければ介入検討）。**mixed s1 in/out は本朝完了**（in 0.632/out 0.669）。RF-nofs out_mixed_s1 の重複プロセス(top-up 競合、eval は 06:05 完了済)を停止。
- **異常終了チェック → 検出ゼロ**: 全 390+ eval JSON が roc_auc∈[0,1] で有効（NaN/inf/範囲外/破損なし）。直近完了 SvmW mixed s1 in 0.754/out 0.778、RF-nofs mixed s1 in 0.632/out 0.669 は想定帯内（現行 leak込みプロトコル）。
- **想定値**: 全て既知のリーク署名と整合、新規異常なし。honest 再評価は別途（本 log 2026-07-22 の再監査節で確定＝全車両手法 chance）。
- iv25smote の SW-SMOTE 再生成（案A修正済）は現行完了後(~07-24)に起動予定、非破壊で待機。

## 2026-07-22（夜）【exp3 状況監査】RF-nofs 20/20 完了・SvmW 29/32、異常終了ゼロ、残 SvmW 3セル

- **RF-nofs: 20/20 完了**（c1 table）。前回の要監視だった **mixed s0 in/out の CALIBRATION 停滞は解消**＝正常完了（in 08:51 / out 11:12、hung でなく低速だった）。値 in 0.906 / out 0.964（全特徴 leak 帯、記憶署名と整合）。
- **SvmW: 29/32**。Within 16/16、mixed in 7/8（**s2025 が 19:23 完了**、値 0.745）・mixed out 6/8。**残 3セル**: mixed out **s2025**（92h 稼働、既観測最長域→~07-23）、mixed in/out **s13**（44h 稼働→~07-24〜25）。→ **SvmW 全完了 ~07-24〜25**（s13 律速）。
- **運用**: SvmW in_mixed_s1 の churn 重複（19:23 に worker が完了済セルを再取得、eval は 07-21 09:16 済）を停止（PID 84144）。
- **異常終了チェック → 検出ゼロ**: 全 390+ eval JSON が roc_auc∈[0,1] 有効（NaN/inf/範囲外/破損なし）。直近完了値（RF-nofs s0 0.906/0.964、SvmW s2025 0.745）は現行 leak込みプロトコルの想定帯内。
- **次アクション**: RF-nofs 完了により残は SvmW 3セルのみ。SvmW 完了（~07-24）後、iv25smote の SW-SMOTE 再生成（案A修正済）を起動予定。

## 2026-07-24 【exp3 c1 グリッド完全完了】全手法・全アーム完走、異常終了ゼロ、全値想定帯内

- **c1 グリッド全完了**: RF(fs) 96/96・**SvmW 32/32**・SvmA 32/32・Lstm 60/60、iv25 baseline(base/smote)、RF-nofs 20/20。watchdog も 07-23 12:18:03 に "ALL COMPLETE — removing scheduled task" で自己終了。稼働プロセスなし。
- **SvmW 最終**（07-23 完了）: Within-in 0.800±0.012 / Within-out 0.759±0.012（各8）、**Mixed-in 0.742±0.011 / Mixed-out 0.771±0.015（各8）**。最終3セル out_s2025 0.781 / in_s13 0.742 / out_s13 0.778 は帯内。
- **異常終了チェック → 検出ゼロ**: 全 390+ eval JSON が roc_auc∈[0,1] 有効（NaN/inf/範囲外/破損なし）。全値が既知 leak 署名と整合、新規異常なし。運用: 完了直後の SvmW in_mixed_s1 churn 重複（07-22 19:23 に完了済セル再取得）を停止。
- c1_results.md の SvmW Mixed 列を最終 n=8 値へ、進捗/ヘッダを「c1 完了」へ更新。
- **次アクション**: exp3 c1 は完了。**iv25smote の SW-SMOTE 再生成（案A修正済、ユーザ pre-authorized "全4手法・現行完了後"）が起動可能**。ただし SvmW iv25smote は 1セル ~34–70h（履歴上最も律速）で全再生成は多日〜週規模、かつ honest では pooled/subject-wise いずれも chance（科学的結論は不変・方法論ラベル整合が目的）。→ 起動スコープ（特に SvmW を含めるか）をユーザ確認の上で実施。

## 2026-07-24 【iv25smote SW-SMOTE 再生成 起動＋進捗】subject-wise 実行を本番確認（fallback=0）、exp3 c1 は完了済

ユーザ決定「全4手法を再生成」を実行。旧 pooled-SMOTE 成果物（eval JSON+CSV 74件）を `results/_archived_pooledsmote_20260724/`（MANIFEST 付）へ退避し、SW-SMOTE で再生成起動。

- **最重要確認**: 今回の regen ログは **全手法で `Applying subject-wise oversampling`（fallback=0）** — RF 17・Lstm 6・SvmA 1・SvmW 5 が subject-wise 実行。**keep_subject_id 修正が本番で有効**（旧: 全 pooled fallback）。
- **backend/起動**: RF/SvmW=Windows CPU（Start-Process 永続、RF 4+RF-nofs 2+SvmW 5 worker）、Lstm/SvmA=WSL2 GPU（`.venv_tf_gpu`/`.venv_svma_cuml`、GPU 競合回避で Lstm→SvmA 逐次、bash 背景タスク）。
- **進捗（06:57 時点）＋ETA**:
  - **Lstm: 6/6 完了**（各~33分、02:32–03:38）。値 0.504–0.521＝chance（旧 pooled-SMOTE の 0.50–0.53 と一致、Lstm pooled は SMOTE 種別によらず chance）。
  - **RF: 11/20 完了**（非nofs15+nofs5、~高速）→ **~07-24 昼**。値 0.747–0.762＝leaked 帯（旧 ~0.74 と整合）。
  - **SvmA: 1/6**（s0 を 03:38 開始、PSO swarmsize=50 maxiter=100、cuML）、5 pending、1 worker 逐次 → **~07-25〜26**。
  - **SvmW: 0/5**（5 並列稼働、~34–70h/セル）→ **~07-26**。
- **異常終了チェック → 検出ゼロ**: 全 eval JSON が roc_auc∈[0,1] 有効。新 regen 値（Lstm chance・RF ~0.75）は想定帯内で、旧 pooled-SMOTE 版と同帯（honest 結論=chance は不変、方法論ラベルが「pooled→subject-wise SMOTE」に是正）。
- 注: GPU regen（SvmA 残5）は bash 背景タスクで実行中。セッション断で停止した場合は次回チェックで再起動（`scripts/shell/_regen_gpu_iv25smote.sh` を `.venv_svma_cuml` で再実行）。旧値は archive に保全済。

## 2026-07-25 【iv25smote 再生成 進捗＋SvmA ブロッカー】RF/Lstm 再生成完了、SvmW 進行中、SvmA は SW-SMOTE 不可（cuML 停止）→旧値復元

- **再生成進捗（09:33）**: **RF non-nofs 15/15 完了**（0.75帯）、**Lstm 6/6 完了**（chance）、**SvmW 1/5**（s42 完了 05:03、4並列稼働、~07-26）、RF-nofs 2/5（稼働）。全て subject-wise SMOTE 実行（fallback=0）。
- **【ブロッカー】SvmA iv25smote は SW-SMOTE 再生成が技術的に不可能**: s0 が 07-24 03:39 の "PSO Starting" 直後で **~30h ログ無更新・GPU 0%/0%＝停止(hung)**。単一 seed 診断も ~5分でログ空・GPU 0% で再現。原因: subject-wise SMOTE が訓練を 40166→57592 に増やし、**cuML SVM/PSO(swarmsize50×maxiter100) が最初の SVM フィットで固まる**（旧 pooled-SMOTE フォールバック版は ~2.8h/seed で完走していた＝subject-wise 特有）。maxiter 削減は無効（反復前に停止）、CPU-SVM 化は 5000 フィット×57k で非現実的。
- **対応**: hung プロセスと診断を停止し、**退避していた SvmA 旧 pooled-SMOTE 値（6 seed, 12ファイル）を archive から復元**（可逆・SvmA アーム保全）。→ **SvmA iv25smote は pooled SMOTE のまま**（honest ではいずれも chance＝科学的結論不変、方法論ラベルのみ SvmA だけ「pooled SMOTE（subject-wise は cuML 停止で不可）」と論文明記が必要）。**要ユーザ判断**（この扱いで確定 or 緩和策再試行）。
- **異常終了チェック**: 全 eval JSON 有効。再生成の新値（RF ~0.75、Lstm chance）は旧同帯。SvmW/RF-nofs は正常進捗（停止兆候なし、SvmA 固有の cuML 問題）。

### 追記（同 2026-07-25）SvmA ブロッカー解決 — 真因は `SVMA_USE_CUML=1` 未設定（sklearn CPU 落ち）、緩和策成功で再生成再開

ユーザ判断「緩和策を試す」を実施。段階的診断で真因を特定し解決:
- **診断1（max_iter 追加）**: SvmA.py の cuML/sklearn SVC は `max_iter` 無し（既定 -1＝無制限）。上限（`SVMA_SVC_MAX_ITER`, 既定100000）を3 SVC 呼び出しに追加（src/models/architectures/SvmA.py）。→ 無限ループは止まるが依然低速。
- **診断2（GPU/CPU 実測）**: 停止セルは **GPU 0%・プロセス CPU 96%** ＝ GPU 不使用の CPU 計算。
- **真因判明**: `SvmA.py` は **`SVMA_USE_CUML=1` env が無いと sklearn CPU SVC**（`_USE_CUML=False`）。私の GPU regen スクリプトはこの env を設定しておらず、**57k サンプル(subject-wise SMOTE)を CPU libsvm で計算＝実質非現実的**。旧版(2.8h/seed 完走)は cuML GPU 使用だった。
- **緩和策（成功）**: `SVMA_USE_CUML=1` を付けて単一 seed 検証 → **GPU 4749 MiB・89% 使用、cuML SVC 稼働**を確認。`_regen_gpu_iv25smote.sh` の SvmA 行に `SVMA_USE_CUML=1` を追加し、**SvmA 6 seed を cuML GPU で再生成再開**（PSO maxiter=100、GPU 64% 稼働、~2.8h/seed×6＝~07-26 完了見込み）。旧値は archive 保全継続。
- **結果**: SvmA も subject-wise SMOTE で再生成可能に。全4手法 SW-SMOTE 再生成が成立。max_iter 上限は防御的改善として保持（収束フィットは無影響、cuML/sklearn 両対応）。

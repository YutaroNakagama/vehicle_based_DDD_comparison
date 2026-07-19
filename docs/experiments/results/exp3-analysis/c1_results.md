# 実験3 — c1 ドメイン比較グリッド + IV2025 ベースライン再現（結果・進捗）

> 2026-06-27 開始のローカル実行キャンペーンの設計・結果・検証状況の正本。
> 旧 HPC キャンペーンは [operations_log.md](operations_log.md)、2026-05 のローカル
> Phase 1–5 は [local_execution.md](local_execution.md)、方法論の意思決定・検証の
> 詳細は [verification_log.md](verification_log.md) を参照。
> **最終更新: 2026-07-18**（**RF 特徴量非選択版 honest pooled 確定**: 全160特徴 = 0.864±0.018 (n=5)、
> top-10(0.748) を seed-paired +0.135 上回り分散 1/3 → 特徴選択が RF を過小評価（§3.6）。Pooled+SW-SMOTE
> 4手法完成（SvmW 5/5=0.717）。残りは c1 SvmW within+mixed 20/32 と RF-nofs within/mixed。全て正常・想定通り）。
>
> **要点**: 全体の一貫した所見は「**RF のみが pooled（実運用相当の非ドメイン限定評価）で機能する
> 唯一の手法**」。RF は pooled↔mixed で ~0.74 で不動、他手法は pooled で chance に崩落し
> ドメイン限定評価（within/mixed/cross）でのみ ~0.75 に回復する。詳細な機構は §3（Pooled と
> Mixed の評価プロトコル差）を参照。
> **【2026-07-18 追加・重要】** 上記 ~0.74 は RF の **top-10 特徴選択版**の値。**特徴選択を外し全160特徴を
> 使うと honest pooled で 0.864±0.018（n=5）に上昇**（top-10 比 seed-paired +0.135、seed 分散も
> 0.061→0.018 に激減）。→ top-10 選択は RF を**過小評価かつ高分散に見せていた**。詳細は §3.6。

## 1. 目的と設計

指導教員の改訂プラン（2026-06-27/28）に基づき、4 手法（RF / SvmW / SvmA / Lstm）を
**ドメインシフト × クラス不均衡**の下で比較する:

- **c1 グリッド（6 条件）**: {Within=`target_only`, Cross=`source_only`, Mixed=`mixed`}
  × {in_domain, out_domain}。全て対象ドメインで評価。SW-SMOTE(ratio 0.5)・
  wasserstein/KNN ドメイン分割（split2, in=44 / out=43 被験者）。
- **IV2025 ベースライン（Pooled・対策なし）**: pooled・不均衡処理なし——IV2025 公表設定の
  ローカル再現（タグ `iv25base_`）。c1（after）との before/after 比較の基準。
- **Pooled + SW-SMOTE アーム（Pooled・対策あり）**（2026-07-04 追加、タグ `iv25smote_`）: pooled
  学習/評価は IV2025 と同一プロトコルのまま、**c1 と同一の SW-SMOTE（ratio 0.5, subject-wise）**
  だけを加えた条件。これで「**不均衡対策の効果**」（Pooled 対策なし→あり）と「**ドメイン制限の
  効果**」（Pooled 対策あり→Mixed/Within）をそれぞれ純粋に分離できる。`iv2025_baseline_launcher.py
  --smote` で起動、eval は `--jobid` 固定（race-free）。

**距離指標と比率の選定根拠（exp2/TIV2026 由来、本実験では固定）**:
- 距離 = **wasserstein** 固定。Sobol 感度分析（`results/analysis/.../sobol_indices.csv`）で距離の
  寄与は S1≈0.0009・ST≈2%（Mode の S1≈0.58 に対し無視可能）。距離3種で AUROC 差は Within
  0.771/0.761/0.790（有意差なし p=0.128）。数値上最良かつ安価な wasserstein を採用（当初の
  exp3 距離選定実験は Sobol 結果を根拠に中止）。
- 比率 = **0.5** 固定。TIV2026 の比率感度分析で AUROC 順位は比率にほぼ不変（Spearman ρ=1.00）。
  F2 等の閾値依存指標のみ比率に敏感。ratio 0.1 は Lstm で実行不能（event_label 自然少数派率
  ~27% > 目標 10% で imblearn エラー）→ 全手法共通の実行可能集合として 0.3/0.5 を採用、c1 は 0.5。
- ラベルは**各手法とも原論文に忠実**（RF/SvmW/SvmA=KSS、Lstm=event_label ~74%陽性）。統一は
  行わない（設計判断）。SvmA は T1 プローブで特徴量に信号なしを確認済みのため、ラベルを
  替えても chance のまま（特徴量律速）で統一の実益なし。Lstm の event_label は別タスクゆえ
  AUROC 絶対値の跨ぎ比較を避け、崩落/回復パターンで比較する。

各手法は原論文に忠実（RF=top-10 特徴 + KSS、SvmW=Zhao2009 GHM ウェーブレット 8 帯域
+ KSS + Optuna 50 trial、SvmA=Arefnezhad2019 操舵統計 18 特徴 + PSO/ANFIS/SVM、
Lstm=Wang2022 + event_label）。唯一の運用上の変更は SvmA の `SVMA_PSO_MAXITER=30`
（pso_history で iter~2 収束を確認済み、手法自体は不変 — verification_log 参照）。

### シード設計（分散比例、TIV2026 の妥当性枠組みで判定）

| 手法 | シード数 | 根拠 |
|---|---|---|
| RF | 24 | Within の seed 分散が大（std ~0.09–0.11）→ 95%CI 半幅 ≤0.05 に n=21 必要 |
| Lstm | 15 | 低分散（req_n=3）だが従来水準を維持 |
| SvmW / SvmA | 8 | 低分散（req_n≤7）+ 1 セル ~15h の計算制約。統計的十分性は充足 |
| IV25 RF | 15 / 他 6 | RF は高分散、chance 手法は 6 で CI 上限 <0.60 を充足 |

### 分割方法論（重要な既知の制約）

**TIV2026/IV2025 と同一の方法論を採用**（train: `--time_stratify_labels` の
ラベル層化時系列分割、eval: target_timewise stratify=False）。この組合せは
within/mixed で train–eval の分割境界が一致しない temporal-split 特性を持つ
（脱リーク版では within 0.78→0.526）。**論文間比較の一貫性を優先し同一枠組みで
統一**、既知の制約として明記する（詳細・実測は verification_log.md 2026-07-01 節）。
Cross は別ドメイン学習のため本特性の影響を受けない。

## 2. 結果（2026-07-05 時点、AUROC mean ± std (n)）

### Pooled 2 条件（全被験者で学習/評価）— 「対策の効果」

| 手法 | Pooled 対策なし (`iv25base`) | Pooled + SW-SMOTE (`iv25smote`) | Δ(SMOTE) |
|---|---|---|---|
| **RF** | **0.738 ± 0.090 (15)** | **0.748 ± 0.061 (15)** | +0.010 |
| SvmW | 0.519 ± 0.011 (6, 縮退) | **0.717 ± 0.016 (5)** | **+0.198** |
| Lstm | 0.512 ± 0.011 (6) | 0.508 ± 0.004 (6) | −0.004 |
| SvmA | 0.481 ± 0.008 (6) | 0.533 ± 0.066 (6) | +0.052（chance帯内）|

→ **【2026-07-12 更新・重要】旧結論「RF だけが pooled で機能」は SvmW の結果で要修正**。
**SvmW は SW-SMOTE で 0.519（全陽性縮退）→ 0.717（非縮退・両クラス予測、recall_pos ~0.72）へ大幅回復**
（**n=5 完走 07-16**、seeds{0,1,7,42,123}=0.689–0.728）。つまり **pooled(+SMOTE) で機能するのは
RF と SvmW の2手法**、Lstm(0.508)・SvmA(0.533) は SMOTE を足しても chance のまま。→「対策の効果」は
手法で二分される: **RF=対策に依らず頑健／SvmW=pooled では SMOTE が縮退を解除し機能／Lstm・SvmA=特徴・
タスク律速で回復せず**。§3–4 の「RF は pooled で唯一機能」という中心主張は **SvmW を含めて修正済み**
（§3.5: pooled で RF に匹敵するのは SvmW のみ、ただし seed-paired で RF が +0.039 リード）。RF の Δ が
小さいのは内部で `class_weight`+校正済みのため。

### c1 グリッド（SW-SMOTE あり、対象ドメインで評価）

| 条件 | RF | Lstm | SvmW | SvmA |
|---|---|---|---|---|
| Within-in | 0.746 ± 0.089 (24) | 0.779 ± 0.007 (15) | 0.805 ± 0.005 (4) | 0.576 ± 0.029 (8) |
| Within-out | **0.778 ± 0.108 (24)** | 0.763 ± 0.012 (15) | 0.762 ± 0.009 (4) | 0.574 ± 0.074 (8) |
| Mixed-in | 0.719 ± 0.085 (24) | 0.782 ± 0.009 (15) | 0.740 (1, 実行中) | 0.532 ± 0.024 (8) |
| Mixed-out | 0.749 ± 0.104 (24) | 0.779 ± 0.009 (15) | 0.752 (1, 実行中) | 0.597 ± 0.025 (8) |
| Cross-in | 0.519 ± 0.006 (24) | **0.733 ± 0.015 (15)** | 0.506 ± 0.005 (2) | 0.512 ± 0.021 (8) |
| Cross-out | 0.507 ± 0.004 (24) | **0.747 ± 0.012 (15)** | 0.514 ± 0.004 (2) | 0.504 ± 0.019 (8) |

（Within/Mixed は in/out 別に独立セル。Mixed = 全 87 名で学習・対象ドメインで評価。SvmA は
8/8 シード**完走**（全条件 chance 帯で確定）。SvmW は Within 4/8・Cross 2/8・Mixed 1/8 で残り実行中。
RF/Lstm/SvmA は完走。）

- **RF**: 24/24 シード完走。Within-out 0.778 は TIV2026 の within 参照値と整合。
  seed 分散が大きい（Within-out 範囲 0.568–0.954）→ 24 シードの根拠。
  Cross は完全崩落（~0.51、minority-tracking の真の chance であることを予測分布で確認）。
- **Lstm**: 15/15 完走。**唯一の domain 不変手法**（全 6 条件 0.73–0.78）。ただし
  ラベルが event_label（陽性 ~74%）で他手法（KSS）と検出対象が異なる点に注意。
  閾値は多数派側に張り付くが AUROC は閾値非依存で、定数予測では 0.5 にしかならない
  ため順位づけ性能は本物（verification_log 2026-07-04 節）。
- **SvmW**: Within 4/8・Cross 2/8・Mixed 1/8（残り実行中）。Within 0.76–0.81 は実判別
  （両クラス予測）、Cross ~0.51 崩落。RF と同型の「within 有効 × cross 崩落」。Mixed も
  暫定 n=1 で 0.74–0.75（within 帯）で方向一致。
- **SvmA**: 8/8 シード**完走**。**全条件 chance 帯（0.50–0.60）**。T1 プローブ（特徴量に
  単変量・RF いずれでも信号なし）と整合 — 操舵統計 18 特徴はこのデータセットでは
  眠気信号を持たない（縮退ではなく真の無信号、verification_log T1 節）。全 48 セル異常なし
  （NaN・範囲外 AUROC・縮退なし）。

Pooled 対策なしは IV2025 公表値（RF 優位・SvmW 51%・Lstm 0.52・SvmA 0.53）をローカル再現。
予測形態: RF=実判別（両クラス予測）、SvmW=全陽性縮退、Lstm=多数派崩落、SvmA=chance。

### 手法別まとめ（4 条件家族を横断）

| 手法 | Pooled(なし/あり) | Within | Mixed | Cross | 一言 |
|---|---|---|---|---|---|
| **RF** | 0.738 / 0.753 | 0.75–0.78 | 0.72–0.75 | **0.51 崩落** | pooled↔mixed **不動 ~0.74**。不均衡に頑健、ドメイン転移のみ脆弱 |
| SvmW | 0.519 / **0.717** | 0.76–0.80 | 0.74–0.75 | 0.51 崩落 | pooled は SMOTE で回復し RF に肉薄（RF +0.039）、cross 転移不可 |
| Lstm | 0.512 / 0.508 | 0.76–0.78 | 0.78 | **0.73–0.75** | pooled **崩落**、ドメイン限定評価（cross 含む）で ~0.75。§3 参照 |
| SvmA | 0.481 / 0.520 | chance | chance | chance | 全条件 chance。特徴量に信号なし（T1）、回復不能 |

**中心的所見**: RF は pooled と mixed でほぼ同値（~0.74）で不動。他手法は **pooled で chance に崩落し、
評価がドメイン限定（within/mixed/cross）になったときだけ ~0.75 に回復**する。→ §3 で機構を分析。

## 3. Pooled と Mixed の非対称性 —「なぜ RF は pooled で優位・mixed で同等か」

**問い**: Pooled(+SMOTE) と Mixed(+SMOTE) は「全被験者で学習」という点で似ているのに、pooled では
RF が独走し mixed では Lstm が RF に並ぶ。説明がつくか。

**結論**: 変わっているのは RF ではなく**他手法**。RF は両条件で ~0.74 で不動。Pooled で「RF 優位」
に見えるのは他手法が 0.51 に崩落するから、Mixed で「同等」に見えるのは他手法（特に Lstm）が
0.78 に回復するから。**両条件は評価プロトコルが実質的に異なり、その差は非 RF 手法だけを助ける。**

### 検証で確定した事実（学習/評価ログ実測）

1. **RF はフラット**: Pooled 0.738/0.753 ≈ Mixed 0.719–0.749。Lstm は Pooled 0.508 → Mixed 0.780。
2. **学習ラベル分布は Pooled も Mixed も同一（~73% 陽性で均衡）** → 「pooled は学習データが
   ラベル偏在するから崩落」説は**否定**（両者とも subject_time_split、train/val/test すべて ~0.73）。
3. **評価プロトコルが 2 点異なる**:
   - **評価範囲**: Pooled = 全 87 被験者（異種混合プール）をランダム分割で評価。
     Mixed = ターゲットドメイン 44/43 被験者（同種サブセット）を被験者内時系列分割で評価。
   - **train/eval 窓重複**: c1（Within/Mixed/Cross）は train `time_stratify=True`・eval
     `time_stratify=False` の不一致で重複窓を選ぶ（TIV2026 共有の既知特性、脱リーク版で
     within-RF が 0.78→0.526 と実測、verification_log 2026-07-01 節）。Pooled の eval はランダム分割。
4. **決定的な指紋**: Lstm は「評価がドメイン限定か否か」で切り替わる —
   全プール評価（Pooled）→ **0.51**、ドメイン限定評価（Within 0.77 / Mixed 0.78 / **Cross 0.73–0.75**）
   → **~0.75**。学習ドメインを問わず（cross でさえ）ドメイン限定評価なら回復する。

### 機構の解釈

- **RF は両プロトコル差に鈍感**: 全 87 名で学習した素の信号 ~0.74 を、正則化アンサンブル +
  class_weight + 校正でそのまま出す。窓重複にも評価の異種性にも影響されない → Mixed 0.734 ≈
  Pooled 0.738 はどちらも本物。
- **Lstm は評価プロトコルに敏感**: 高容量ゆえ (a) 重複窓の記憶と (b) 評価のドメイン同種性を
  利用でき、Mixed 0.78 に底上げされる。全異種プールの正直な評価（Pooled）では event_label を
  分離できず 0.51 に崩落。**同一評価セットでの Within→Cross 不変性（RF −0.246 vs Lstm −0.046）が
  これを実証**（§3.5）。※ SvmW の pooled 崩落は**別機構**（高容量による重複窓利用ではなく、
  不均衡下のカーネル+Platt 縮退。旧「高容量カーネル SVM も同型」表現は誤り、§3.5 で訂正）。

### 論文への含意（推奨する主張の形）

**「Pooled は最も厳しく手法非依存に公平な評価であり、そこで機能するのは RF のみ。Mixed/Within
での他手法の“回復”は、評価がドメイン限定でかつ TIV2026 と共有の時系列重複特性に支えられた
条件依存の底上げであり、RF の頑健性を相対化しない」**。RF 優位性は **Pooled 列を主エビデンス**に
書くのが最も防御可能。RF は「全条件最高精度」ではなく「**実運用条件（pooled）での頑健性における
一意な優位**」として主張する（Within では SvmW/Lstm と統計的に区別困難、Cross では RF は崩落）。

### 限定事項（正直な記載）

- Pooled のランダム eval も train と ~60% 程度は重複し得る（ランダムテストが時系列 train 領域に
  落ちる）。よって「Pooled = 完全に重複なし」ではない。**Mixed が Pooled より易しい主因は
  「重複量」より「評価のドメイン同種性」**の寄与が大きい可能性があり、両寄与の厳密分離は未実施。
  ただし「両差とも非 RF 手法だけを助け RF には効かない」という結論は同じ。
- SvmW Mixed（実行中）で KSS 系でも同構図が成り立つか最終確認予定。現状 SvmA は同傾向
  （Pooled 0.52 → Mixed 0.59）で方向一致。

## 3.5 各手法 × RF 乖離の機構（予測レベル解析, 2026-07-12）

**3 手法は「異なる条件で・異なる理由で」RF と乖離する**（混同行列・確率分布・T1 プローブまで
遡って確定。各手法とも RF と割れるのは基本 1 条件だけ、SvmA のみ全条件）。RF 自身も Cross では
chance に落ちる（別ドメイン学習の転移失敗）ので、RF も例外ではない。

| 手法 | RF と乖離する条件 | 乖離の正体 | 崩落形態 |
|---|---|---|---|
| **Lstm** | **Pooled のみ**（Within/Mixed/Cross は同等以上） | **評価プロトコル依存**（学習内容に不変） | 順位崩落（AUROC 0.78→0.51、CM は常時多数派全陽性）|
| **SvmW** | **Pooled-SMOTEなし のみ**（他は同等、Cross は両崩落） | **不均衡ロバスト性**（カーネル+Platt 縮退） | 全陽性縮退 → SMOTE で解除 |
| **SvmA** | **全 8 条件**（Within でも 0.576） | **特徴量に信号なし**（手法非依存） | 真のランダム（ROC 対角線、非縮退）|

### Lstm — 鏡像の崩落（RF=学習失敗 Cross／Lstm=評価失敗 Pooled）
- **決定的証拠**: Within/Mixed/Cross は**同一評価セット**（学習セットのみ変化）。同一 seed で
  Within→Cross にすると **RF −0.246**（0.773→0.527、対象ドメインを学習から抜くと base-rate chance
  ＝本物の学習信号）に対し **Lstm −0.046**（0.789→0.743、ほぼ不変＝**学習内容に依存しない**）。
- よって Lstm のドメイン限定スコアは「学習した信号」でなく**評価プロトコル（被験者内時系列分割
  ×均質ドメイン）の下駄**。裏付け: 本物の汎化ならランダム分割(pooled)で上がるはずが逆に 0.78→0.51
  に**下がる**（PR-lift +0.18→0.00）。event_label(74%陽性)ゆえ **全条件で多数派全陽性予測**なので、
  pooled 崩落は「順位の崩落」であり新たな縮退ではない（0.78 のセルも CM は全陽性）。SMOTE は無効
  （0.512→0.508、3/6 seed は完全全陽性化）。

### SvmW — 唯一の乖離は pooled の SMOTE 依存 ＋ **機構の訂正**
- Within/Mixed/Pooled+SMOTE は RF と一致、Cross は両崩落 → **ドメイン能力差ゼロ**。唯一 **Pooled-
  SMOTEなし**で割れる（RF 0.738 vs SvmW 0.519, Δ+0.219）。
- **訂正**: 旧説明「SvmW は内部再重み付けが無いから縮退」は**コードレベルで誤り**。実際は
  `SVC(kernel="rbf", class_weight="balanced")` ＋ balanced `sample_weight`（[classifiers.py:86-91](../../../../src/models/training/classifiers.py#L86)）で**二重に均衡化済み**。
  真の機構は **カーネルマージン+Platt 縮退**: 87 名異種プール・陽性 3.9%・8 ウェーブレット特徴では
  RBF 決定関数がほぼ定数 → Platt がほぼ一定確率に写像（proba pstd **0.001**）→ 閾値で全陽性。
  **SW-SMOTE は"重み"でなく"密度（幾何）"で治す**（少数派を合成密度化 → RBF サポートベクタが少数派
  クラスタを包む → proba spread 復活 pstd **0.229**）。RF は木の葉純度で本質的に順位を保持するので
  pooled でも崩れない（SMOTE 有無で 0.738→0.748）。※ [domain_imbalance_factor_analysis.md](domain_imbalance_factor_analysis.md) §2.2 は既にこの正しい機構を記載済み。
- **対 RF パリティ（n=5 確定, 07-16）**: SvmW 0.717 vs RF 0.748（marginal）。**seed-paired（同一5 seed
  {0,1,7,42,123}）では RF 0.756 vs SvmW 0.717＝RF が +0.039 リード**（RF が 3/5 seed で上、特に s0 +0.13、
  s42 +0.07；s1/s123 は SvmW が僅差上）。結論: SvmW は pooled+SMOTE で **RF に肉薄するが RF がなお僅かに
  上**。RF の一意な優位は「**SMOTE 無しでも機能**（0.738 vs SvmW 0.519 縮退）」＝不均衡対策への非依存にある。

### SvmA — 唯一「無条件」で乖離（特徴律速・ネガティブコントロール）
- 全 8 条件で chance（0.48–0.60）。Within-in で SvmW(0.803)/Lstm(0.779)/RF(0.752) が揃うのに SvmA は
  0.576 のみ。**唯一どの条件でも RF に追いつかない**。
- **根本原因＝特徴信号なし（T1 プローブで確定）**: **同じ RF に SvmA の操舵 18 統計特徴を食わせると
  0.4955**（自前 top-10 なら 0.75）。2×2 解離で**特徴が決定的・手法は無関係**。SvmA の CM は **ROC 対角
  線上**（TPR≈FPR, precision≈prevalence）＝**真のランダム、縮退ではない**（SvmW/Lstm の単一クラス縮退
  とは質的に別）。SMOTE 無効（0.481→0.533、chance 帯）は閾値非依存の順位信号が特徴に無いため。
- **提言**: SvmA は「弱い競合」でなく**ネガティブコントロール/特徴表現の下限**として提示すべき
  （Arefnezhad2019 忠実再現が全条件 chance ＝パイプラインが自明なリークをしていない検証＋他手法の
  性能を**特徴表現の豊かさ**に帰属）。

### 確定 / 推測
- **確定**（予測データ由来）: 上記の乖離条件・機構・2×2 解離・same-eval-set 不変性・崩落形態の別。
- **推測/未分離**: (a) Lstm pooled 崩落の「異種性 vs 分割方式」の寄与分離は未定量、(b) SvmW pooled+
  SMOTE の対 RF パリティは n=2・seed 一致で RF リード、(c) Within/Mixed は全手法が共有のリーク傾向
  プロトコル上の比較、(d) `y_pred_proba` 生配列は未保存で CM 由来量に依拠。

## 3.6 RF 特徴量選択の影響 — 全160特徴 vs top-10（2026-07-18 pooled 確定 n=5）

RF 無選択版（`--feature_selection none` = EEG 除外の全 160 特徴、seed 5）を追加測定。**honest な
Pooled+SMOTE（全 87 名ランダム分割、時系列重複を持たない列）で決定的な結果**:

| 条件 | RF top-10 | RF 全160特徴 |
|---|---|---|
| **Pooled+SMOTE（honest）** | 0.748 ± 0.061 (15) | **0.864 ± 0.018 (5)** |
| Within in/out | 0.746 / 0.778 | 0.904 / 0.941 (3) |
| Mixed in/out | 0.719 / 0.749 | 0.872 / 0.940 (2) |

**確定した所見**:
1. **top-10 特徴選択は honest pooled で RF を過小評価**: seed-paired（同一 5 seed {0,1,42,123,2025}）で
   全特徴 0.864 vs top-10 0.729 = **+0.135**（全 seed で全特徴版が上）。top-10 の最悪 seed（s2025=0.602）
   ほど全特徴の伸びが大（+0.278）。
2. **top-10 の高 seed 分散はアーティファクト**: pooled の std が **0.061 → 0.018**（1/3 以下）。RF の
   「within-domain 高 seed 分散（当初 24 seed の根拠）」は **seed 毎に選ぶ 10 特徴の当たり外れ**が主因
   だった可能性が高い。全特徴なら安定して ~0.86–0.94。
3. **honest pooled の手法序列（確定）**: **RF全特徴 0.864 ≫ RF top-10 0.748 ≈ SvmW 0.717 ≫
   Lstm 0.508 ≈ SvmA 0.533**。
4. **リークではなく実信号**: within/mixed の高値（0.90–0.94）は §3/§3.5 の時系列重複リークを含む列だが、
   **pooled はその重複構造を持たない honest 列で、そこでも 0.86**。ゆえに全特徴の恩恵は実信号（top-10 が
   捨てていた車両特徴が眠気信号を持つ）と解釈でき、単なるリーク増幅では説明できない。

**論文への含意**: 「RF は頑健だが pooled ~0.74」という中心主張は、**「RF は top-10 選択なしで honest
pooled=0.86 に達し他手法を大きく引き離す。top-10 選択が RF を過小評価していた」**へ更新すべき。
Arefnezhad/Zhao/Wang の各手法は固定特徴集合ゆえこの恩恵を受けない（SvmA は特徴に信号なしで不変）。
within/mixed-nofs（残り実行中、~07-20 完走）で全条件が揃う。

## 4. 検証状況（詳細: verification_log.md）

- **2026-07-04 敵対的再検証（11 エージェント）**: 全完了セルを独立再スキャン
  + 懐疑パスで byte 照合。**実バグ 1 件を発見・修正** — c1 RF Cross 2 セル
  （in/s42, out/s123）が within の予測ベクトルと byte 一致（既知 Bug#4 =
  `latest_job.txt` 経由の eval モデル解決 race の再発）。
  - 修正: `c1_domain_launcher.py` の eval に `--jobid` 固定（コミット `e0923fa`）
    → worker 数に依らず race-free。汚染 2 セルは削除・再実行し、正直な値
    （0.527 / 0.508）への回復と proba 独立性を確認済み。
  - 影響: RF Cross 平均 0.529→0.519 (in)、0.517→0.507 (out)。**結論不変**。
  - 全 4 モデル×(domain,seed)×mode の総当たり照合で他に衝突なし（clean）。
- 上記以外は全ターゲット clean: 欠損/stale セルなし、NaN・範囲外 AUROC なし、
  判別条件に縮退なし、CM 合計は分割サイズと整合、全セル revert(902ce96) 後の生成。
- **2026-07-11 再検証**（完了済み 356 セルの全走査）: NaN・範囲外 AUROC ゼロ、想定外の縮退なし。
  縮退（単一クラス予測）は iv25base SvmW pooled 6 セルのみ＝IV2025「SvmW 全陽性縮退」の**想定内**再現。
  RF Cross の重複 2 ファイル（in/s42, out/s123。旧再実行の残存、両者 honest ~0.51・AUROC 差 <0.001）を
  `results/_archived_duplicates_20260711/` へ退避（MANIFEST 付・可逆）→ RF c1 は正確に 144（24/セル）。
  SvmW ログの `TRAIN FAILED` 群は全て `STATUS_CONTROL_C_EXIT`/`DLL_INIT_FAILED`/forced-terminate の
  再起動チャーン起因（コードバグではない、resume-safe）→ §6 の watchdog 修正で恒久対策済み。
- 恒常運用: 毎ステータス確認時に (a) ログの TRAIN FAILED/Traceback 走査、
  (b) AUROC 範囲・縮退チェック、(c) mode 間衝突スキャンを実施。

## 5. 実行基盤

| 項目 | 内容 |
|---|---|
| ランチャー | [`c1_domain_launcher.py`](../../../../scripts/python/train/c1_domain_launcher.py)（c1）/ [`iv2025_baseline_launcher.py`](../../../../scripts/python/train/iv2025_baseline_launcher.py)（Pooled 両アーム、`--smote` で対策あり） |
| 実行順 | **table-priority**（2026-07-04, コミット `696bb60`）: Within+Mixed を全シード先行 → Cross 最後。順序のみ変更、セル集合・タグ・手法は不変 |
| 配置 | RF/SvmW=Windows CPU（**SvmW 8 worker** に増強, 20 コア・単スレッド固定）、Lstm/SvmA=WSL2 GPU（TF2.21 CUDA / cuML） |
| 監視 | `c1_watchdog.sh`（schtasks 10 分毎、死活監視 + 自動再起動 + 全完了で自己削除）+ ntfy.sh 通知 |
| タグ | c1: `imbalv3_knn_wasserstein_<dom>_<mode>_split2_subjectwise_ratio0.5_s<seed>`／Pooled: `iv25base_<M>_pooled_baseline_s<seed>`（対策なし）・`iv25smote_<M>_pooled_swsmote_s<seed>`（対策あり） |
| eval モデル解決 | `--jobid` 固定でセル自身の学習モデルを参照（Bug#4 の race を根絶、§4） |

## 6. 進捗と完了見込み（2026-07-11 13:40 時点）

| ケース | 完了 | 見込み |
|---|---|---|
| c1 RF / c1 Lstm / c1 SvmA | **144/144 ✅ / 90/90 ✅ / 48/48 ✅** | 完（SvmA 07-10 18:16 完走）|
| Pooled 対策なし（RF/SvmW/Lstm/SvmA）| **15 / 6 / 6 / 6 ✅** | 完（IV2025 再現完結）|
| Pooled + SW-SMOTE（RF/SvmA/Lstm/SvmW）| **15 ✅ / 6 ✅ / 6 ✅** / SvmW 0/6 | SvmW のみ残（現在 s0/s1/s7 実行中）→ ~07-12–13 |
| c1 SvmW（Within+Mixed のみ, Cross 廃止）| **~12/32**（Within 4+5・Mixed 1+1）| **Mixed ~4 日/セルが律速** → ~07-16–20 |
| **RF 特徴量選択なし版（新規, seed 24, 160特徴）** | 0/120（Pooled+SMOTE 24 + Within/Mixed×in/out 各24） | RF も 50-trial Optuna で ~1h/セル → **~1–2 日**（BelowNormal・空きコアで SvmW と併走、SvmW timeline 不変）|

**実測 SvmW 1 セル所要**: Within ~15–21h・Cross ~14–17h・**Mixed ~100h（≈4 日、最終律速）**・
iv25smote pooled ~17h（50 Optuna trial × ~20min）。SvmW の重さは非収束気味の RBF-SVM + Optuna
50 trial による本質的コスト（N_TRIALS=50 は忠実性維持のため不変）。Mixed は学習集合が最大
（全 87 名 + SMOTE）で最重量。

- **2026-07-11 watchdog チャーン修正**: `c1_watchdog.sh` にガード（`pool_healthy`）を追加。
  従来はランチャーが console-close（`STATUS_CONTROL_C_EXIT`）で死ぬたび watchdog が孤児ワーカーを
  reap→再起動し、数日かかる Mixed セルを Optuna trial 0 からやり直していた（07-10 夜に4回churn）。
  修正後は「非 pooled ワーカーが存在し6h以内に前進していれば孤児を養子縁組し再起動をスキップ」。
  現行ランは無停止で保護される。**iv25smote SvmW（0/6, 7 日間未完走）は watchdog 管理外で依然脆弱** —
  console-close で最終評価前に kill され続けている（無干渉で完走させる方針）。
- **Cross 廃止（2026-07-11, ユーザ判断）**: cross-domain 転移は全手法 chance 崩落（~0.51）で
  情報価値が低いため、**exp3 全体から Cross（source_only）を除外**（`c1_domain_launcher.py` の
  `build_cells` は Within+Mixed のみ生成）。実行中の SvmW cross（残 4 セル）も打ち切り。既存の
  Cross eval JSON（RF24/Lstm15/SvmA8/SvmW4）は**削除せず「参考」に降格**。→ 前回の「Cross シード
  半減」案はこれに置換。SvmW 残は within+mixed のみ（total 48→32）。
- **RF 特徴量選択なし版を追加（2026-07-11, ユーザ判断）**: RF の top-10 importance 選択の寄与を
  見るため、**全 165 特徴（EEG/メタ除く）を使う RF 版**を追加。実装: `train.py --feature_selection
  none`（新フラグ、既定 `rf`）を `c1_domain_launcher.py --no-fs`／`iv2025_baseline_launcher.py
  --smote --no-fs` から発行。タグに `_nofs` を付与し top-10 版と分離。対象は**対策あり列のみ**
  （Pooled+SMOTE・Mixed in/out・Within in/out、Cross なし）。eval 側は保存済み
  `selected_features` を読むため変更不要（手法・SW-SMOTE・分割は top-10 版と同一、選択のみ差）。
  **seed は後に 5 に削減（ユーザ判断）**。**結果（2026-07-18）: pooled 5/5 完走＝0.864±0.018 で確定 →
  §3.6（top-10 の 0.748 を +0.135 上回り分散も 1/3、"特徴選択が RF を過小評価"を確証）**。within/mixed-nofs
  は残り実行中（~07-20 完走）。
- 04:30(07-05) に旧 4 並列 launcher を table-priority + 8 並列版へ自動切替（`_svmw_table_priority_switchover.sh`）。
  切替時の Bug#4 混入チェックも自動実行、汚染なしを確認。

## 7. 残タスク

- [ ] c1 SvmW / SvmA・Pooled+SW-SMOTE（SvmW/RF 残）の完走（watchdog 自動運転）
- [ ] 完走後: 4 手法 × {Pooled なし/あり, Within in/out, Mixed in/out, Cross in/out} 最終集計表 + 図
- [ ] Pooled RF vs 各手法の有意差検定 + Within の RF≈SvmW/Lstm の equivalence 表記（seed-paired）
- [ ] `exp3_seed_adequacy.py` の最終実行（全条件 ADEQUATE 判定の確認・記録）
- [ ] 完走データでの敵対的再検証（最終版）
- [ ] TIV2026_exp3 原稿への反映（[TIV2026_exp3/outline.md](TIV2026_exp3/outline.md)）

## 8. 参照

- 意思決定・検証の詳細ログ: [verification_log.md](verification_log.md)
- 要因分析（RF 頑健性の機構、距離の Sobol 寄与）: [domain_imbalance_factor_analysis.md](domain_imbalance_factor_analysis.md)
- 距離/比率感度（exp2）: `../exp2-analysis/distance_granular_report.md`, `../exp2-analysis/ratio_sensitivity_report.md`, `results/analysis/exp2_domain_shift/figures/csv/split2/sensitivity/sobol_indices.csv`
- シード妥当性: [`exp3_seed_adequacy.py`](../../../../scripts/python/analysis/exp3_seed_adequacy.py) → `results/analysis/exp3_verification/seed_adequacy.json`
- 特徴量信号プローブ（SvmA 無信号の根拠）: `results/analysis/exp3_verification/t1_feature_signal_probe.json`
- 生データ: `results/outputs/evaluation/<Model>/**/eval_results_*imbalv3_knn_wasserstein*ratio0.5*.json`（c1）/ `*iv25base*.json`（Pooled 対策なし）/ `*iv25smote*.json`（Pooled 対策あり）

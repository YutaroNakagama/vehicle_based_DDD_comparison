# 実験3 — c1 ドメイン比較グリッド + IV2025 ベースライン再現（結果・進捗）

> 2026-06-27 開始のローカル実行キャンペーンの設計・結果・検証状況の正本。
> 旧 HPC キャンペーンは [operations_log.md](operations_log.md)、2026-05 のローカル
> Phase 1–5 は [local_execution.md](local_execution.md)、方法論の意思決定・検証の
> 詳細は [verification_log.md](verification_log.md) を参照。
> **最終更新: 2026-07-21**（状況監査: SvmW 26/32・RF-nofs 20/25 稼働中、他手法完走、全 390 セル異常なし＝下記＆ verification_log 2026-07-21 節）。**方針決定（ユーザ 07-20）**: exp3 は **IV2025/TIV2026 と同一方法論で完遂**し
> 論文間比較性を最優先とする（verification_log 2026-07-01 の判断を踏襲）。§2 の数値はこの**同一プロトコル
> （train/eval 分割不一致を含む既知の制約）**下の値。**§3.6 は脱リーク・ロバストネス附録**として保持し
> （honest 分割では車両→EEG は chance の見込み）、**実装見直し・脱リーク再テストは非破壊で"構える"**（下記
> §3.7 計画）。前日の「RF 全特徴=特徴選択が RF を過小評価」主張のみは事実誤認として撤回（記憶リーク）。）
>
> **要点**: 全体の一貫した所見は「**RF のみが pooled（実運用相当の非ドメイン限定評価）で機能する
> 唯一の手法**」。RF は pooled↔mixed で ~0.74 で不動、他手法は pooled で chance に崩落し
> ドメイン限定評価（within/mixed/cross）でのみ ~0.75 に回復する。詳細な機構は §3（Pooled と
> Mixed の評価プロトコル差）を参照。
> **【2026-07-19 訂正】** 前日ここに書いた「全160特徴なら honest pooled=0.86」は**撤回**（train/eval
> 分割不一致の記憶リークだった、§3.6）。**さらに深刻な含意**: この pooled 列の絶対 AUROC（RF ~0.74 含む）
> 自体がリーク混入で、"RF は pooled で機能" の中心主張は honest 分割での再検証が必要（正しく分離すると
> 車両→EEG は全手法 chance ~0.52 の見込み。[[project_exp2_rf_087_unreproducible]] と整合）。

## 1. 目的と設計

指導教員の改訂プラン（2026-06-27/28）に基づき、4 手法（RF / SvmW / SvmA / Lstm）を
**ドメインシフト × クラス不均衡**の下で比較する:

- **c1 グリッド（6 条件）**: {Within=`target_only`, Cross=`source_only`, Mixed=`mixed`}
  × {in_domain, out_domain}。全て対象ドメインで評価。SW-SMOTE(ratio 0.5)・
  wasserstein/KNN ドメイン分割（split2, in=44 / out=43 被験者）。
- **IV2025 ベースライン（Pooled・対策なし）**: pooled・不均衡処理なし——IV2025 公表設定の
  ローカル再現（タグ `iv25base_`）。c1（after）との before/after 比較の基準。
- **Pooled + SW-SMOTE アーム（Pooled・対策あり）**（2026-07-04 追加、タグ `iv25smote_`）: pooled
  学習/評価は IV2025 と同一プロトコルのまま、**c1 と同一の SMOTE（ratio 0.5）**
  だけを加えた条件。これで「**不均衡対策の効果**」（Pooled 対策なし→あり）と「**ドメイン制限の
  効果**」（Pooled 対策あり→Mixed/Within）をそれぞれ純粋に分離できる。`iv2025_baseline_launcher.py
  --smote` で起動、eval は `--jobid` 固定（race-free）。
  > **【2026-07-21・重要／意図した方法論からの逸脱】** この pooled アームは `--subject_wise_oversampling`
  > を渡していても **全4手法で実際は pooled SMOTE で走っていた**（`falling back to pooled oversampling`:
  > RF 20/20・SvmW 5/6・SvmA 6/6・**Lstm 6/6**、`Applying subject-wise`=0）。**c1 の Within/Mixed は subject-wise
  > が正常動作**（SvmW target 16/16・mixed 16/16）。**根本原因**: iv25 launcher が pooled で `--time_stratify_labels`
  > を渡さない→`data_time_split_by_subject`（split.py L468 が subject_id を無条件 drop、keep_subject_id 非対応）
  > 経由→subject-wise 不能。**「不均衡対策は全て SW-SMOTE」の設計に反するため、修正（案A: `data_time_split_by_subject`
  > に keep_subject_id 追加）＋ iv25smote pooled の SW-SMOTE 再生成が必要**（verification_log 2026-07-21 追記2、要ユーザ判断）。

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

> **注記（2026-07-20 方針）**: 以下の Pooled/Within/Mixed の数値は **IV2025/TIV2026 と同一の評価プロトコル
> （train は `--time_stratify_labels`・eval は stratify=False）下の値**であり、論文間比較性のためこの枠組みで
> 完遂・報告する（既知の制約、verification_log 2026-07-01）。**この枠組みには train/eval 分割不一致に伴う
> 行重複が含まれ、絶対 AUROC はその影響を受ける**（脱リーク時の挙動と honest 値は **§3.6 ロバストネス附録**、
> 相対比較＝手法間/before-after は同一枠組み内で解釈）。**脱リーク再テストは §3.7 に準備**（構え）。

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
| Within-in | 0.746 ± 0.089 (24) | 0.779 ± 0.007 (15) | 0.800 ± 0.012 (8) | 0.576 ± 0.029 (8) |
| Within-out | **0.778 ± 0.108 (24)** | 0.763 ± 0.012 (15) | 0.759 ± 0.012 (8) | 0.574 ± 0.074 (8) |
| Mixed-in | 0.719 ± 0.085 (24) | 0.782 ± 0.009 (15) | 0.738 ± 0.013 (5, 残3実行中) | 0.532 ± 0.024 (8) |
| Mixed-out | 0.749 ± 0.104 (24) | 0.779 ± 0.009 (15) | 0.766 ± 0.017 (5, 残3実行中) | 0.597 ± 0.025 (8) |
| Cross-in | 0.519 ± 0.006 (24) | **0.733 ± 0.015 (15)** | 0.506 ± 0.005 (2) | 0.512 ± 0.021 (8) |
| Cross-out | 0.507 ± 0.004 (24) | **0.747 ± 0.012 (15)** | 0.514 ± 0.004 (2) | 0.504 ± 0.019 (8) |

（Within/Mixed は in/out 別に独立セル。Mixed = 全 87 名で学習・対象ドメインで評価。SvmA は
8/8 シード**完走**（全条件 chance 帯で確定）。**SvmW は Within 8/8 完走・Mixed 10/16（残 6 実行中）**・
Cross 2/8（demoted）。RF/Lstm/SvmA は完走。**[2026-07-21 現況]** SvmW 26/32、RF-nofs 20/25、両者稼働中。）

- **RF**: 24/24 シード完走。Within-out 0.778 は TIV2026 の within 参照値と整合。
  seed 分散が大きい（Within-out 範囲 0.568–0.954）→ 24 シードの根拠。
  Cross は完全崩落（~0.51、minority-tracking の真の chance であることを予測分布で確認）。
- **Lstm**: 15/15 完走。**唯一の domain 不変手法**（全 6 条件 0.73–0.78）。ただし
  ラベルが event_label（陽性 ~74%）で他手法（KSS）と検出対象が異なる点に注意。
  閾値は多数派側に張り付くが AUROC は閾値非依存で、定数予測では 0.5 にしかならない
  ため順位づけ性能は本物（verification_log 2026-07-04 節）。
- **SvmW**: Within 8/8 完走・Mixed 10/16（残 6 実行中）・Cross 2/8。Within 0.76–0.81 は実判別
  （両クラス予測）、Cross ~0.51 崩落。RF と同型の「within 有効 × cross 崩落」。Mixed は
  n=5 で in 0.738 / out 0.766（within 帯）で方向一致（暫定 n=1 の 0.74–0.75 から確定へ）。
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

## 3.6 ロバストネス附録: 脱リーク感度解析（2026-07-19〜20, 4エージェント敵対的監査＋実モデル脱リーク）

> **位置づけ（2026-07-20 方針）**: 本節は §2 の主結果（IV2025/TIV2026 と同一プロトコル）に対する**脱リーク・
> ロバストネス附録／感度解析**。exp3 は同一枠組みで完遂・報告し、本節は「同枠組みが持つ train/eval 分割
> 不一致の影響」を定量する（既知の制約 verification_log 2026-07-01 の深掘り）。**要点**: honest 分割
> （train/eval 一致・被験者ホールドアウト）では車両→EEG(KSS) は全手法・全特徴数で chance ~0.52 に低下し得る。
> ※ 本節の当初の派生主張「RF 全特徴=特徴選択が RF を過小評価」のみは事実誤認として撤回（記憶リーク由来）。
> 正式な脱リーク再テストは **§3.7** に準備。

### 監査で確定した機構（コード＋実モデル＋実ログ、敵対的反証を通過）
- **pooled**: train=被験者内時系列（`data_time_split_by_subject` 前半 **60%**, `TRAIN_RATIO=0.6`）／eval=
  **ランダム 20%**（`eval_pipeline.py:152`、eval に `--subject_wise_split` を渡さない）→ 独立分割で
  **eval テスト行の ~60% が学習セットに含まれる**。
- **within(target_only)/mixed**: train=`time_stratified_three_way_split`(全体70%)／eval=
  `data_time_split_by_subject`(被験者別後半20%) の**関数不一致**→ 重複 **69% / 61–78%**。
- **honest（重複0%）は Cross(source_only, 被験者非交差) と domain_train(同一分割) のみ**。
  matched-split の反実仮想は 0% 重複＝機構が分割不一致由来であることを証明。

### 実モデルでの分解（記録JSONを小数3桁再現）
| モデル | 記録値(leaked) | SEEN(学習済) | UNSEEN(正直) | 被験者holdout | cross-subj |
|---|---|---|---|---|---|
| RF top-10 | 0.765 | 0.848 | 0.650 | **0.517** | 0.495 |
| RF top-10+SMOTE | 0.758 | 0.830 | 0.658 | **0.525** | — |
| RF nofs(160) | 0.870 | **0.974** | 0.714 | **0.534** | 0.514 |

**正しく分離すると全て chance（0.49–0.53）、特徴数は無関係**。nofs が最良に見えたのは記憶容量が大きく
leaked 行をより完全に覚えるため。陽性対照: EEG バンドパワーは同 honest 分割で 0.62（>車両 0.53）＝
ハーネスは本物の信号を通す（車両 chance は検証アーティファクトではない）。

### honest な結論（論文の方向性）
1. **車両動特性 → EEG眠気(KSS) は honest 評価で全手法 chance（~0.50–0.53）**（RF 選択有無・SvmW・SvmA）。
   §2 の pooled/within/mixed の全数値は分割不一致による水増し。[[project_exp2_rf_087_unreproducible]] と整合。
2. **唯一の実信号は Lstm（event_label=DRT イベント、別タスク）**。honest cross-subject で Within/Cross
   ≈ 0.72–0.75 と domain 不変に生存（EEG眠気ではなく DRT 検出）。pooled Lstm 0.51 は無対策設定。
3. **先行研究(IV2025/TIV2026/exp2 0.89)の高 AUROC は同じ train/eval 分割不一致のアーティファクト**の可能性
   （verification_log 2026-07-01 が within/mixed について既に記録、比較性のため意図的に残していた機構）。
4. **要確認**: 出荷 KSS ラベルの陽性対照は 0.56–0.62 止まり（`kss.py` 整合問題、spearman 0.157）。
   車両=chance の結論は不変だが「ハーネスが信号検出」の強い対照にはラベル整合の確認が必要。

### 【直接実測】within-domain 脱リーク再評価 — 論文の目玉「SvmW 回復」も leak（2026-07-19, task2）
保存済み c1 within(target_only,in_domain) モデルを、(A)leaked eval で再現→記録値と一致で忠実性確認、
(B)モデル自身の train と**同一 `time_stratified_three_way_split` の held-out test（disjoint）**で honest 測定:

| 手法 | LEAKED再現 (=記録値) | **HONEST-matched（脱リーク）** |
|---|---|---|
| RF within-in | 0.77（記録 0.746） | **0.47（chance）** |
| SvmW within-in | 0.77（記録 0.805） | **0.52（chance）** |
| SvmA within-in | — (cuml env 不在でロード不可) | Cross 0.51 + T1 無信号で chance 確定 |

→ **論文(TIV2026_exp3)の中心 finding「SW-SMOTE が SvmW を within で 0.52→0.80 に回復（信号あり）／SvmA は
不変（信号なし）」は leak アーティファクト**。honest では **SvmW も RF も within=chance**、"回復" も
"SvmW vs SvmA の解離" も消える（両方 chance）。honest な唯一の信号は Lstm（別ラベル=DRT event、Cross 0.73–0.75）。
（注: honest cross-subject は保存モデルが全44被験者を学習済みのため hold-out しても leak するので、被験者分離は
**再学習が必須**＝監査の GroupShuffleSplit 0.51／Cross 列 0.51 が正しい honest cross-subject 値。）

### 【honest 全条件・確定表】de-leaked 再評価（2026-07-20, task2 完了）
バグ排除後（下記注意）の clean な honest 値。位置指定・standalone 再学習で index 混入を排除、監査 agent3 と一致:

| 評価プロトコル | leaked(記録) | **HONEST(clean)** | 判定 |
|---|---|---|---|
| within(target_only) RF/SvmW | 0.77 / 0.80 | **0.47–0.52** | chance |
| pooled RF/SvmW | 0.76 / 0.71 | **0.51 / ~0.51** | chance |
| 被験者内-時系列（personalized, 同一ドライバー後半+gap）160/10特徴 | — | **0.496 / 0.515** | chance |
| cross-subject（新ドライバー）160/10特徴 | — | **0.518 / 0.507** | chance |
| （honest 陽性対照）EEG バンドパワー cross-subject | — | **0.62** | 信号あり=ハーネス妥当 |

→ **車両動特性→EEG眠気(KSS)は、あらゆる honest 分割（被験者内-時系列・被験者間）・特徴数（10/160）で chance
（~0.50–0.52）**。personalized（同一ドライバー）でも信号なし。leaked の高値(0.72–0.86)は全て train/eval 行重複。
唯一の実信号は Lstm（別ラベル=DRT event、Cross honest 0.73–0.75）。

> **⚠️ 方法論的注意（今回ハマった落とし穴・要記録）**: `data_time_split_by_subject` は **index をリセットして
> 返す**。返り値の `X_test.index` を元 df に `df.loc[]` すると**先頭（学習領域）行を誤取得**し、honest テストのつもりで
> 学習行を評価して 0.75 と水増しされる。honest 再評価は**位置(iloc/numpy)指定**か、split 関数に元 index を保持させる
> こと。監査の 0.52 が正、途中の 0.75 は本バグの産物（不採用）。

### 脱リーク方針（次アクション、要ユーザ判断）
train と eval で**同一分割**を使う（`evaluate.py` の eval 分割を train の held-out に一致させる／保存済み split を
再利用）。全セル（4手法×全条件）を honest に測り直し、§2 を再構築。**論文含意**: RQ1/中心主張を「leakage-free
評価では車両動特性は EEG眠気を chance 以上に予測せず（negative result）、先行の高値・"回復"・"解離" は
train/eval 分割不一致のアーティファクト。唯一の実信号は DRT-event 検出(Lstm、別ラベル)」へ再構成が必要。
旧・撤回主張の記録は以下。

---
（以下、07-19 に判明した当初の撤回メモ — 上記監査で詳細化・拡大された）

**リークの機構（実測で確定）**: pooled は **train=被験者内時系列分割（各被験者の前半70%）／eval=ランダム
15%**（[eval_pipeline.py:152](../../../../src/evaluation/eval_pipeline.py#L152)、`iv2025_baseline_launcher` が eval に
`--subject_wise_split` を渡さない）。両分割が独立なため **eval のテスト行の 69% が train の学習セットに
含まれる**。標準 RF での再現（common データ・KSS・同一160特徴）:

| 評価 | 160特徴 | top-10 |
|---|---|---|
| pipeline再現（train=時系列 / eval=ランダム）**全体** | **0.925** | 0.907 |
| ┗ **SEEN 行（学習済み）** | **1.000（完全記憶）** | 1.000 |
| ┗ **UNSEEN 行（未学習=正直）** | **0.505（chance）** | 0.448 |
| **正しく分離**（単一ランダム分割 train∩test=∅） | **0.524** | 0.507 |
| **被験者ホールドアウト（cross-subject）** | **0.529** | 0.506 |

**結論**:
1. **正しく train/test を分離すると RF は全特徴でも top-10 でも chance（~0.52）** = キャンペーンの中心所見
   「車両動特性→EEG眠気 ≈ chance」（[[project_exp2_rf_087_unreproducible]]）と一致。
2. **160特徴が top-10 より高く出たのは記憶容量が大きく leaked 行をより完全に記憶するため**で、実信号の差
   ではない（UNSEEN では両方 chance）。「特徴選択が RF を過小評価」は成立しない。
3. **この分割不一致リークは nofs 特有ではなく pooled 列全体に及ぶ**。iv25base/iv25smote の pooled 値
   （RF 0.738/0.748、SvmW 0.519/0.717 等）も同機構で水増しされている可能性が高く、**§2 の pooled 表・
   §3 の「RF は pooled で機能」中心主張は要再検証**（honest 分割では全手法 chance の見込み）。within/mixed
   も §3/§3.5 記載の時系列重複リークを持つため同様。
4. **教訓**: pooled を「honest 列」と扱ったのが誤り。train と eval の分割方式が一致しない限り、絶対 AUROC は
   リーク混入。IV2025/TIV2026 との**相対**比較（before/after）には使えるが、**絶対値を "実信号" と解釈しては
   いけない**。正直な絶対評価には train/eval で同一の被験者ホールドアウト（cross-subject）分割が必要。

## 3.7 実装見直し・脱リーク再テストの"構え"（2026-07-20, 完遂後に実行する準備）

### ★先行実証（方式1）— 完了・独立2エージェントで二重検証（2026-07-20, exp3 走行継続・非破壊）
保存済みモデルを (a) leaked eval で再現→**記録JSONを小数4桁で一致（忠実性ゲート合格）**、(b) honest 分割
（物理行=subject_id+Timestamp で**学習行と 0% 重複を確認**）で再スコア:

| model | cell | recorded(=leaked) | **honest(disjoint)** | 学習行重複 |
|---|---|---|---|---|
| RF | within-in | 0.773–0.800 | **0.472–0.475** | leaked 69% → honest **0%** |
| RF | **pooled** | 0.721–0.765 | **0.517–0.525** | leaked **59.6%** → honest 0% |
| SvmW | within-in | 0.683–0.795 | **0.516–0.526** | leaked 69% → honest 0% |
| SvmW | pooled+SMOTE | 0.689–0.726 | **0.489–0.492** | leaked 59.6% → honest 0% |
| （陽性対照）EEG バンドパワー | 同 honest 分割 | — | **0.56–0.58（>車両0.52）** | — |

**確定**: leaked の高値は 59–69% の学習行再利用による記憶で、**物理行を 0% 重複にすると全 chance（0.47–0.53）**。
faithfulness 完全一致＋陽性対照合格＝「分割が信号を壊しただけ」ではない。**「RF は pooled で機能する唯一の手法」
は honest 評価で成立しない**（0.76→0.517=chance）。スクリプト: scratchpad `deleak_demo_A.py`／`adv_verify.py`。

**精緻化（検証で判明・要記録）**:
- **記録値はコード版に依存**: 一部の RF within JSON（2026-06-22 生成, `split_data_domain_train`=被験者ホールドアウト
  commit `466e5b1`）は当時 train/eval が**一致していて既に honest（重複0%, ~0.52）**。現行コード（06-27〜, within-subject
  commit `3e67282`）で再生成すると重複 68.6%・~0.58 に上昇。→ **§2 の数値は "生成時のコード版" によって leaked/honest が
  混在し得る**。脱リーク再評価では全セルを現行 honest 分割で**一括再生成**して統一する必要。
- **SvmW pooled（対策なし iv25base）は SMOTE 無しで縮退（全陽性, 0.52）＝元々 chance**。記憶で水増しされるのは
  SMOTE 版(0.72)。SvmW baseline は leaked/honest どちらでも chance。

### 計画（完遂後に本実行）
方針: **exp3 は現行プロトコルで完遂を最優先**（§2 主結果）。脱リーク再テストは**走行中実験・共有コードに
非破壊**で構え、完遂後に実行する。以下は準備物と手順（本節はチェックリスト、実行は保留）。

**A. 実装見直し（read-only、コード変更なし）**
1. `eval_pipeline.py:151-165`（pooled→random）／`138-149`（within/mixed→per-subject last20%）と train 側
   （`model_pipeline.py:108-191`）の分割不一致を、mode 別に「train partition vs eval-test partition の
   row 重複 %」で表化（監査 `overlap_audit.py` を正式スクリプト化：pooled ~60%・within ~69%・mixed 61–78%・
   Cross/domain_train 0%）。→ 「どの列がどれだけ leak か」を数値で確定。
2. 陽性対照（EEG→KSS）とラベル整合（`kss.py`、spearman 0.157 の懸念）を別途確認し、「ハーネスは信号を通す」
   ことを担保（現状 cross-subject 0.62）。

**B. 脱リーク再テスト（完遂後に実行、既存成果物は不変）**
- 方式1（最小・安全）: **保存済みモデルを honest 分割の held-out で再スコア**する standalone（`honest_*.py` を
  正式化）。⚠️ `data_time_split_by_subject` は index リセット→**位置(iloc)指定必須**（§3.6 の落とし穴）。
  4手法×全条件で honest 表を生成。SvmA は cuml 環境が要る（Cross+T1 で代替可）。
- 方式2（本格）: `evaluate.py` に **`--honest_split`（train と同一分割の held-out を eval に使う）フラグを新設**し、
  **新タグ `honest_` で別途 eval を回す**（既存 leaked JSON は温存、比較用に併記）。共有コードは追加のみ・
  デフォルト挙動不変なので走行中実験に非干渉。
- 期待結果（監査より）: 車両→EEG(KSS) は honest で全手法 chance ~0.52、Lstm(DRT) のみ Cross honest 0.73–0.75。

**C. 論文への反映（草稿 TIV2026_exp3、判断保留）**: 主結果は IV2025/TIV2026 準拠で提示し、**Limitation/
Robustness 節に脱リーク感度解析（§3.6/§3.7-B）を明記**する構成が、比較性と誠実性を両立する最有力案。

**次アクション（要ユーザ判断）**: exp3 完遂（c1 SvmW 残・RF-nofs 残）を待ってから B を実行。急ぐ場合は
B方式1を今すぐ 1〜2 セルで実証も可（非破壊）。

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
  **seed は後に 5 に削減（ユーザ判断）**。**⚠️ 結果の解釈は撤回（2026-07-19）**: pooled 5/5=0.864 は
  実信号ではなく **train/eval 分割不一致の記憶リーク**（正しく分離すると chance ~0.52、§3.6）。この検証を
  通じて **pooled 列全体のリーク**が判明したのが実質的な成果。
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

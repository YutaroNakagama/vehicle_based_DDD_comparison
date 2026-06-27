# exp3 検証・解析タスク (compute-PC で実行)

*作成: 2026-06-27. 解析機(WSL2/HPC, `data/processed` + eval JSON あり)で実行する想定。*
*背景: 各手法の考察(救済/非救済/頑健/別ターゲット)が「我々の想定どおり正しいか」を確定するための、
未実施の検証と図表。根拠は [`domain_imbalance_factor_analysis.md`](domain_imbalance_factor_analysis.md)。*

---

## P0(最優先 — 主結論の妥当性を左右)

### T1. SvmA を Arefnezhad 完全特徴セットで再テスト(faithfulness バグの解消)
**問題(確定):** SvmA の特徴フィルタが Arefnezhad(2019)の18特徴と不一致。
- `SVMA_PAPER_FEATURE_SUFFIXES`(14種, [`SvmA.py:65-70`](../../../../src/models/architectures/SvmA.py#L65))でフィルタ([`SvmA.py:664-684`](../../../../src/models/architectures/SvmA.py#L664))。
- **欠落(8種)**: Sample Entropy, Katz Fractal Dimension, Shannon Entropy, Spectral Flux, Frequency Variability, Q1/Q2/Q3。
- **余分(4種)**: Mean, Variance, Max, Min(Arefnezhad に無い)。
- **致命的**: Sample Entropy は原論文の最終選択5特徴のうち2つ(I11ᵃ, I11ᵛ)。Katz FD・Shannon Entropy も主力。
- これらは **`simlsl.py` で計算済み**(SampleEntropy `:215`, KatzFractal `:207`, ShannonEntropy `:208`, SpectralFlux `:211`, Quartile `:202/204`)なのに **SvmA がフィルタで捨てている**。Frequency Variability のみ未計算(0 hits)。

**現状の「信号なし」(univariate 0.515 / multivariate 0.509)は、原論文の主力特徴を除外した特徴セットでの値**
→ 忠実な Arefnezhad 再現ではない。この null は現状 airtight でない。

**やること:**
1. `SVMA_PAPER_FEATURE_SUFFIXES` を Arefnezhad の18種に修正
   (追加: SampleEntropy, KatzFractal, ShannonEntropy, SpectralFlux, Q1/Q2/Q3 / 削除: Mean, Variance, Max, Min)。
   - 正確な列名は処理済み CSV ヘッダで確認。Frequency Variability は未計算なので必要なら `simlsl.py` に追加。
2. 再実行:
   - **univariate directionless AUROC**(全特徴、特に Sample Entropy / Katz FD)
   - **multivariate RBF-SVM**(完全特徴セット、被験者分離 split)
   - **SvmA(ANFIS+PSO 選択込み)** を B1 条件で
3. **期待/判定**:
   - 依然 chance(<0.55) → 「信号なし」が**強化**(原論文特徴込みでも chance)→ 考察確定。
   - 信号が出る → 元の null は**特徴フィルタの産物** → 考察を修正。

### T2. SvmW clean-split 検証(0.79 が honest signal か split 依存か)
**問題:** SvmW の8 wavelet 帯域は univariate 0.510 / **multivariate random-split 0.485(chance)**。
にもかかわらず B1(target_only split)で 0.79。→ **0.79 は B1 split 構造(時間/被験者)に依存している疑い**。
「8帯域が潜在 drowsiness 信号を持つ」は現状未証明・反証寄り。

**やること:**
- B1 と同条件(SvmW, in_domain, SW-SMOTE 0.3/0.5, 同 seed)で、`split_data` の
  `subject_split_strategy` を **被験者ホールドアウト(subject-disjoint random split)** に変えて AUROC を取得。
  (現行の `subject_time_split` = 被験者順ソート単一カットでなく、被験者単位で train/test を分離)
- **期待/判定**:
  - 0.79 が残る → 特徴に本物の(多変量)信号 → 「潜在信号を持つ」OK。
  - chance に落ちる → 「SMOTE は decision function を回復させるが within-domain regime 特有の構造に乗るだけ」と書換え。
- これは **within-domain明示の self-consistency**(leakage 批判との整合)も同時に決める。

---

## P1(主結論を補強)

### T3. RF の SMOTE 単独効果(2×2 の1セル)
- `pooled + SMOTE` を実行し `pooled + baseline`(=IV2025)と比較 → RF で「SMOTE 効果小」を交絡なしに確認。

### T4. SvmA の分類器×特徴 deconfound
- **RF-on-SvmA特徴**(SvmA の23/完全特徴を RF で学習)と **RBF-SVM-on-RF特徴** を同 split で。
- 期待: RF-on-SvmA特徴 ≈ chance(特徴が壁)/ SVM-on-RF特徴 ≈ 0.78(分類器は壁でない)→ SvmA の null が分類器/選択に依らないことを確定。

### T5. Lstm の domain 帰属を確定
- **local before(IV2025 pooled)を完走** → 公表値0.52を実測へ置換(SvmA before も同様、〜7/1見込み)。
- **Lstm cross-domain(ω=0)** を測定 → within ≫ cross なら「向上は domain 由来」を確定。

### T6. road-curve 除去の faithfulness(SvmA 補助)
- Arefnezhad Eq.1-2(steering の sliding-window 平均を引き道路ジオメトリ除去)は**未実装**。
- Aygun のコースが曲線を含むなら steering が道路形状で汚染され得る。除去を入れて T1 を再確認するか、
  Aygun のコースが直線/該当しないことを確認して「不要」と注記。

---

## 生成すべきプロット(データ機で出力 → 論文/スライド用)

| # | プロット | 何を示すか | 優先 |
|---|---|---|---|
| 1 | SvmA per-feature univariate AUROC(棒): 完全18 vs 現行14 | 欠落特徴(Sample Entropy 等)の信号有無 → T1 | P0 |
| 2 | SvmW: target_only-split vs clean-split AUROC(seed 箱ひげ) | 0.79 が honest か split 依存か → T2 | P0 |
| 3 | predict_proba ヒストグラム: SvmW IV25(定数スパイク)/ B1(spread)/ SvmA B1(両クラス重なり) | degeneracy→回復の視覚証拠 | P1 |
| 4 | before/within/cross AUROC(手法別グループ棒) | within-domain明示(within回復・cross崩落) | P1 |
| 5 | seed 収束: running-mean AUROC ±95%CI vs k(手法別) | seed数の正当化(exp3 では未保存) | P1 |
| 6 | 混同行列ヒートマップ(条件別) | 崩壊/非崩壊の一目証拠 | P2 |

---

## 完了後に確定する考察(判定表)

| 考察 | 現状 | T で確定 |
|---|---|---|
| RF: 不均衡で崩壊しない | ✅ 確認済 | — |
| RF: SMOTE 効果小 | 🟡 交絡 | T3 |
| SvmA: 特徴に信号なし | ⚠️ **特徴セット不忠実** | **T1**(+T4, T6) |
| SvmW: SMOTE で決定境界回復 | ✅ 確認済 | — |
| SvmW: 8帯域が潜在信号を持つ | ❌ 反証寄り | **T2** |
| Lstm: 均衡ゆえ SMOTE 無効 | ✅ 確認済 | — |
| Lstm: 向上は domain 由来 | 🟡 方向のみ | T5 |

---

## 参照(コード)
- SvmA 特徴フィルタ: [`src/models/architectures/SvmA.py:65-70, 664-684`](../../../../src/models/architectures/SvmA.py#L65)
- 特徴計算(欠落特徴は計算済み): [`src/data_pipeline/features/simlsl.py:194-215`](../../../../src/data_pipeline/features/simlsl.py#L194)
- split 切替: [`src/utils/io/split_helpers.py`](../../../../src/utils/io/split_helpers.py)(`subject_time_split` ↔ 被験者ホールドアウト)
- 機序根拠: [`domain_imbalance_factor_analysis.md`](domain_imbalance_factor_analysis.md) §2, §9

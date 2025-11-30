# 不均衡データ対策手法

本プロジェクトで実装・評価した不均衡データ対策手法をまとめます。

## 概要

居眠り運転検知タスクでは、**正例（居眠り）が約3.9%** という深刻なクラス不均衡が存在します。
この問題に対処するため、以下の3カテゴリの手法を実装・比較しました。

| カテゴリ | 手法 | データ量変化 |
|---------|------|-------------|
| **オーバーサンプリング** | SMOTE, SMOTE+Tomek, SMOTE+ENN, SMOTE+RUS | 増加 |
| **アンダーサンプリング** | RUS, Tomek Links | 減少 |
| **データ拡張** | Jittering + Scaling | 増加 |
| **アンサンブル** | BalancedRF, EasyEnsemble | 変化なし |

---

## 1. オーバーサンプリング手法

### 1.1 SMOTE (Synthetic Minority Over-sampling Technique)

少数クラスのサンプル間を線形補間して合成サンプルを生成する手法。

**アルゴリズム:**
1. 少数クラスのサンプル $x_i$ を選択
2. $k$近傍の少数クラスサンプル $x_{nn}$ を選択
3. 合成サンプル: $x_{new} = x_i + \lambda (x_{nn} - x_i)$, where $\lambda \in [0, 1]$

**参考文献:**
> Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002).
> **SMOTE: Synthetic Minority Over-sampling Technique.**
> *Journal of Artificial Intelligence Research*, 16, 321-357.
> https://doi.org/10.1613/jair.953

**実装:** `imblearn.over_sampling.SMOTE`

---

### 1.2 SMOTE + Tomek Links

SMOTEでオーバーサンプリング後、Tomek Linksで境界ノイズを除去するハイブリッド手法。

**Tomek Links の定義:**
サンプルペア $(x_i, x_j)$ がTomek Linkであるとは、$x_i$ と $x_j$ が異なるクラスに属し、
かつ $d(x_i, x_j) < d(x_i, x_k)$ かつ $d(x_i, x_j) < d(x_j, x_k)$ を満たす $x_k$ が存在しないこと。

**参考文献:**
> Batista, G. E., Prati, R. C., & Monard, M. C. (2004).
> **A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data.**
> *ACM SIGKDD Explorations Newsletter*, 6(1), 20-29.
> https://doi.org/10.1145/1007730.1007735

**実装:** `imblearn.combine.SMOTETomek`

---

### 1.3 SMOTE + ENN (Edited Nearest Neighbours)

SMOTEでオーバーサンプリング後、ENNでより積極的にノイズサンプルを除去。

**ENNアルゴリズム:**
各サンプルについて、$k$近傍の多数決でクラスを決定し、実際のラベルと異なる場合は除去。

**参考文献:**
> Wilson, D. L. (1972).
> **Asymptotic Properties of Nearest Neighbor Rules Using Edited Data.**
> *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-2(3), 408-421.
> https://doi.org/10.1109/TSMC.1972.4309137

> Batista, G. E., Prati, R. C., & Monard, M. C. (2004).
> **A Study of the Behavior of Several Methods for Balancing Machine Learning Training Data.**
> *ACM SIGKDD Explorations Newsletter*, 6(1), 20-29.

**実装:** `imblearn.combine.SMOTEENN`

---

### 1.4 SMOTE + RUS (Random Under-Sampling)

SMOTEで少数クラスを増やした後、RUSで多数クラスを削減するハイブリッド手法。

**本実装の設定:**
- SMOTE: 少数クラスを多数クラスの50%まで増加
- RUS: 最終的に少数:多数 = 0.8:1 に調整

**参考文献:**
> Chawla, N. V., et al. (2002). SMOTE. *JAIR*, 16, 321-357.

**実装:** `imblearn.pipeline.Pipeline` with SMOTE + RandomUnderSampler

---

## 2. アンダーサンプリング手法

### 2.1 Random Under-Sampling (RUS)

多数クラスからランダムにサンプルを削除してクラスバランスを調整。

**メリット:**
- シンプルで高速
- 実データのみ使用（合成データなし）

**デメリット:**
- 情報損失の可能性
- データ量が激減する場合がある

**参考文献:**
> He, H., & Garcia, E. A. (2009).
> **Learning from Imbalanced Data.**
> *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
> https://doi.org/10.1109/TKDE.2008.239

**実装:** `imblearn.under_sampling.RandomUnderSampler`

---

### 2.2 Tomek Links (単独使用)

決定境界付近のノイズとなる多数クラスサンプルのみを除去。

**特徴:**
- 大幅なダウンサンプリングは行わない（境界サンプルのみ削除）
- 決定境界をクリーンにする効果

**参考文献:**
> Tomek, I. (1976).
> **Two Modifications of CNN.**
> *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-6(11), 769-772.
> https://doi.org/10.1109/TSMC.1976.4309452

**実装:** `imblearn.under_sampling.TomekLinks`

---

## 3. 時系列データ拡張手法

### 3.1 Jittering + Scaling

時系列データの特性を考慮したデータ拡張手法。SMOTEのような特徴空間での補間ではなく、
現実的なノイズ・変動をシミュレート。

**Jittering:**
特徴量にガウシアンノイズを追加。
$$x_{aug} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

**Scaling:**
振幅変動をシミュレート。
$$x_{aug} = x \cdot s, \quad s \sim \mathcal{N}(1, \sigma_s^2)$$

**本実装の設定:**
- `jitter_sigma = 0.03` (特徴量の標準偏差に対する比率)
- `scale_sigma = 0.1` (スケーリング係数の標準偏差)

**参考文献:**
> Um, T. T., Pfister, F. M., Pichler, D., Endo, S., Lang, M., Hirche, S., ... & Kulić, D. (2017).
> **Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional Neural Networks.**
> *Proceedings of the 19th ACM International Conference on Multimodal Interaction (ICMI)*, 216-220.
> https://doi.org/10.1145/3136755.3136817

**実装:** `src/data_pipeline/augmentation.py`

---

## 4. アンサンブル手法

### 4.1 Balanced Random Forest

各決定木の学習時にブートストラップサンプリングでクラスバランスを調整。

**アルゴリズム:**
各木について、少数クラスの全サンプルと、同数の多数クラスサンプルをランダムに選択して学習。

**参考文献:**
> Chen, C., Liaw, A., & Breiman, L. (2004).
> **Using Random Forest to Learn Imbalanced Data.**
> *University of California, Berkeley*, Technical Report 666.
> https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf

**実装:** `imblearn.ensemble.BalancedRandomForestClassifier`

---

### 4.2 EasyEnsemble

複数のAdaBoost分類器をアンサンブル。各分類器は異なるアンダーサンプリングされたデータセットで学習。

**アルゴリズム:**
1. 多数クラスを $T$ 個のサブセットにランダム分割
2. 各サブセットと少数クラス全体でAdaBoost分類器を学習
3. $T$ 個の分類器の予測を統合

**参考文献:**
> Liu, X. Y., Wu, J., & Zhou, Z. H. (2008).
> **Exploratory Undersampling for Class-Imbalance Learning.**
> *IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics)*, 39(2), 539-550.
> https://doi.org/10.1109/TSMCB.2008.2007853

**実装:** `imblearn.ensemble.EasyEnsembleClassifier`

---

## 5. 評価指標

不均衡データでは、Accuracyは不適切な指標となるため、以下の指標を使用：

| 指標 | 説明 | 特徴 |
|------|------|------|
| **AUPRC (PR-AUC)** | Precision-Recall曲線下面積 | 閾値非依存、不均衡に最適 |
| **F2 Score** | Recall重視のF-measure ($\beta=2$) | 見逃し削減を重視 |
| **Recall** | 真陽性率 (感度) | 居眠り検出率 |
| **Precision** | 適合率 | 誤警報率の逆数 |

**F-beta Score の定義:**
$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

**参考文献:**
> Saito, T., & Rehmsmeier, M. (2015).
> **The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.**
> *PLOS ONE*, 10(3), e0118432.
> https://doi.org/10.1371/journal.pone.0118432

---

## 6. 実験結果サマリ

### 実装済み手法と対応するジョブID

| 手法 | Job ID | タグ |
|------|--------|------|
| Baseline RF (class_weight) | 14468417 | `imbal_v2_baseline` |
| SMOTE + Tomek | 14468418 | `imbal_v2_smote_tomek` |
| SMOTE + ENN | 14468419 | `imbal_v2_smote_enn` |
| SMOTE + RUS | 14468421 | `imbal_v2_smote_rus` |
| EasyEnsemble | 14468501 | `imbal_v2_easyensemble` |
| Jittering + Scaling | 14471460 | `imbal_v2_jitter_scale` |
| Undersample RUS | 14471478 | `imbal_v2_undersample_rus` |
| Undersample Tomek | 14471479 | `imbal_v2_undersample_tomek` |

---

## 7. 推奨読み物

### サーベイ論文

> He, H., & Garcia, E. A. (2009).
> **Learning from Imbalanced Data.**
> *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

> Krawczyk, B. (2016).
> **Learning from Imbalanced Data: Open Challenges and Future Directions.**
> *Progress in Artificial Intelligence*, 5(4), 221-232.
> https://doi.org/10.1007/s13748-016-0094-0

### 時系列データ拡張

> Iwana, B. K., & Uchida, S. (2021).
> **An Empirical Survey of Data Augmentation for Time Series Classification with Neural Networks.**
> *PLOS ONE*, 16(7), e0254841.
> https://doi.org/10.1371/journal.pone.0254841

### 不均衡データの深層学習

> Johnson, J. M., & Khoshgoftaar, T. M. (2019).
> **Survey on Deep Learning with Class Imbalance.**
> *Journal of Big Data*, 6(1), 1-54.
> https://doi.org/10.1186/s40537-019-0192-5

---

## 8. 関連ファイル

- **Augmentation実装:** `src/data_pipeline/augmentation.py`
- **Training Pipeline:** `src/models/architectures/common.py`
- **CLI Helper:** `src/utils/cli/train_cli_helpers.py`
- **可視化:** `src/analysis/imbalance_analysis.py`
- **HPCスクリプト:** `scripts/hpc/imbalance_comparison_v2/`

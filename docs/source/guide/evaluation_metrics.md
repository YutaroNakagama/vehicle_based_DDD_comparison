# 時系列データ異常検知の評価指標

このドキュメントでは、時系列データ異常検知（居眠り運転検知など）で使用される評価指標について解説します。

## 概要

不均衡データにおける異常検知では、指標の選択が非常に重要です。
本プロジェクトでは陽性率（Drowsy）が約4%と低いため、適切な指標を選択する必要があります。

## 推奨指標の優先順位

| 優先度 | 指標 | 用途 |
|--------|------|------|
| ⭐⭐⭐ | **AUPRC** | モデル比較の主指標（閾値非依存） |
| ⭐⭐⭐ | **Recall** | 異常検出率（見逃し防止） |
| ⭐⭐ | **F2 Score** | Recall重視のバランス指標 |
| ⭐ | Precision | 誤報率の評価 |
| ⭐ | F1 Score | バランス指標（Recall/Precision同等重視） |
| △ | AUROC | 不均衡データでは過大評価されやすい |
| △ | Accuracy | 不均衡データでは不適切 |

---

## 各指標の詳細

### 1. Accuracy（正解率）

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**特徴:**
- 全予測のうち正解した割合
- **不均衡データでは不適切** - 多数派クラスを全て正解するだけで高い値になる

**例（本プロジェクト）:**
- 陽性率4%のデータで「全て陰性」と予測 → Accuracy = 96%
- 見かけ上高いが、異常検知としては無価値

**推奨:** 参考値としてのみ使用。主指標にしない。

---

### 2. Precision（適合率）

$$\text{Precision} = \frac{TP}{TP + FP}$$

**特徴:**
- 陽性と予測したもののうち、実際に陽性だった割合
- **誤報（False Positive）の少なさ**を評価
- 「オオカミ少年」を避けたい場合に重要

**ユースケース:**
- 誤報コストが高い場合（製造ライン停止など）
- ユーザー体験を重視する場合

**本プロジェクトでの値:** 約4%（低い = 多くの誤報がある）

---

### 3. Recall（再現率・感度）

$$\text{Recall} = \frac{TP}{TP + FN}$$

**特徴:**
- 実際の陽性のうち、正しく検出できた割合
- **見逃し（False Negative）の少なさ**を評価
- **安全クリティカルな異常検知で最重要**

**ユースケース:**
- 居眠り運転検知 - 見逃しが致命的
- 医療診断 - 病気の見逃しは危険
- セキュリティ - 侵入の見逃しは重大

**本プロジェクトでの目標:** 50%以上

---

### 4. F1 Score

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**特徴:**
- PrecisionとRecallの調和平均
- 両者を同等に重視
- 不均衡データではRecallが犠牲になりやすい

**推奨:** RecallとPrecisionを同程度に重視する場合に使用

---

### 5. F2 Score ⭐推奨

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

$\beta = 2$ の場合、**Recallを2倍重視**します。

**特徴:**
- 見逃しを減らすことを優先
- 安全系アプリケーションで広く使用
- F1よりも異常検知に適した指標

**本プロジェクトでの使用:** 閾値最適化の目的関数として使用

---

### 6. AUROC（Area Under ROC Curve）

$$\text{AUROC} = \int_0^1 \text{TPR}(FPR) \, d(\text{FPR})$$

**特徴:**
- ROC曲線の下の面積
- 閾値に依存しない総合評価
- 0.5 = ランダム、1.0 = 完璧

**問題点（不均衡データ）:**
- **True Negative Rate（TNR）が高いと過大評価**
- 多数派クラスが多いほどFPRが低くなりやすい
- 不均衡データでは0.9以上でも実用的でない場合がある

**推奨:** 参考値として使用。AUPRCを優先。

---

### 7. AUPRC（Area Under Precision-Recall Curve）⭐⭐⭐最重要

$$\text{AUPRC} = \int_0^1 \text{Precision}(Recall) \, d(\text{Recall})$$

**特徴:**
- PR曲線の下の面積
- **不均衡データで最も信頼性の高い指標**
- ランダム分類器のAUPRC = 陽性率（本プロジェクトでは約4%）

**なぜAUPRCが重要か:**
1. True Negativeに影響されない
2. 少数派クラスの検出性能を正確に反映
3. 閾値選択前のモデル比較に最適

**本プロジェクトでの値:**
- ランダムベースライン: 0.039（3.9%）
- 現在の最高値: 0.073（SMOTE+RUS）

**解釈の目安:**
| AUPRC | 評価 |
|-------|------|
| < 0.039 | ランダム以下 |
| 0.039-0.06 | 改善の余地あり |
| 0.06-0.10 | やや有効 |
| 0.10-0.20 | 有効 |
| > 0.20 | 良好 |

---

## 指標選択のガイドライン

### ユースケース別推奨指標

| ユースケース | 主指標 | 副指標 | 理由 |
|-------------|--------|--------|------|
| **安全クリティカル**（居眠り検知、医療） | Recall, F2 | AUPRC | 見逃しが致命的 |
| **コスト重視**（製造業異常検知） | Precision, F1 | AUPRC | 誤報による停止コスト |
| **モデル比較**（研究開発） | AUPRC | Recall, F2 | 閾値非依存の総合評価 |
| **運用監視** | Recall@固定FPR | - | 実運用の制約を反映 |

### 本プロジェクトでの評価方針

1. **モデル選択**: AUPRCで比較
2. **閾値最適化**: F2スコアを最大化
3. **最終評価**: Recall（検出率）を報告

---

## Confusion Matrix（混同行列）

```
                 予測
              Negative  Positive
実際 Negative    TN        FP      ← 誤報
     Positive    FN        TP      ← 見逃し
```

- **TP (True Positive)**: 正しく陽性と予測
- **TN (True Negative)**: 正しく陰性と予測
- **FP (False Positive)**: 誤報（陰性を陽性と誤予測）
- **FN (False Negative)**: 見逃し（陽性を陰性と誤予測）

**異常検知での重要度:**
- FN（見逃し）> FP（誤報）の場合 → Recall重視
- FP（誤報）> FN（見逃し）の場合 → Precision重視

---

## 閾値の選択

確率的分類器は閾値によって性能が変化します。

### 閾値選択の方法

1. **デフォルト（0.5）**: 一般的だが、不均衡データでは不適切
2. **F2最大化**: Recall重視の閾値を自動選択
3. **Recall固定**: 目標Recall（例: 80%）を達成する閾値を選択
4. **コスト最小化**: 誤報・見逃しのコストを定義して最小化

### 本プロジェクトでの閾値選択

```python
# F2スコアを最大化する閾値を検証データで選択
from sklearn.metrics import fbeta_score
best_threshold = max(thresholds, key=lambda t: fbeta_score(y_val, y_pred > t, beta=2))
```

---

## 参考文献

1. Davis, J., & Goadrich, M. (2006). "The relationship between Precision-Recall and ROC curves." *ICML*.
2. Saito, T., & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." *PLOS ONE*.
3. He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data." *IEEE TKDE*.

---

## 関連ドキュメント

- [不均衡データ対策手法](imbalance_methods.md)
- [評価パイプライン](evaluation.rst)

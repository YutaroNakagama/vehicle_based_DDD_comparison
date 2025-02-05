import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import time
from scipy.stats import ttest_ind

# データの読み込み
X_data = pd.read_csv('X.csv').drop(columns=['Unnamed: 0'])
y_data = pd.read_csv('y.csv').drop(columns=['Unnamed: 0']).values.flatten()

# T検定でp値が0.05未満の特徴量を選択
X_data['label'] = y_data
X_label_1 = X_data[X_data['label'] == 1].drop(columns=['label'])
X_label_0 = X_data[X_data['label'] == 0].drop(columns=['label'])

print("feature selection")
selected_features = []
for feature in X_label_1.columns:
    t_stat, p_val = ttest_ind(X_label_1[feature], X_label_0[feature], equal_var=False)
    if p_val < 0.05:
        selected_features.append(feature)

# 選択した特徴量のみで再構成
X_data = X_data[selected_features]

# 5-foldクロスバリデーションの設定
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_per_fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []
auc_per_fold = []
time_per_fold = []

print("5 fold cross val")
for train_index, test_index in kf.split(X_data):
    X_train, X_test = X_data.values[train_index], X_data.values[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    
    # SVMの学習と実行時間計測
    svm = SVC(kernel='rbf', probability=True)  # RBFカーネルを使用
    start_time = time.time()
    svm.fit(X_train, y_train)
    end_time = time.time()
    running_time = (end_time - start_time) * 1000
    
    # テストデータでの予測
    y_pred = svm.predict(X_test)
    y_pred_prob = svm.predict_proba(X_test)[:, 1]
    
    # 各評価指標の計算
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_pred_prob) * 100
    
    # 評価結果をリストに追加
    accuracy_per_fold.append(accuracy)
    precision_per_fold.append(precision)
    recall_per_fold.append(recall)
    f1_per_fold.append(f1)
    auc_per_fold.append(auc)
    time_per_fold.append(running_time)
    
    # 各foldごとの結果を表示
    print(f"Fold - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, "
          f"F1 Score: {f1:.2f}%, AUC: {auc:.2f}%, Running Time: {running_time:.2f} ms")

# 各foldの評価指標の平均を表示
print("\nAverage results across folds:")
print(f"  Accuracy: {np.mean(accuracy_per_fold):.2f}%")
print(f"  Precision: {np.mean(precision_per_fold):.2f}%")
print(f"  Recall: {np.mean(recall_per_fold):.2f}%")
print(f"  F1 Score: {np.mean(f1_per_fold):.2f}%")
print(f"  AUC: {np.mean(auc_per_fold):.2f}%")
print(f"  Running Time: {np.mean(time_per_fold):.2f} ms")

# 最後のfoldのROC曲線を表示
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


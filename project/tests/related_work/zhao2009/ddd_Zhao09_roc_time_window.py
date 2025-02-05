import time
import csv
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from scipy.signal import convolve

# パフォーマンス計測開始
start_time = time.time()

# データ読み込み
data_raw_all_df = pd.read_csv('../dataset/data_raw_all_12.csv')

# 被験者リスト取得
unique_subjects = data_raw_all_df['subj'].unique()

# GHMフィルタ定義
ghm_lowpass = np.array([0.48296, 0.8365, 0.22414, -0.12941])
ghm_highpass = np.array([-0.12941, -0.22414, 0.8365, -0.48296])

# 波レットパケット分解関数
def wavelet_packet_decompose(data, lowpass_filter, highpass_filter):
    approx = convolve(data, lowpass_filter, mode='full')[::2]
    detail = convolve(data, highpass_filter, mode='full')[::2]
    return approx, detail

# 時間ウィンドウラベリングパラメータ
time_window = 10  # 600秒
overlap_ratio = 0.5  # 80%

# ラベリング結果を保存するリスト
labeling_results = []

# 各被験者のデータをスライディングウィンドウで処理
for subj in unique_subjects:
    print(subj)
    subject_data = data_raw_all_df[data_raw_all_df['subj'] == subj]
    time_values = subject_data['row LSL time'].values
    alpha_p_beta_values = subject_data['alpha_p_beta'].values

    step = int(time_window * (1 - overlap_ratio))  # ステップサイズ
    for start_idx in range(0, len(time_values) - time_window, step):
        print(start_idx)
        end_idx = start_idx + time_window
        window_data = alpha_p_beta_values[start_idx:end_idx]

        # 平均値計算とラベル付け
        mean_alpha_p_beta = np.mean(window_data)
        #label = 1 if mean_alpha_p_beta >= 0.90 else 0
        # 平均値計算とラベル付け
        mean_alpha_p_beta = np.mean(window_data)
        if mean_alpha_p_beta >= 0.85:
            label = 1
        elif mean_alpha_p_beta <= 0.85:
            label = 0
        else:
            continue  # 0.1 < alpha_p_beta < 0.9 の場合は捨てる

        # 結果を保存
        labeling_results.append({
            'subj': subj,
            'start_time': time_values[start_idx],
            'end_time': time_values[end_idx - 1],
            'mean_alpha_p_beta': mean_alpha_p_beta,
            'label': label
        })

# ラベリング結果をデータフレームに変換
time_window_labels_df = pd.DataFrame(labeling_results)

# ラベルを元に波レットエネルギー特徴量を計算
all_packet_results = []
print("wavelet")
for subj in unique_subjects:
    print(subj)
    subject_data = data_raw_all_df[data_raw_all_df['subj'] == subj]['Steering_Wheel_Pos'].values
    packet_result_subj = {'subj': subj}

    nodes = {'A': subject_data}
    for level in range(1, 4):
        new_nodes = {}
        for label, signal in nodes.items():
            approx, detail = wavelet_packet_decompose(signal, ghm_lowpass, ghm_highpass)
            new_nodes[f'{label}A'] = approx
            new_nodes[f'{label}D'] = detail
            packet_result_subj[f'{label}A_energy_level_{level}'] = np.sum(approx ** 2)
            packet_result_subj[f'{label}D_energy_level_{level}'] = np.sum(detail ** 2)
        nodes = new_nodes
    all_packet_results.append(packet_result_subj)

# 波レットエネルギー特徴量をデータフレームに変換
wavelet_energy_df = pd.DataFrame(all_packet_results)

# ラベルデータと結合
print('merge')
wavelet_energy_df_labeled = pd.merge(wavelet_energy_df, time_window_labels_df[['subj', 'label']], on='subj')

# 特徴量とラベルの選択
selected_features = ['AAAA_energy_level_3', 'AAAD_energy_level_3',
                     'AADA_energy_level_3', 'AADD_energy_level_3',
                     'ADAA_energy_level_3', 'ADAD_energy_level_3',
                     'ADDA_energy_level_3', 'ADDD_energy_level_3']

X = wavelet_energy_df_labeled[selected_features]
y = wavelet_energy_df_labeled['label']

# トレーニングと検証データに分割
print('split')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# SVMモデルの初期化と訓練
print('svm train')
svm_rbf_model = svm.SVC(kernel='rbf', C=300, random_state=42)
svm_rbf_model.fit(X_train, y_train)

# 予測と評価
print('evaluate')
train_accuracy = accuracy_score(y_train, svm_rbf_model.predict(X_train))
val_accuracy = accuracy_score(y_val, svm_rbf_model.predict(X_val))
val_conf_matrix = confusion_matrix(y_val, svm_rbf_model.predict(X_val))
classification_rep = classification_report(y_val, svm_rbf_model.predict(X_val))

# ハイパーパラメータ最適化
#grid_search = GridSearchCV(
#    svm.SVC(kernel='rbf', C=300, random_state=42),
#    param_grid={'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
#    cv=5, scoring='accuracy', n_jobs=-1
#)
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['rbf']  # 他のカーネルを試す場合はリストに追加
}
grid_search = GridSearchCV(
    svm.SVC(random_state=42),
    param_grid=param_grid,
    cv=5, scoring='accuracy', n_jobs=-1, verbose=2
)
print('grid')
grid_search.fit(X_train, y_train)

# 最適なパラメータとスコアを取得
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# 最適モデルでの評価
best_svm_model = grid_search.best_estimator_
y_val_scores = best_svm_model.decision_function(X_val)

# 精度と評価指標の計算
val_predictions = best_svm_model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
precision = classification_report(y_val, val_predictions, output_dict=True)['weighted avg']['precision']
recall = classification_report(y_val, val_predictions, output_dict=True)['weighted avg']['recall']
f1 = classification_report(y_val, val_predictions, output_dict=True)['weighted avg']['f1-score']
fpr, tpr, thresholds = roc_curve(y_val, y_val_scores)
roc_auc = auc(fpr, tpr)

# 結果を辞書形式で保存
results = {
    'Best Parameters': [best_params],
    'Best Grid Search Score': [best_score],
    'Validation Accuracy': [accuracy],
    'Validation Precision': [precision],
    'Validation Recall': [recall],
    'Validation F1 Score': [f1],
    'Validation AUC': [roc_auc]
}

# 結果をCSVに書き込み
output_file = 'model_evaluation_results.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(results.keys())
    writer.writerow(results.values())

print(f"評価結果を {output_file} に保存しました。")

# 最適化されたモデルで評価
best_svm_model = grid_search.best_estimator_
val_accuracy_best = accuracy_score(y_val, best_svm_model.predict(X_val))
classification_rep_best = classification_report(y_val, best_svm_model.predict(X_val))

# ROC曲線計算
y_val_scores = best_svm_model.decision_function(X_val)
fpr, tpr, thresholds = roc_curve(y_val, y_val_scores)
roc_auc = auc(fpr, tpr)

# ROC曲線データを保存
roc_data = pd.DataFrame({
    'False Positive Rate': fpr,
    'True Positive Rate': tpr,
    'Thresholds': thresholds
})
roc_data.to_csv('roc_time_window.csv', index=False)

# 結果の出力
print("Training Accuracy:", train_accuracy)
print("Validation Accuracy:", val_accuracy)
print("Validation Confusion Matrix:\n", val_conf_matrix)
print("Classification Report:\n", classification_rep)
print("Optimized Validation Accuracy:", val_accuracy_best)
print("Optimized Classification Report:\n", classification_rep_best)
print("ROC AUC:", roc_auc)

# 予測と評価
train_predictions = svm_rbf_model.predict(X_train)
val_predictions = svm_rbf_model.predict(X_val)

# 訓練データと検証データの混同行列
train_conf_matrix = confusion_matrix(y_train, train_predictions)
val_conf_matrix = confusion_matrix(y_val, val_predictions)

# 各種スコアの出力
print("Training Accuracy:", accuracy_score(y_train, train_predictions))
print("Validation Accuracy:", accuracy_score(y_val, val_predictions))
print("Training Confusion Matrix:\n", train_conf_matrix)
print("Validation Confusion Matrix:\n", val_conf_matrix)
print("Validation Classification Report:\n", classification_report(y_val, val_predictions))



print("Elapsed Time:", time.time() - start_time)



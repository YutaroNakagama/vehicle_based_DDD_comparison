# Import necessary libraries
import time
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
from scipy.signal import convolve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
#import pywt

start_time = time.time()

# Load the data
data_raw_all_df = pd.read_csv('../dataset/data_raw_all_12.csv')

# Extract unique subjects
unique_subjects = data_raw_all_df['subj'].unique()

# Define GHM-like filters as placeholders
ghm_lowpass = np.array([0.48296, 0.8365, 0.22414, -0.12941])
ghm_highpass = np.array([-0.12941, -0.22414, 0.8365, -0.48296])

# Function for wavelet packet decomposition
def wavelet_packet_decompose(data, lowpass_filter, highpass_filter):
    approx = convolve(data, lowpass_filter, mode='full')[::2]
    detail = convolve(data, highpass_filter, mode='full')[::2]
    return approx, detail

# Calculate wavelet packet energy for each 'subj'
all_packet_results = []
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

# Convert all packet results into a DataFrame
wavelet_energy_df = pd.DataFrame(all_packet_results)

# Calculate the mean of alpha_p_beta for each subject
alpha_p_beta_means = data_raw_all_df.groupby('subj')['alpha_p_beta'].mean().reset_index()

# Assign label based on the mean alpha_p_beta with a threshold of 0.95
alpha_p_beta_means['alpha_p_beta_label'] = alpha_p_beta_means['alpha_p_beta'].apply(lambda x: 1 if x >= 0.90 else 0)

# Count the number of subjects with label=1 and label=0 with the new threshold
label_counts_0_95 = alpha_p_beta_means['alpha_p_beta_label'].value_counts()
print(label_counts_0_95)

# Merge the updated labels with wavelet energy features for SVM input
wavelet_energy_df_labeled_0_95 = pd.merge(wavelet_energy_df, alpha_p_beta_means[['subj', 'alpha_p_beta_label']], on='subj')

# Select the specific wavelet energy features and labels for SVM
selected_features = ['AAAA_energy_level_3', 'AAAD_energy_level_3',
                     'AADA_energy_level_3', 'AADD_energy_level_3',
                     'ADAA_energy_level_3', 'ADAD_energy_level_3',
                     'ADDA_energy_level_3', 'ADDD_energy_level_3']

X = wavelet_energy_df_labeled_0_95[selected_features]
y = wavelet_energy_df_labeled_0_95['alpha_p_beta_label']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.7, random_state=42)

# Initialize and train the SVM model with RBF kernel, C=300
print('train svm')
svm_rbf_model = svm.SVC(kernel='rbf', C=300, random_state=42)
svm_rbf_model.fit(X_train, y_train)

# Predict labels train on training and validation sets
y_train_pred = svm_rbf_model.predict(X_train)
y_val_pred = svm_rbf_model.predict(X_val)

# Calculate training and validation accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

# Generate confusion matrices for training and validation sets
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
val_conf_matrix = confusion_matrix(y_val, y_val_pred)

# Classification report for validation set
classification_rep = classification_report(y_val, y_val_pred)

# Output results
print(
    "\nTraining Accuracy                  :\n", train_accuracy,
    "\nValidation Accuracy                :\n", val_accuracy,
    "\nTraining Confusion Matrix          :\n", train_conf_matrix,
    "\nValidation Confusion Matrix        :\n", val_conf_matrix,
    "\nValidation Classification Report   :\n", classification_rep
)

from sklearn.model_selection import GridSearchCV

print("------------------ gammma optimization --------------------")

# Define a parameter grid for gamma optimization
param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
        }

# Initialize SVM model with RBF kernel and fixed C=300
svm_rbf_for_optimization = SVC(kernel='rbf', random_state=42)

# Apply GridSearchCV for gamma optimization with 5-fold cross-validation
grid_search = GridSearchCV(
        svm_rbf_for_optimization, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best gamma parameter and score from grid search
best_gamma = grid_search.best_params_['gamma']
best_score = grid_search.best_score_

# Retrain the model with the best gamma on the full training data
best_svm_model = grid_search.best_estimator_
y_val_pred_best = best_svm_model.predict(X_val)

# Calculate validation accuracy and classification report with the optimized gamma
val_accuracy_best = accuracy_score(y_val, y_val_pred_best)
classification_rep_best = classification_report(y_val, y_val_pred_best)

print("best_gamma              ",best_gamma                )
print("val_accuracy_best       ",val_accuracy_best         )
print("classification_rep_best\n",classification_rep_best   )

# Predict labels on training and validation sets
y_train_pred = best_svm_model.predict(X_train)
y_val_pred = best_svm_model.predict(X_val)

# Calculate training and validation accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

# Generate confusion matrices for training and validation sets
train_conf_matrix = confusion_matrix(y_train, y_train_pred)
val_conf_matrix = confusion_matrix(y_val, y_val_pred)

# Classification report for validation set
classification_rep = classification_report(y_val, y_val_pred)

# Output results
print(
    "result after opt\n",
    "\nTraining Accuracy                  :\n", train_accuracy,
    "\nValidation Accuracy                :\n", val_accuracy,
    "\nTraining Confusion Matrix          :\n", train_conf_matrix,
    "\nValidation Confusion Matrix        :\n", val_conf_matrix,
    "\nValidation Classification Report   :\n", classification_rep
)

# サポートベクトルを取得
support_vectors = best_svm_model.support_vectors_

# Dual coefficients（サポートベクトルに関連する重み係数）を取得
dual_coefficients = best_svm_model.dual_coef_

# 各サポートベクトルの数とマージンに相当する情報の表示
#print("Number of Support Vectors for Each Class:", best_svm_model.n_support_)
#print("Support Vectors:\n", support_vectors)
#print("Dual Coefficients (Weight Coefficients):\n", dual_coefficients)



# Predict decision function scores for validation data
y_val_scores = best_svm_model.decision_function(X_val)

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_val, y_val_scores)
roc_auc = auc(fpr, tpr)

# ROC曲線データをCSVに保存
roc_data = pd.DataFrame({
    'False Positive Rate': fpr,
    'True Positive Rate': tpr,
    'Thresholds': thresholds
})
roc_data.to_csv('roc_each_case.csv', index=False)
print("ROC曲線データが 'roc_each_case.csv' に保存されました。")

import csv

# 評価指標を計算
val_predictions = best_svm_model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
precision = classification_report(y_val, val_predictions, output_dict=True)['weighted avg']['precision']
recall = classification_report(y_val, val_predictions, output_dict=True)['weighted avg']['recall']
f1 = classification_report(y_val, val_predictions, output_dict=True)['weighted avg']['f1-score']

# ROC曲線とAUCスコアを計算
y_val_scores = best_svm_model.decision_function(X_val)
fpr, tpr, thresholds = roc_curve(y_val, y_val_scores)
auc_score = auc(fpr, tpr)

# 評価結果を辞書に保存
evaluation_results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Value': [accuracy, precision, recall, f1, auc_score]
}

# CSVに書き込む
output_file = 'evaluation_metrics.csv'
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(evaluation_results.keys())  # ヘッダー
    writer.writerows(zip(*evaluation_results.values()))  # データ行

print(f"評価指標を {output_file} に保存しました。")

print("elapse time: ", start_time - time.time())

## Plot ROC curve
#plt.figure()
#plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver Operating Characteristic')
#plt.legend(loc='lower right')
#plt.show()

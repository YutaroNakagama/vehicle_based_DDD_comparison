import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

# List of all file paths with subject ID placeholders
subject_ids = [
    'S0116_1', 'S0113_1', 'S0120_1', 'S0116_2', 'S0139_1', 'S0120_2',
    'S0134_1', 'S0134_2', 'S0135_2', 'S0140_1', 'S0140_2', 'S0148_1',
    'S0155_1', 'S0171_1', 'S0171_2', 'S0172_1', 'S0174_2', 'S0178_1',
    'S0181_1', 'S0181_2', 'S0189_1', 'S0196_1', 'S0197_1', 'S0204_1'
]

# Constructing file paths using subject IDs
all_file_paths = [f'./merged_data_table_{subject_id}.csv' for subject_id in subject_ids]

# Function to load data and assign alertness levels based on Theta/Beta Ratio
def load_and_process_data(file_paths):
    datasets = []
    for path in file_paths:
        data = pd.read_csv(path)
        if 'Theta/Beta Ratio' in data.columns:
            theta_beta = data['Theta/Beta Ratio']
            quantiles = theta_beta.quantile([0.33, 0.67])
            data['Alertness_Level'] = pd.cut(
                theta_beta,
                bins=[-float('inf'), quantiles[0.33], quantiles[0.67], float('inf')],
                labels=['Low', 'Medium', 'High']
            )
            datasets.append(data)
    return datasets

# Load and process all datasets
datasets = load_and_process_data(all_file_paths)

# Prepare data for SVM by filtering out "Medium" level
def prepare_data(datasets):
    X_combined = pd.DataFrame()
    y_combined = pd.Series(dtype='int')
    for data in datasets:
        # Filter to include only "Low" and "High" levels
        data_filtered = data[data['Alertness_Level'].isin(['Low', 'High'])].copy()
        # Map labels to binary values: "Low" = 0, "High" = 1
        data_filtered['Alertness_Level'] = data_filtered['Alertness_Level'].map({'Low': 0, 'High': 1})
        # Separate features and target
        X = data_filtered.drop(columns=['Alertness_Level', 'Theta/Beta Ratio', 'Timestamp'], errors='ignore')
        y = data_filtered['Alertness_Level']
        X_combined = pd.concat([X_combined, X], axis=0)
        y_combined = pd.concat([y_combined, y], axis=0)
    return X_combined, y_combined

# Prepare combined data for training
X_combined, y_combined = prepare_data(datasets)

# Step 1: Calculate feature indices (Fisher, Correlation, T-test, Mutual Information)
X_low = X_combined[y_combined == 0]
X_high = X_combined[y_combined == 1]

fisher_index = [(X_high[col].mean() - X_low[col].mean()) / (X_high[col].std()**2 + X_low[col].std()**2) for col in X_combined.columns]
correlation_index = [np.cov(X_combined[col], y_combined)[0, 1] / (X_combined[col].std() * y_combined.std()) for col in X_combined.columns]
t_test_index = [abs(ttest_ind(X_low[col], X_high[col], equal_var=False)[0]) for col in X_combined.columns]
mutual_info_index = mutual_info_classif(X_combined, y_combined, discrete_features=False)

# Step 2: Create a DataFrame for the indices and calculate Importance Degree (ID) using fuzzy inference
feature_indices = pd.DataFrame({
    'Feature': X_combined.columns,
    'Fisher Index': fisher_index,
    'Correlation Index': correlation_index,
    'T-test Index': t_test_index,
    'Mutual Information Index': mutual_info_index
})

# Define Gaussian membership function
def gaussian_membership(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)

# Membership function parameters for L, M, H
centers = {'L': 0.2, 'M': 0.5, 'H': 0.8}
std_devs = {'L': 0.15, 'M': 0.15, 'H': 0.15}

importance_degrees = []
for idx, row in feature_indices.iterrows():
    fisher_max = max([gaussian_membership(row['Fisher Index'], centers[mf], std_devs[mf]) for mf in ['L', 'M', 'H']])
    corr_max = max([gaussian_membership(row['Correlation Index'], centers[mf], std_devs[mf]) for mf in ['L', 'M', 'H']])
    ttest_max = max([gaussian_membership(row['T-test Index'], centers[mf], std_devs[mf]) for mf in ['L', 'M', 'H']])
    mi_max = max([gaussian_membership(row['Mutual Information Index'], centers[mf], std_devs[mf]) for mf in ['L', 'M', 'H']])
    avg_max = (fisher_max + corr_max + ttest_max + mi_max) / 4
    importance_degrees.append(1 if avg_max >= 0.8 else 0.5 if avg_max >= 0.5 else 0)

feature_indices['Importance Degree (ID)'] = importance_degrees

# Step 3: Select features with ID >= 0.5 and perform SVM with RBF kernel
important_features = feature_indices[feature_indices['Importance Degree (ID)'] >= 0.5]['Feature']
X_important = X_combined[important_features]

X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split(
    X_important, y_combined, test_size=0.2, random_state=42, stratify=y_combined)

scaler_imp = StandardScaler()
X_train_imp_scaled = scaler_imp.fit_transform(X_train_imp)
X_test_imp_scaled = scaler_imp.transform(X_test_imp)

# Grid search for RBF SVM on selected features
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
svm_rbf_imp = SVC(kernel='rbf', random_state=42)
grid_search_imp = GridSearchCV(svm_rbf_imp, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search_imp.fit(X_train_imp_scaled, y_train_imp)

# Best model and evaluation
best_params_imp = grid_search_imp.best_params_
svm_rbf_optimized_imp = SVC(kernel='rbf', C=best_params_imp['C'], gamma=best_params_imp['gamma'], random_state=42)
svm_rbf_optimized_imp.fit(X_train_imp_scaled, y_train_imp)

train_accuracy_imp = svm_rbf_optimized_imp.score(X_train_imp_scaled, y_train_imp)
test_accuracy_imp = svm_rbf_optimized_imp.score(X_test_imp_scaled, y_test_imp)

y_pred_imp = svm_rbf_optimized_imp.predict(X_test_imp_scaled)
classification_report_imp = classification_report(y_test_imp, y_pred_imp, target_names=['Low', 'High'])
conf_matrix_imp = confusion_matrix(y_test_imp, y_pred_imp)

print("Best Parameters:", best_params_imp)
print("Training Accuracy:", train_accuracy_imp)
print("Test Accuracy:", test_accuracy_imp)
print("\nClassification Report:\n", classification_report_imp)
print("Confusion Matrix:\n", conf_matrix_imp)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_imp, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Optimized RBF SVM with Selected Features')
plt.show()


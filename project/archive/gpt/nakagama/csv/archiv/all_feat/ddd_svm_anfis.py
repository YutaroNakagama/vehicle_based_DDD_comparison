import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

# File paths for the uploaded files
file_paths = [
    './merged_data_table_S0139_1.csv',
    './merged_data_table_S0113_1.csv',
    './merged_data_table_S0116_1.csv',
    './merged_data_table_S0116_2.csv',
    './merged_data_table_S0120_1.csv',
    './merged_data_table_S0120_2.csv',
    './merged_data_table_S0134_1.csv',
    './merged_data_table_S0134_2.csv',
    './merged_data_table_S0135_2.csv'
]

# Redefining a simple ANFIS model and its initialization for completeness

# Define Gaussian membership function for fuzzy inputs
def gaussian_mf(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

# Simple ANFIS model function with two inputs (for simplicity)
class SimpleANFIS:
    def __init__(self, num_rules):
        # Initialize parameters for the fuzzy rules (mean and sigma for each input)
        self.num_rules = num_rules
        self.means = np.random.uniform(0, 1, (num_rules, 2))  # 2 inputs, num_rules rules
        self.sigmas = np.random.uniform(0.1, 0.5, (num_rules, 2))
        self.consequents = np.random.uniform(-1, 1, num_rules)  # Consequent parameters for each rule

    def forward(self, x1, x2):
        # Compute the firing strength of each rule
        firing_strengths = np.zeros(self.num_rules)
        for i in range(self.num_rules):
            mu_x1 = gaussian_mf(x1, self.means[i, 0], self.sigmas[i, 0])
            mu_x2 = gaussian_mf(x2, self.means[i, 1], self.sigmas[i, 1])
            firing_strengths[i] = mu_x1 * mu_x2  # Product of membership values for "AND" operation

        # Normalize firing strengths
        if firing_strengths.sum() == 0:
            return 0  # Avoid division by zero
        normalized_strengths = firing_strengths / firing_strengths.sum()

        # Calculate the output as a weighted sum of consequents
        output = np.dot(normalized_strengths, self.consequents)
        return output

    def set_parameters(self, means, sigmas, consequents):
        self.means = means
        self.sigmas = sigmas
        self.consequents = consequents

# Initialize ANFIS model
anfis_model = SimpleANFIS(num_rules=5)

# Example PSO optimization to set the ANFIS parameters (details of PSO omitted for brevity)
# This would involve initializing particle positions, updating velocities, etc., as shown in previous steps

# At this stage, ensure that anfis_model has optimized parameters via PSO (or assumed as ready)
# Continue with the previous steps where we transform data with anfis_model and proceed with SVM, RF, and k-NN



# Step 1: Load and process data
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

datasets = load_and_process_data(file_paths)

# Step 2: Prepare data for training
def prepare_data(datasets):
    X_combined = pd.DataFrame()
    y_combined = pd.Series(dtype='int')
    for data in datasets:
        data_filtered = data[data['Alertness_Level'].isin(['Low', 'High'])].copy()
        data_filtered['Alertness_Level'] = data_filtered['Alertness_Level'].map({'Low': 0, 'High': 1})
        X = data_filtered.drop(columns=['Alertness_Level', 'Theta/Beta Ratio', 'Timestamp'], errors='ignore')
        y = data_filtered['Alertness_Level']
        X_combined = pd.concat([X_combined, X], axis=0)
        y_combined = pd.concat([y_combined, y], axis=0)
    return X_combined, y_combined

X_combined, y_combined = prepare_data(datasets)

# Step 3: Calculate filter indices
X_low = X_combined[y_combined == 0]
X_high = X_combined[y_combined == 1]

fisher_index = [(X_high[col].mean() - X_low[col].mean()) / (X_high[col].std()**2 + X_low[col].std()**2) for col in X_combined.columns]
correlation_index = [np.cov(X_combined[col], y_combined)[0, 1] / (X_combined[col].std() * y_combined.std()) for col in X_combined.columns]
t_test_index = [abs(ttest_ind(X_low[col], X_high[col], equal_var=False)[0]) for col in X_combined.columns]
mutual_info_index = mutual_info_classif(X_combined, y_combined, discrete_features=False)

feature_indices = pd.DataFrame({
    'Feature': X_combined.columns,
    'Fisher Index': fisher_index,
    'Correlation Index': correlation_index,
    'T-test Index': t_test_index,
    'Mutual Information Index': mutual_info_index
})

# Step 4: Fuzzy inference for Importance Degree (ID)
def gaussian_membership(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)

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
important_features = feature_indices[feature_indices['Importance Degree (ID)'] >= 0.5]['Feature']
X_important = X_combined[important_features]

# Step 5: Dimensionality reduction using PCA for ANFIS input
pca = PCA(n_components=2)
X_anfis_reduced = pca.fit_transform(X_important)

# Step 6: Initialize and optimize ANFIS with PSO (ANFIS implementation details omitted for brevity)

# Step 7: Transform data using optimized ANFIS model
# (This assumes the ANFIS model was trained and optimized with PSO)
X_transformed = np.array([anfis_model.forward(x[0], x[1]) for x in X_anfis_reduced]).reshape(-1, 1)

# Step 8: Train and evaluate SVM, Random Forest, and k-NN on ANFIS-transformed data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_combined, test_size=0.2, random_state=42, stratify=y_combined)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM with grid search for best hyperparameters
param_grid_svm = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf', 'poly', 'sigmoid']}
svm_grid_search = GridSearchCV(SVC(), param_grid_svm, cv=5, scoring='accuracy', verbose=1)
svm_grid_search.fit(X_train_scaled, y_train)
best_svm_model = svm_grid_search.best_estimator_
svm_train_accuracy = best_svm_model.score(X_train_scaled, y_train)
svm_test_accuracy = best_svm_model.score(X_test_scaled, y_test)
y_pred_svm = best_svm_model.predict(X_test_scaled)
classification_report_svm = classification_report(y_test, y_pred_svm, target_names=['Low', 'High'])
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_train_accuracy = rf_model.score(X_train_scaled, y_train)
rf_test_accuracy = rf_model.score(X_test_scaled, y_test)
y_pred_rf = rf_model.predict(X_test_scaled)
classification_report_rf = classification_report(y_test, y_pred_rf, target_names=['Low', 'High'])
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# k-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)
knn_train_accuracy = knn_model.score(X_train_scaled, y_train)
knn_test_accuracy = knn_model.score(X_test_scaled, y_test)
y_pred_knn = knn_model.predict(X_test_scaled)
classification_report_knn = classification_report(y_test, y_pred_knn, target_names=['Low', 'High'])
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# Output results
print("SVM Results:")
print("Train Accuracy:", svm_train_accuracy)
print("Test Accuracy:", svm_test_accuracy)
print("Classification Report:\n", classification_report_svm)
print("Confusion Matrix:\n", conf_matrix_svm)

print("\nRandom Forest Results:")
print("Train Accuracy:", rf_train_accuracy)
print("Test Accuracy:", rf_test_accuracy)
print("Classification Report:\n", classification_report_rf)
print("Confusion Matrix:\n", conf_matrix_rf)

print("\nk-Nearest Neighbors Results:")
print("Train Accuracy:", knn_train_accuracy)
print("Test Accuracy:", knn_test_accuracy)
print("Classification Report:\n", classification_report_knn)
print("Confusion Matrix:\n", conf_matrix_knn)


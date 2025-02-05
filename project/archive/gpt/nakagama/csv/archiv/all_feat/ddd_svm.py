import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
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

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Defining the parameter grid for RBF kernel's hyperparameters
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [0.001, 0.01, 0.1, 1, 10]  # Kernel coefficient for RBF
}

# Setting up the GridSearchCV with RBF kernel SVM
svm_rbf = SVC(kernel='rbf', random_state=42)
grid_search = GridSearchCV(svm_rbf, param_grid, cv=5, scoring='accuracy', verbose=1)

# Fitting the grid search on training data
grid_search.fit(X_train_scaled, y_train)

# Extracting the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Training a new model with the best parameters on the full training set
svm_rbf_optimized = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], random_state=42)
svm_rbf_optimized.fit(X_train_scaled, y_train)

# Calculating training and test accuracy for the optimized RBF SVM
train_accuracy_rbf_opt = svm_rbf_optimized.score(X_train_scaled, y_train)
test_accuracy_rbf_opt = svm_rbf_optimized.score(X_test_scaled, y_test)

# Predicting and evaluating with optimized RBF SVM
y_pred_rbf_opt = svm_rbf_optimized.predict(X_test_scaled)
classification_report_rbf_opt = classification_report(y_test, y_pred_rbf_opt, target_names=['Low', 'High'])
conf_matrix_rbf_opt = confusion_matrix(y_test, y_pred_rbf_opt)

# Displaying the results
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)
print("Training Accuracy (Optimized RBF):", train_accuracy_rbf_opt)
print("Test Accuracy (Optimized RBF):", test_accuracy_rbf_opt)
print("\nClassification Report (Optimized RBF):\n", classification_report_rbf_opt)
print("Confusion Matrix (Optimized RBF):\n", conf_matrix_rbf_opt)

# Plotting the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_rbf_opt, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Optimized RBF SVM Classification of Alertness Level')
plt.show()


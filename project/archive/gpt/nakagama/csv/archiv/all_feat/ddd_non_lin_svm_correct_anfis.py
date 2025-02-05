import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind
from sklearn.feature_selection import mutual_info_classif

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

# Step 1: Load and prepare data
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

# Prepare data for processing
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

# Load and process data to define X_combined and y_combined
X_combined, y_combined = prepare_data(datasets)

# Step 2: Calculate filter indices for each feature
X_low = X_combined[y_combined == 0]
X_high = X_combined[y_combined == 1]

fisher_index = [(X_high[col].mean() - X_low[col].mean()) / (X_high[col].std()**2 + X_low[col].std()**2) for col in X_combined.columns]
correlation_index = [np.cov(X_combined[col], y_combined)[0, 1] / (X_combined[col].std() * y_combined.std()) for col in X_combined.columns]
t_test_index = [abs(ttest_ind(X_low[col], X_high[col], equal_var=False)[0]) for col in X_combined.columns]
mutual_info_index = mutual_info_classif(X_combined, y_combined, discrete_features=False)

filter_indices = pd.DataFrame({
    'Feature': X_combined.columns,
    'Fisher Index': fisher_index,
    'Correlation Index': correlation_index,
    'T-test Index': t_test_index,
    'Mutual Information Index': mutual_info_index
})

# Define Gaussian membership function for fuzzy inputs
def gaussian_mf(x, mean, sigma):
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

# Step 3: Define ANFIS model for ID calculation and integrate with PSO
class ANFISForID:
    def __init__(self, num_rules=5):
        self.num_rules = num_rules
        self.means = np.random.uniform(0, 1, (num_rules, 4))  # 4 inputs (filter indices)
        self.sigmas = np.random.uniform(0.1, 0.5, (num_rules, 4))
        self.consequents = np.random.uniform(0, 1, num_rules)  # Output range [0, 1]

    def forward(self, inputs):
        firing_strengths = np.zeros(self.num_rules)
        for i in range(self.num_rules):
            mu_values = [gaussian_mf(inputs[j], self.means[i, j], self.sigmas[i, j]) for j in range(4)]
            firing_strengths[i] = np.prod(mu_values)
        if firing_strengths.sum() == 0:
            return 0
        normalized_strengths = firing_strengths / firing_strengths.sum()
        importance_degree = np.dot(normalized_strengths, self.consequents)
        return importance_degree

    def set_parameters(self, means, sigmas, consequents):
        self.means = means
        self.sigmas = sigmas
        self.consequents = consequents

# Initialize the ANFIS model
anfis_id_model = ANFISForID(num_rules=5)

# Define objective function for PSO
def objective_function_pso(anfis, X_features, y_labels):
    IDs = np.array([anfis.forward(row) for row in X_features])
    selected_features = X_combined.loc[:, IDs >= 0.5]
    if selected_features.empty:
        return 1.0  # Penalize if no features are selected

    X_train, X_test, y_train, y_test = train_test_split(selected_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(kernel='rbf', C=1, gamma=0.1, random_state=42)
    svm.fit(X_train_scaled, y_train)
    accuracy = svm.score(X_test_scaled, y_test)
    return -accuracy

# PSO optimization settings
class Particle:
    def __init__(self, anfis):
        self.position_means = np.random.uniform(0, 1, anfis.means.shape)
        self.position_sigmas = np.random.uniform(0.1, 0.5, anfis.sigmas.shape)
        self.position_consequents = np.random.uniform(0, 1, anfis.consequents.shape)

        self.velocity_means = np.zeros_like(self.position_means)
        self.velocity_sigmas = np.zeros_like(self.position_sigmas)
        self.velocity_consequents = np.zeros_like(self.position_consequents)

        self.best_position_means = np.copy(self.position_means)
        self.best_position_sigmas = np.copy(self.position_sigmas)
        self.best_position_consequents = np.copy(self.position_consequents)
        self.best_score = float('inf')

c1, c2 = 2, 2
npop = 20  # Reduced particle count for faster processing
w = 0.95
max_iter = 50  # Reduced iterations for faster processing

particles = [Particle(anfis_id_model) for _ in range(npop)]
global_best_score = float('inf')
global_best_means, global_best_sigmas, global_best_consequents = None, None, None

# PSO optimization loop
for iteration in range(max_iter):
    for particle in particles:
        anfis_id_model.set_parameters(particle.position_means, particle.position_sigmas, particle.position_consequents)
        score = objective_function_pso(anfis_id_model, filter_indices[['Fisher Index', 'Correlation Index', 'T-test Index', 'Mutual Information Index']].values, y_combined)

        if score < particle.best_score:
            particle.best_score = score
            particle.best_position_means = np.copy(particle.position_means)
            particle.best_position_sigmas = np.copy(particle.position_sigmas)
            particle.best_position_consequents = np.copy(particle.position_consequents)

        if score < global_best_score:
            global_best_score = score
            global_best_means = np.copy(particle.position_means)
            global_best_sigmas = np.copy(particle.position_sigmas)
            global_best_consequents = np.copy(particle.position_consequents)

    # Update velocities and positions
    for particle in particles:
        r1, r2 = np.random.rand(), np.random.rand()
        particle.velocity_means = w * particle.velocity_means + c1 * r1 * (particle.best_position_means - particle.position_means) + c2 * r2 * (global_best_means - particle.position_means)
        particle.velocity_sigmas = w * particle.velocity_sigmas + c1 * r1 * (particle.best_position_sigmas - particle.position_sigmas) + c2 * r2 * (global_best_sigmas - particle.position_sigmas)
        particle.velocity_consequents = w * particle.velocity_consequents + c1 * r1 * (particle.best_position_consequents - particle.position_consequents) + c2 * r2 * (global_best_consequents - particle.position_consequents)

        # Update positions
        particle.position_means += particle.velocity_means
        particle.position_sigmas += particle.velocity_sigmas
        particle.position_consequents += particle.velocity_consequents

# Set the ANFIS model to the best found parameters
anfis_id_model.set_parameters(global_best_means, global_best_sigmas, global_best_consequents)
best_accuracy = -global_best_score  # Convert back to positive accuracy for reporting

print("Best SVM accuracy achieved with optimized ANFIS-selected features:", best_accuracy)

# Transform data using the optimized ANFIS model to calculate ID scores for each feature
IDs = np.array([anfis_id_model.forward(row) for row in filter_indices[['Fisher Index', 'Correlation Index', 'T-test Index', 'Mutual Information Index']].values])

# Select features with ID >= 0.5
selected_features = X_combined.loc[:, IDs >= 0.5]
if not selected_features.empty:
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(selected_features, y_combined, test_size=0.2, random_state=42, stratify=y_combined)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define a dictionary to store results for each SVM kernel
    svm_results = {}

    # 1. RBF Kernel
    svm_rbf = SVC(kernel='rbf', C=1, gamma=0.1, random_state=42)
    svm_rbf.fit(X_train_scaled, y_train)
    rbf_train_accuracy = svm_rbf.score(X_train_scaled, y_train)
    rbf_test_accuracy = svm_rbf.score(X_test_scaled, y_test)
    y_pred_rbf = svm_rbf.predict(X_test_scaled)
    svm_results['RBF'] = {
        'Train Accuracy': rbf_train_accuracy,
        'Test Accuracy': rbf_test_accuracy,
        'Classification Report': classification_report(y_test, y_pred_rbf, target_names=['Low', 'High']),
        'Confusion Matrix': confusion_matrix(y_test, y_pred_rbf)
    }

    # 2. Polynomial Kernel
    svm_poly = SVC(kernel='poly', C=1, degree=3, gamma='auto', random_state=42)
    svm_poly.fit(X_train_scaled, y_train)
    poly_train_accuracy = svm_poly.score(X_train_scaled, y_train)
    poly_test_accuracy = svm_poly.score(X_test_scaled, y_test)
    y_pred_poly = svm_poly.predict(X_test_scaled)
    svm_results['Polynomial'] = {
        'Train Accuracy': poly_train_accuracy,
        'Test Accuracy': poly_test_accuracy,
        'Classification Report': classification_report(y_test, y_pred_poly, target_names=['Low', 'High']),
        'Confusion Matrix': confusion_matrix(y_test, y_pred_poly)
    }

    # 3. Sigmoid Kernel
    svm_sigmoid = SVC(kernel='sigmoid', C=1, gamma=0.1, random_state=42)
    svm_sigmoid.fit(X_train_scaled, y_train)
    sigmoid_train_accuracy = svm_sigmoid.score(X_train_scaled, y_train)
    sigmoid_test_accuracy = svm_sigmoid.score(X_test_scaled, y_test)
    y_pred_sigmoid = svm_sigmoid.predict(X_test_scaled)
    svm_results['Sigmoid'] = {
        'Train Accuracy': sigmoid_train_accuracy,
        'Test Accuracy': sigmoid_test_accuracy,
        'Classification Report': classification_report(y_test, y_pred_sigmoid, target_names=['Low', 'High']),
        'Confusion Matrix': confusion_matrix(y_test, y_pred_sigmoid)
    }

    # Display results for each kernel
    for kernel, results in svm_results.items():
        print(f"{kernel} Kernel SVM Results:")
        print("Train Accuracy:", results['Train Accuracy'])
        print("Test Accuracy:", results['Test Accuracy'])
        print("Classification Report:\n", results['Classification Report'])
        print("Confusion Matrix:\n", results['Confusion Matrix'])
        print("\n" + "-"*50 + "\n")

else:
    print("No features selected with ID >= 0.5. Please review the ANFIS parameter optimization.")


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import glob

# EEGデータの読み込み
fp = '../../../../dataset/Aygun2024/physio'

# Load all EEG and SIMlsl files
file_paths_eeg      = sorted(glob.glob(fp + '/S0116/EEG_*.mat'))
file_paths_simlsl   = sorted(glob.glob(fp + '/S0116/SIMlsl_*.mat'))

# Load the data for all subjects
eeg_data_list = [scipy.io.loadmat(file)['rawEEG'] for file in file_paths_eeg]
simlsl_data_list = [scipy.io.loadmat(file)['SIM_lsl'] for file in file_paths_simlsl]

# Define bandpass filter function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data, axis=-1)
    return y

# Function to calculate power using Welch's method
def calculate_band_power(data, fs):
    freqs, power = welch(data, fs=fs, nperseg=fs*2)
    return np.mean(power)

# Calculate arousal index from EEG data
def label_arousal_state(eeg_data, fs):
    num_samples = eeg_data.shape[1]
    window_size = fs * 50  # 50-second window
    step_size = fs * 50  # 50-second step
    arousal_index = []

    for start in range(0, num_samples - window_size, step_size):
        alpha_power_total = 0
        beta_power_total = 0

        for i in range(1, eeg_data.shape[0]):
            window_data = eeg_data[i, start:start + window_size]
            alpha_power = calculate_band_power(bandpass_filter(window_data, 8, 13, fs), fs)
            beta_power = calculate_band_power(bandpass_filter(window_data, 13, 30, fs), fs)
            alpha_power_total += alpha_power
            beta_power_total += beta_power

        if alpha_power_total > 0:
            arousal_index.append(beta_power_total / alpha_power_total)
        else:
            arousal_index.append(0)

    threshold = np.median(arousal_index)
    arousal_labels = [1 if index >= threshold else 0 for index in arousal_index]

    return arousal_index, arousal_labels

# Prepare features and labels for all subjects
X_list = []
y_list = []

for eeg_data, simlsl_data in zip(eeg_data_list, simlsl_data_list):
    # Calculate sampling frequency from EEG timestamps
    fs_eeg = int(1 / np.median(np.diff(eeg_data[0, :])))

    # Calculate arousal labels
    _, arousal_labels = label_arousal_state(eeg_data, fs_eeg)

    # Prepare features (steering angle and lateral acceleration)
    num_samples = len(arousal_labels)
    steering_data = simlsl_data[29, :num_samples]
    lateral_acceleration_data = simlsl_data[19, :num_samples]
    X = np.vstack((steering_data, lateral_acceleration_data)).T
    y = arousal_labels

    # Append to the overall list
    X_list.append(X)
    y_list.append(y)

# Concatenate all data
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Function to train and evaluate a classifier
def train_and_evaluate_classifier(classifier, classifier_name):
    classifier.fit(X_train, y_train)
    y_pred_train = classifier.predict(X_train)
    y_pred_test = classifier.predict(X_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)

    # Display results
    print(f"{classifier_name} Classifier:\n")
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Training Confusion Matrix:\n{conf_matrix_train}")
    print(f"Test Confusion Matrix:\n{conf_matrix_test}\n")

# List of classifiers to evaluate
classifiers = [
    (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
    (SVC(kernel='rbf', gamma='scale', random_state=42), "SVM (RBF Kernel, Default Gamma)"),
    (SVC(kernel='poly', degree=3, random_state=42), "SVM (Polynomial Kernel)"),
    (KNeighborsClassifier(n_neighbors=5), "k-Nearest Neighbors"),
    (LogisticRegression(random_state=42), "Logistic Regression"),
    (DecisionTreeClassifier(random_state=42), "Decision Tree"),
    (GradientBoostingClassifier(n_estimators=100, random_state=42), "Gradient Boosting"),
    (MLPClassifier(random_state=42, max_iter=500), "Neural Network (MLP)")
]

# Train and evaluate each classifier
for clf, name in classifiers:
    train_and_evaluate_classifier(clf, name)

# Optimize SVM (RBF Kernel) with GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10], 'kernel': ['rbf']}
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
print("Best Parameters for SVM (RBF Kernel):")
print(grid_search.best_params_)

# Evaluate the best SVM model
best_svm = grid_search.best_estimator_
train_and_evaluate_classifier(best_svm, "Optimized SVM (RBF Kernel)")


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# EEGデータの読み込み
fp = '../../../../dataset/Aygun2024/physio/'
eeg_1 = scipy.io.loadmat(fp + 'S0113/EEG_S0113_1.mat')['rawEEG']
eeg_2 = scipy.io.loadmat(fp + 'S0116/EEG_S0116_1.mat')['rawEEG']
eeg_3 = scipy.io.loadmat(fp + 'S0116/EEG_S0116_2.mat')['rawEEG']

# SIMlslデータの読み込み
sim_lsl_1 = scipy.io.loadmat(fp + 'S0113/SIMlsl_S0113_1.mat')['SIM_lsl']
sim_lsl_2 = scipy.io.loadmat(fp + 'S0116/SIMlsl_S0116_1.mat')['SIM_lsl']
sim_lsl_3 = scipy.io.loadmat(fp + 'S0116/SIMlsl_S0116_2.mat')['SIM_lsl']

## Load the .mat files
#sim_1_data = scipy.io.loadmat('/mnt/data/SIMlsl_S0113_1.mat')
#sim_2_data = scipy.io.loadmat('/mnt/data/SIMlsl_S0116_2.mat')
#sim_3_data = scipy.io.loadmat('/mnt/data/SIMlsl_S0116_1.mat')

## Extracting the SIM_lsl data
#sim_lsl_1 = sim_1_data['SIM_lsl']
#sim_lsl_2 = sim_2_data['SIM_lsl']
#sim_lsl_3 = sim_3_data['SIM_lsl']

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

## Assuming EEG data is loaded and processed to get arousal labels
## Load EEG data (dummy example, replace with actual EEG file paths)
#eeg_1_data = scipy.io.loadmat('/mnt/data/EEG_S0113_1.mat')
#eeg_2_data = scipy.io.loadmat('/mnt/data/EEG_S0116_2.mat')
#eeg_3_data = scipy.io.loadmat('/mnt/data/EEG_S0116_1.mat')

## Extract raw EEG data
#eeg_1 = eeg_1_data['rawEEG']
#eeg_2 = eeg_2_data['rawEEG']
#eeg_3 = eeg_3_data['rawEEG']

# Calculate arousal labels for each subject
fs_eeg_1 = int(1 / np.median(np.diff(eeg_1[0, :])))
fs_eeg_2 = int(1 / np.median(np.diff(eeg_2[0, :])))
fs_eeg_3 = int(1 / np.median(np.diff(eeg_3[0, :])))

_, arousal_labels_1 = label_arousal_state(eeg_1, fs_eeg_1)
_, arousal_labels_2 = label_arousal_state(eeg_2, fs_eeg_2)
_, arousal_labels_3 = label_arousal_state(eeg_3, fs_eeg_3)

# Prepare features and labels
def prepare_features_and_labels_adjusted(sim_lsl_data, arousal_labels):
    num_samples = len(arousal_labels)
    steering_data = sim_lsl_data[29, :num_samples]
    lateral_acceleration_data = sim_lsl_data[19, :num_samples]
    X = np.vstack((steering_data, lateral_acceleration_data)).T
    y = arousal_labels
    return X, y

X1, y1 = prepare_features_and_labels_adjusted(sim_lsl_1, arousal_labels_1)
X2, y2 = prepare_features_and_labels_adjusted(sim_lsl_2, arousal_labels_2)
X3, y3 = prepare_features_and_labels_adjusted(sim_lsl_3, arousal_labels_3)

X = np.concatenate((X1, X2, X3), axis=0)
y = np.concatenate((y1, y2, y3), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train RandomForest classifier
classifier_rf = RandomForestClassifier(n_estimators=100, random_state=42)
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_test)
train_accuracy_rf = accuracy_score(y_train, classifier_rf.predict(X_train))
test_accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_train_rf = confusion_matrix(y_train, classifier_rf.predict(X_train))
conf_matrix_test_rf = confusion_matrix(y_test, y_pred_rf)

# Train SVM classifier with RBF kernel
classifier_svm_rbf = SVC(kernel='rbf', random_state=42)
classifier_svm_rbf.fit(X_train, y_train)
y_pred_svm_rbf = classifier_svm_rbf.predict(X_test)
train_accuracy_svm_rbf = accuracy_score(y_train, classifier_svm_rbf.predict(X_train))
test_accuracy_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)
conf_matrix_train_svm_rbf = confusion_matrix(y_train, classifier_svm_rbf.predict(X_train))
conf_matrix_test_svm_rbf = confusion_matrix(y_test, y_pred_svm_rbf)

# Train SVM classifier with polynomial kernel
classifier_svm_poly = SVC(kernel='poly', degree=3, random_state=42)
classifier_svm_poly.fit(X_train, y_train)
y_pred_svm_poly = classifier_svm_poly.predict(X_test)
train_accuracy_svm_poly = accuracy_score(y_train, classifier_svm_poly.predict(X_train))
test_accuracy_svm_poly = accuracy_score(y_test, y_pred_svm_poly)
conf_matrix_train_svm_poly = confusion_matrix(y_train, classifier_svm_poly.predict(X_train))
conf_matrix_test_svm_poly = confusion_matrix(y_test, y_pred_svm_poly)

# Display results
print("Random Forest Classifier:\n")
print(f"Training Accuracy: {train_accuracy_rf}")
print(f"Test Accuracy: {test_accuracy_rf}")
print(f"Training Confusion Matrix:\n{conf_matrix_train_rf}")
print(f"Test Confusion Matrix:\n{conf_matrix_test_rf}")

print("\nSVM Classifier (RBF Kernel):\n")
print(f"Training Accuracy: {train_accuracy_svm_rbf}")
print(f"Test Accuracy: {test_accuracy_svm_rbf}")
print(f"Training Confusion Matrix:\n{conf_matrix_train_svm_rbf}")
print(f"Test Confusion Matrix:\n{conf_matrix_test_svm_rbf}")

print("\nSVM Classifier (Polynomial Kernel):\n")
print(f"Training Accuracy: {train_accuracy_svm_poly}")
print(f"Test Accuracy: {test_accuracy_svm_poly}")
print(f"Training Confusion Matrix:\n{conf_matrix_train_svm_poly}")
print(f"Test Confusion Matrix:\n{conf_matrix_test_svm_poly}")

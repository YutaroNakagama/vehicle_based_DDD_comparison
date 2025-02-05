import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch, find_peaks
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import pdist
from antropy import sample_entropy
from scipy.stats import entropy
import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Attention, Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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

    if len(arousal_index) == 0:
        return [], []

    threshold = np.median(arousal_index)
    arousal_labels = [1 if index >= threshold else 0 for index in arousal_index]

    return arousal_index, arousal_labels

# Extract features from a signal window
def extract_features(signal):
    features = {}
    features['range'] = np.max(signal) - np.min(signal)
    features['std_dev'] = np.std(signal)
    features['energy'] = np.sum(signal ** 2)
    features['zcr'] = ((np.diff(np.sign(signal)) != 0).sum()) / len(signal)
    features['first_quartile'] = np.percentile(signal, 25)
    features['second_quartile'] = np.median(signal)
    features['third_quartile'] = np.percentile(signal, 75)
    features['kfd'] = np.log(len(signal)) / (np.log(len(signal)) + np.log(np.mean(np.diff(signal))))
    features['skewness'] = skew(signal)
    features['kurtosis'] = kurtosis(signal)
    features['sample_entropy'] = sample_entropy(signal, order=2)
    features['shannon_entropy'] = entropy(signal)
    return features

# Prepare features and labels for all subjects
X_list = []
y_list = []

for eeg_data, simlsl_data in zip(eeg_data_list, simlsl_data_list):
    # Calculate sampling frequency from EEG timestamps
    fs_eeg = int(1 / np.median(np.diff(eeg_data[0, :])))

    # Calculate arousal labels
    _, arousal_labels = label_arousal_state(eeg_data, fs_eeg)

    if len(arousal_labels) == 0:
        continue

    # Prepare features (steering angle and lateral acceleration)
    num_samples = len(arousal_labels)
    steering_data = simlsl_data[29, :num_samples]
    lateral_acceleration_data = simlsl_data[19, :num_samples]

    # Extract features using sliding window of 50 seconds
    window_size = fs_eeg * 50
    step_size = fs_eeg * 50
    X_features = []

    for start in range(0, num_samples - window_size, step_size):
        steering_window = steering_data[start:start + window_size]
        lateral_accel_window = lateral_acceleration_data[start:start + window_size]

        steering_features = extract_features(steering_window)
        lateral_accel_features = extract_features(lateral_accel_window)

        combined_features = list(steering_features.values()) + list(lateral_accel_features.values())
        X_features.append(combined_features)

    if len(X_features) > 0:
        X_list.extend(X_features)
        y_list.extend(arousal_labels[:len(X_features)])

# Convert feature lists to numpy arrays
X = np.array(X_list)
y = np.array(y_list)

# Ensure there are enough samples for training and testing
if len(y) == 0:
    raise ValueError("No samples available for training and testing after feature extraction.")

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

# Bi-LSTM with Attention model
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

input_layer = Input(shape=(1, X_train.shape[1]))
lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
attention_data = TimeDistributed(Dense(1, activation='tanh'))(lstm_layer)
attention_weights = tf.keras.layers.Softmax()(attention_data)
context_vector = tf.keras.layers.Multiply()([lstm_layer, attention_weights])
context_vector = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context_vector)
output_layer = Dense(1, activation='sigmoid')(context_vector)

bi_lstm_model = Model(inputs=input_layer, outputs=output_layer)
bi_lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the Bi-LSTM model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
bi_lstm_model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test), epochs=50, batch_size=32, callbacks=[early_stopping], verbose=2)

# Evaluate the Bi-LSTM model
train_accuracy_lstm = bi_lstm_model.evaluate(X_train_lstm, y_train, verbose=0)[1]
test_accuracy_lstm = bi_lstm_model.evaluate(X_test_lstm, y_test, verbose=0)[1]
print(f"Bi-LSTM with Attention Classifier:\n")
print(f"Training Accuracy: {train_accuracy_lstm}")
print(f"Test Accuracy: {test_accuracy_lstm}")


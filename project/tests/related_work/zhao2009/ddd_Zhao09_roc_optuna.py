import time
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.utils import resample
from scipy.signal import convolve
import optuna
import matplotlib.pyplot as plt

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

def objective(trial):
    gamma = trial.suggest_loguniform('gamma', 1e-3, 1e2)
    
    # Initialize SVM model with RBF kernel and fixed C=300
    svm_rbf_for_optimization = SVC(kernel='rbf', C=300, gamma=gamma, random_state=42, probability=True)
    
    # Train the model
    svm_rbf_for_optimization.fit(X_train, y_train)
    
    # Predict probabilities for validation data
    y_val_probs = svm_rbf_for_optimization.predict_proba(X_val)[:, 1]
    
    # Compute ROC AUC
    roc_auc = auc(*roc_curve(y_val, y_val_probs)[:2])
    
    return roc_auc

# Use Optuna to optimize the gamma parameter
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get the best gamma parameter
best_gamma = study.best_params['gamma']

# Retrain the model with the best gamma
best_svm_model = SVC(kernel='rbf', C=300, gamma=best_gamma, random_state=42, probability=True)
best_svm_model.fit(X_train, y_train)

y_val_probs_best = best_svm_model.predict_proba(X_val)[:, 1]

# Calculate validation accuracy and classification report with the optimized gamma
val_accuracy_best = accuracy_score(y_val, best_svm_model.predict(X_val))
classification_rep_best = classification_report(y_val, best_svm_model.predict(X_val))

print("best_gamma              ", best_gamma)
print("val_accuracy_best       ", val_accuracy_best)
print("classification_rep_best\n", classification_rep_best)

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

# Compute ROC curve and ROC area
y_val_scores = best_svm_model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_val_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


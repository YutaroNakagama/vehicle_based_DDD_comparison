import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# File paths for the uploaded CSVs
file_paths = {
#    "Test Aref": "roc_test_aref.csv",
    "Curve Data Zhao": "roc_curve_data_zhao.csv",
    "Data Integ": "roc_data_integ.csv",
    "Validation Aref": "roc_val_aref.csv",
}

# Load all datasets
roc_data = {}
for name, path in file_paths.items():
    df = pd.read_csv(path)
    if name == "Curve Data Zhao":
        # Standardize column names for "Curve Data Zhao"
        df.rename(columns={
            "False Positive Rate": "FPR",
            "True Positive Rate": "TPR"
        }, inplace=True)
    roc_data[name] = df

# Plot all ROC curves
plt.figure(figsize=(6, 5))
for name, df in roc_data.items():
    if name == "Data Integ":
        # Special handling for models in "Data Integ"
        models = df["Model"].unique()
        for model in models:
            model_data = df[df["Model"] == model].sort_values(by="FPR")
            fpr = model_data["FPR"]
            tpr = model_data["TPR"]
            model_auc = model_data["AUC"].iloc[0]
            plt.plot(fpr, tpr, label=f"{model} ({name}, AUC = {model_auc:.2f})")
    else:
        # Standard ROC plot
        df_sorted = df.sort_values(by="FPR")
        fpr = df_sorted["FPR"]
        tpr = df_sorted["TPR"]
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=0.8, label="Random Guess (AUC = 0.50)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
#plt.title("Combined ROC Curves and AUCs")
plt.legend(loc="lower right", fontsize='small')
plt.grid(alpha=0.3)
plt.show()


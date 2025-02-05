import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# File paths for the uploaded CSVs
file_paths = {
#    "Test Aref": "roc_test_aref.csv",
    "Data Integ": "roc_data_integ.csv",
    "SVM_wavelet": "roc_curve_data_zhao.csv",
    "SVM_ANFIS": "roc_val_aref.csv",
}

# Load all datasets
roc_data = {}
for name, path in file_paths.items():
    df = pd.read_csv(path)
    if name == "SVM_wavelet":
        # Standardize column names for "Curve Data Zhao"
        df.rename(columns={
            "False Positive Rate": "FPR",
            "True Positive Rate": "TPR"
        }, inplace=True)
    roc_data[name] = df

# Plot all ROC curves
plt.figure(figsize=(6, 5))
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 14})  # Set font to Times New Roman and size to 10

for name, df in roc_data.items():
    if name == "Data Integ":
        # Special handling for models in "Data Integ"
        models = df["Model"].unique()
        for model in models:
            model_data = df[df["Model"] == model].sort_values(by="FPR")
            fpr = model_data["FPR"]
            tpr = model_data["TPR"]
            model_auc = model_data["AUC"].iloc[0]
            plt.plot(fpr, tpr, label=f"{model} (AUC = {model_auc:.2f})")
    else:
        # Standard ROC plot
        df_sorted = df.sort_values(by="FPR")
        fpr = df_sorted["FPR"]
        tpr = df_sorted["TPR"]
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=0.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.tick_params(direction='in')  # Set ticks to point inward
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
#plt.title("Combined ROC Curves and AUCs")
legend = plt.legend(loc="lower right", fontsize='small', frameon=True)
legend.get_frame().set_edgecolor('black')  # Set legend frame color to black
legend.get_frame().set_linewidth(1)  # Set frame line width
legend.get_frame().set_boxstyle('Square')  # Set box style to square
plt.grid(alpha=0.3)

# Save plot to a PDF file
plt.savefig("roc.pdf", format="pdf", bbox_inches="tight")
plt.show()


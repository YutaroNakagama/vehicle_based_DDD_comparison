import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns

# 読み込む被験者リストファイルのパス
subject_list_path = '../../../../dataset/Aygun2024/subject_list_temp.txt'

# 被験者リストを読み込む
with open(subject_list_path, 'r') as file:
    subjects = [line.strip().split('/') for line in file.readlines()]

# データフレームリストを作成し、CSVファイルを読み込む
dfs = []
for subject in subjects:
    subject_id, version = subject
    file_name = f'{subject_id}_{version}_merged_data_with_KSS.csv'
    try:
        df = pd.read_csv(f'./csv/all/{file_name}')
        dfs.append(df)
    except FileNotFoundError:
        print(f"File not found: {file_name}")

# データを統合し、クラス4とクラス8に絞り込む
all_data = pd.concat(dfs, ignore_index=True)
filtered_data = all_data[all_data["KSS_Theta_Alpha"].isin([4, 8])]

# 特徴量とターゲットを選択し、欠損値を削除
X = filtered_data.iloc[:, 1:46].dropna()
y = filtered_data["KSS_Theta_Alpha"].loc[X.index]

# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 複数の分類器を定義
classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM (Linear Kernel)": SVC(kernel="linear", probability=True, random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# 各分類器で訓練と評価を行う
results = {}
plt.figure(figsize=(6, 4))
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    # 精度と分類レポートを保存
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, target_names=["Class 4", "Class 8"], output_dict=True)
    results[name] = {"accuracy": accuracy, "classification_report": classification_rep}

    # ROCとAUCの計算
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test.replace({4: 0, 8: 1}), y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        print(f'{name} (AUC = {roc_auc:.2f})')
        plt.savefig(f'{name}_ROC.svg')

    # 混同行列のプロット
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[4, 8])
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 4", "Class 8"], yticklabels=["Class 4", "Class 8"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f'{name}_conf_mat.svg')
    plt.show()

# ROC曲線のプロット
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# 各分類器の結果を表示
for name, metrics in results.items():
    print(f"\nClassifier: {name}")
    print(f"Accuracy: {metrics['accuracy']}")
    print("Classification Report:")
    print(pd.DataFrame(metrics["classification_report"]).transpose())

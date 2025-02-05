import pandas as pd
import matplotlib.pyplot as plt

# CSVデータを読み込んでROC曲線を描画
def plot_roc_from_csv(file_name, label):
    data = pd.read_csv(file_name)
    plt.plot(data['FPR'], data['TPR'], label=label)

# 描画
plt.figure(figsize=(10, 8))
plot_roc_from_csv('roc_train.csv', 'Train')
plot_roc_from_csv('roc_val.csv', 'Validation')
plot_roc_from_csv('roc_test.csv', 'Test')

# グラフの装飾
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
plt.title("ROC Curve", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.legend(fontsize=12)
plt.grid()
plt.show()


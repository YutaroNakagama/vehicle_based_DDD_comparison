# 必要なライブラリをインポート
import pandas as pd
import matplotlib.pyplot as plt

# ROC曲線データをCSVから読み込み
roc_data = pd.read_csv('roc_time_window.csv')

# ROC曲線をプロット
plt.figure()
plt.plot(roc_data['False Positive Rate'], roc_data['True Positive Rate'], color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


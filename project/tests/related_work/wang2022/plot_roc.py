import pandas as pd
import matplotlib.pyplot as plt

# CSVを読み込み、ROC曲線を描画
roc_csv_path = 'roc_curve_fold_1.csv'  # 保存されたCSVファイルのパス
roc_data = pd.read_csv(roc_csv_path)

# ROC曲線をプロット
plt.plot(roc_data['FPR'], roc_data['TPR'], label=f'ROC curve (AUC = {np.trapz(roc_data["TPR"], roc_data["FPR"]):.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


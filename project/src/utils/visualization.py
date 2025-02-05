
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def plot_custom_colored_distribution(data, output_path=None, threshold=3):
    """
    外れ値を除外し、カスタムカラーで分布図をプロットする関数。

    Parameters:
        data (np.ndarray or list): 分布データ
        output_path (str, optional): プロットを保存するパス (指定しない場合は画面に表示)
        threshold (float): 外れ値除外の標準偏差の閾値
    """
    # 外れ値を除外
    #data = remove_outliers(data, threshold)

    # 横軸を9等分する範囲を設定
    bins = np.linspace(min(data), max(data), 10)  # 9等分の境界
    bin_centers = (bins[:-1] + bins[1:]) / 2  # 各ビンの中心

    # プロットの作成
    plt.figure(figsize=(12, 8))

    # データのヒストグラムを描画
    counts, _, patches = plt.hist(data, bins=bins, color='gray', alpha=0.6, edgecolor='black', label="Histogram")

    # 色分けを設定
    for i, patch in enumerate(patches):
        if i < 6:  # 下位6つは緑
            patch.set_facecolor('green')
        elif i >= 7:  # 上位2つは黄色
            patch.set_facecolor('yellow')
        else:  # その他はグレー
            patch.set_facecolor('gray')

    # KDEプロットを追加
    sns.kdeplot(data, color='blue', linewidth=2, label="KDE", alpha=0.8)

    # 軸ラベルとタイトル
    plt.title("Custom Colored Distribution of Data", fontsize=16)
    plt.xlabel("Theta/Alpha Ratio", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)

    # 凡例を追加
    plt.legend(fontsize=12)
    plt.grid(True)

    # プロットを保存または表示
    if output_path:
        plt.savefig(output_path)
        #print(f"Plot saved to {output_path}")
        logging.info(f"Plot saved to {output_path}")
    else:
        plt.show()


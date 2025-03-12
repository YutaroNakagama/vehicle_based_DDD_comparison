import warnings
warnings.filterwarnings("ignore")

# 必要なライブラリのインポート
import time
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import ttest_ind, pearsonr
from pyswarm import pso  # PSOのためのライブラリ
from tqdm import tqdm

# 特徴量の計算関数
def calculate_features_for_signals(angle_window, speed_window):
    features = {}
    features['angle_range'] = angle_window.max() - angle_window.min()
    features['angle_standard_dev'] = angle_window.std()
    features['angle_energy'] = np.sum(angle_window ** 2)
    features['angle_zcr'] = ((angle_window[:-1] * angle_window[1:]) < 0).sum() / len(angle_window)
    features['angle_first_quartile'] = np.percentile(angle_window, 25)
    features['angle_second_quartile'] = np.percentile(angle_window, 50)
    features['angle_third_quartile'] = np.percentile(angle_window, 75)
    features['angle_kfd'] = np.log10(len(angle_window)) / (np.log10(len(angle_window)) + np.log10(np.mean(np.abs(np.diff(angle_window)))))
    features['angle_skewness'] = angle_window.skew()
    features['angle_kurtosis'] = angle_window.kurtosis()
    features['angle_sam_entropy'] = -np.sum(np.log2(angle_window / angle_window.sum()) * (angle_window / angle_window.sum()))
    features['angle_shannon_entropy'] = -np.sum(angle_window * np.log2(angle_window + 1e-10))
    features['angle_freq_variability'] = np.var(np.fft.fft(angle_window))
    features['angle_spectral_entropy'] = -np.sum(np.log2(np.abs(np.fft.fft(angle_window)) + 1e-10))
    features['angle_spectral_flux'] = np.sum(np.abs(np.diff(np.fft.fft(angle_window))))
    features['angle_center_gravity_freq'] = np.sum(np.fft.fftfreq(len(angle_window)) * np.abs(np.fft.fft(angle_window)))
    features['angle_dominant_freq'] = np.argmax(np.abs(np.fft.fft(angle_window)))
    features['angle_avg_psd'] = np.mean(np.abs(np.fft.fft(angle_window))**2)

    features['speed_range'] = speed_window.max() - speed_window.min()
    features['speed_standard_dev'] = speed_window.std()
    features['speed_energy'] = np.sum(speed_window ** 2)
    features['speed_zcr'] = ((speed_window[:-1] * speed_window[1:]) < 0).sum() / len(speed_window)
    features['speed_first_quartile'] = np.percentile(speed_window, 25)
    features['speed_second_quartile'] = np.percentile(speed_window, 50)
    features['speed_third_quartile'] = np.percentile(speed_window, 75)
    features['speed_kfd'] = np.log10(len(speed_window)) / (np.log10(len(speed_window)) + np.log10(np.mean(np.abs(np.diff(speed_window)))))
    features['speed_skewness'] = speed_window.skew()
    features['speed_kurtosis'] = speed_window.kurtosis()
    features['speed_sam_entropy'] = -np.sum(np.log2(speed_window / speed_window.sum()) * (speed_window / speed_window.sum()))
    features['speed_shannon_entropy'] = -np.sum(speed_window * np.log2(speed_window + 1e-10))
    features['speed_freq_variability'] = np.var(np.fft.fft(speed_window))
    features['speed_spectral_entropy'] = -np.sum(np.log2(np.abs(np.fft.fft(speed_window)) + 1e-10))
    features['speed_spectral_flux'] = np.sum(np.abs(np.diff(np.fft.fft(speed_window))))
    features['speed_center_gravity_freq'] = np.sum(np.fft.fftfreq(len(speed_window)) * np.abs(np.fft.fft(speed_window)))
    features['speed_dominant_freq'] = np.argmax(np.abs(np.fft.fft(speed_window)))
    features['speed_avg_psd'] = np.mean(np.abs(np.fft.fft(speed_window))**2)
    return features

# 4つのインデックスを計算
def calculate_feature_indices(features_df, label_df):
    fisher_indices, correlation_indices, t_test_indices, mutual_info_indices = [], [], [], []
    for column in features_df.columns:
        group1 = features_df[column][label_df == 1]
        group0 = features_df[column][label_df == 0]
        fisher_index = ((group1.mean() - group0.mean()) ** 2) / (group1.var() + group0.var() + 1e-10)
        fisher_indices.append(fisher_index)
        correlation, _ = pearsonr(features_df[column], label_df)
        correlation_indices.append(abs(correlation))
        t_stat, _ = ttest_ind(group1, group0, equal_var=False)
        t_test_indices.append(abs(t_stat))
        mutual_info = mutual_info_classif(features_df[column].values.reshape(-1, 1), label_df)[0]
        mutual_info_indices.append(mutual_info)
    indices_df = pd.DataFrame({
        'Feature': features_df.columns,
        'Fisher_Index': fisher_indices,
        'Correlation_Index': correlation_indices,
        'T-test_Index': t_test_indices,
        'Mutual_Information_Index': mutual_info_indices
    })
    return indices_df

# ANFISによる特徴量選択とSVM最適化
def calculate_importance_degree(params, indices_df):
    weighted_scores = (indices_df['Fisher_Index'] * params[0] +
                       indices_df['Correlation_Index'] * params[1] +
                       indices_df['T-test_Index'] * params[2] +
                       indices_df['Mutual_Information_Index'] * params[3])
    importance_degree = np.where(weighted_scores > 0.75, 1, np.where(weighted_scores > 0.4, 0.5, 0))
    return importance_degree

def select_features_by_importance(importance_degree, features_df):
    selected_features = features_df.loc[:, importance_degree == 1]
    return selected_features

# PSOを用いた最適化
def optimize_anfis_svm_with_pso(X_train, y_train, X_val, y_val, indices_df):
    def objective(params):
        anfis_params = params[:4]
        C, gamma = params[4], params[5]
        importance_degree = calculate_importance_degree(anfis_params, indices_df)
        selected_features_train = select_features_by_importance(importance_degree, X_train)
        selected_features_val = select_features_by_importance(importance_degree, X_val)
        
        if selected_features_train.shape[1] > 0 and selected_features_val.shape[1] > 0:
            svm_model = SVC(kernel='rbf', C=C, gamma=gamma)
            svm_model.fit(selected_features_train, y_train)
            val_accuracy = accuracy_score(y_val, svm_model.predict(selected_features_val))
            return -val_accuracy
        else:
            return float('inf')

    # パラメータの範囲設定
    lb = [0, 0, 0, 0, 0.1, 0.001]
    ub = [1, 1, 1, 1, 10, 1]

    # PSOの実行
    best_params, _ = pso(objective, lb, ub, swarmsize=50, maxiter=100)

#    # tqdmで進捗バーを管理
#    max_iterations = 100
#    progress_bar = tqdm(total=max_iterations, desc="PSO Progress")
#
#    # カスタムPSO関数で進捗を追跡
#    def custom_pso(*args, **kwargs):
#        def wrapped_func(*wrapped_args, **wrapped_kwargs):
#            progress_bar.update(1)  # 各イテレーションごとに進捗を進める
#            return objective(*wrapped_args, **wrapped_kwargs)
#        
#        # 元のPSO関数を呼び出す
#        return pso(wrapped_func, *args, **kwargs)
#    
#    # 実行
#    best_params, _ = pso(
#        objective=lambda x: progress_bar.update(1) or objective(x),  # 進捗を更新しつつ目的関数を評価
#        lb=lb,
#        ub=ub,
#        swarmsize=50,
#        maxiter=max_iterations,
#        debug=False
#    )
#    
#    progress_bar.close()

    return best_params

#start_time = time.time()
## データの読み込み
#file_path = '../dataset/data_feat_Arefnezhad2019_all.csv'
##file_path = './dataset/data_raw_all_12.csv'
#data_new = pd.read_csv(file_path)
#
## ステアリング速度の計算
#time_diffs = data_new['row LSL time'].diff().dropna()
#sampling_rate = 1 / time_diffs.mean()
#print("sampling_rate",sampling_rate)
#data_new['Steering_Speed'] = data_new['Steering_Wheel_Pos'].diff().fillna(0) * sampling_rate
#
## 3秒間の時間窓で特徴量を抽出
#window_size = int(3 * sampling_rate)
#num_windows = len(data_new) // window_size
#features_list = []
#for i in tqdm(range(num_windows)):
#    angle_window = data_new['Steering_Wheel_Pos'][i*window_size:(i+1)*window_size]
#    speed_window = data_new['Steering_Speed'][i*window_size:(i+1)*window_size]
#    features = calculate_features_for_signals(angle_window, speed_window)
#    features_list.append(features)
#features_df = pd.DataFrame(features_list)
#features_df['drowsy'] = data_new['drowsy'][:num_windows * window_size:window_size].reset_index(drop=True)
#
#
## インデックスの計算とクリーンアップ
#label_df = features_df['drowsy']
#features_only_df = features_df.drop(columns=['drowsy']).replace([np.inf, -np.inf, np.nan], 0)
#indices_df = calculate_feature_indices(features_only_df, label_df)
#
#
## 訓練と検証データでの最適化実行
#X_train, X_temp, y_train, y_temp = train_test_split(features_only_df, label_df, test_size=0.4, random_state=42)
#X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

def SvmA_train(X_train, X_val, y_train, y_val, indices_df):
    # 最適化実行
    optimal_params = optimize_anfis_svm_with_pso(X_train, y_train, X_val, y_val, indices_df)
    
    # 最適化されたパラメータでANFISとSVMを設定
    best_anfis_params = optimal_params[:4]
    best_C, best_gamma = optimal_params[4], optimal_params[5]
    importance_degree = calculate_importance_degree(best_anfis_params, indices_df)
    
    # 選択された特徴量で再度評価
    selected_features_train = select_features_by_importance(importance_degree, X_train)
    selected_features_val = select_features_by_importance(importance_degree, X_val)
    #selected_features_test = select_features_by_importance(importance_degree, X_test)
    
    print(selected_features_train.columns.tolist())
    print(selected_features_val.columns.tolist())
    #print(selected_features_test.columns.tolist())
    
    # SVMモデルの訓練と評価
    #svm_model_final = SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True)
    svm_model_final = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
    svm_model_final.fit(selected_features_train, y_train)
    
    # 各セットでの精度を計算
    train_accuracy = accuracy_score(y_train, svm_model_final.predict(selected_features_train))
    val_accuracy = accuracy_score(y_val, svm_model_final.predict(selected_features_val))
    #test_accuracy = accuracy_score(y_test, svm_model_final.predict(selected_features_test))
    
    # テストデータでの予測確率を取得
    #y_test_prob = svm_model_final.predict_proba(X_test)[:, 1]
    
    # テストデータでの予測結果を取得して、混同行列を計算
    train_conf = confusion_matrix(y_train, svm_model_final.predict(selected_features_train))
    val_conf = confusion_matrix(y_val, svm_model_final.predict(selected_features_val))
    #test_conf = confusion_matrix(y_test, svm_model_final.predict(selected_features_test))
    
    print("train_conf   \n",train_conf)
    print("val_conf     \n",val_conf)
    #print("test_conf    \n",test_conf)
    
    # AUCの計算
    train_auc = roc_auc_score(y_train, svm_model_final.predict(selected_features_train))
    val_auc = roc_auc_score(y_val, svm_model_final.predict(selected_features_val))
    #test_auc = roc_auc_score(y_test, svm_model_final.predict(selected_features_test))
    print(f"AUC_train   : {train_auc:.3f}")
    print(f"AUC_val     : {val_auc:.3f}")
    #print(f"AUC_test    : {test_auc:.3f}")
    
    # 結果の表示
    {
        "Best ANFIS Parameters (Weights)": best_anfis_params,
        "Best SVM Parameters (C, gamma)": (best_C, best_gamma),
        "Train Accuracy": train_accuracy,
        "Validation Accuracy": val_accuracy,
        "Test Accuracy": test_accuracy,
    }
    
    # 結果の表示
    print(
            "results",
            "\nBest ANFIS Parameters (Weights):", best_anfis_params,
            "\nBest SVM Parameters (C, gamma) :", (best_C, best_gamma),
            "\nTrain Accuracy                 :", train_accuracy,
            "\nValidation Accuracy            :", val_accuracy,
            "\nTest Accuracy                  :", test_accuracy,
    )
    
    import csv
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    
    # ROCデータを保存する関数
    def save_roc_data(file_name, y_true, y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['FPR', 'TPR', 'Thresholds'])
            for i in range(len(fpr)):
                writer.writerow([fpr[i], tpr[i], thresholds[i]])
    
    # データの保存
    save_roc_data('roc_train.csv', y_train, svm_model_final.decision_function(selected_features_train))
    save_roc_data('roc_val.csv', y_val, svm_model_final.decision_function(selected_features_val))
    save_roc_data('roc_test.csv', y_test, svm_model_final.decision_function(selected_features_test))
    
    #
    ## ROC曲線を描画する関数
    #def plot_roc_curve(y_true, y_scores, label):
    #    fpr, tpr, _ = roc_curve(y_true, y_scores)
    #    plt.plot(fpr, tpr, label=f"{label} (AUC: {roc_auc_score(y_true, y_scores):.3f})")
    #
    ## 訓練、検証、テストデータセットのROC曲線を描画
    #plt.figure(figsize=(10, 8))
    #
    ## 訓練セット
    #train_scores = svm_model_final.decision_function(selected_features_train)
    #plot_roc_curve(y_train, train_scores, "Train")
    #
    ## 検証セット
    #val_scores = svm_model_final.decision_function(selected_features_val)
    #plot_roc_curve(y_val, val_scores, "Validation")
    #
    ## テストセット
    #test_scores = svm_model_final.decision_function(selected_features_test)
    #plot_roc_curve(y_test, test_scores, "Test")
    #
    ## グラフの装飾
    #plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    #plt.title("ROC Curve", fontsize=16)
    #plt.xlabel("False Positive Rate", fontsize=14)
    #plt.ylabel("True Positive Rate", fontsize=14)
    #plt.legend(fontsize=12)
    #plt.grid()
    #plt.show()
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # 各データセットの予測結果を取得
    train_pred = svm_model_final.predict(selected_features_train)
    val_pred = svm_model_final.predict(selected_features_val)
    test_pred = svm_model_final.predict(selected_features_test)
    
    # 指標の計算
    def calculate_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')  # 'binary' for binary classification
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        return accuracy, precision, recall, f1
    
    # Training metrics
    train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(y_train, train_pred)
    
    # Validation metrics
    val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(y_val, val_pred)
    
    # Test metrics
    test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(y_test, test_pred)
    
    # 結果の表示
    print("\n=== Training Metrics ===")
    print(f"Accuracy : {train_accuracy:.3f}")
    print(f"Precision: {train_precision:.3f}")
    print(f"Recall   : {train_recall:.3f}")
    print(f"F1 Score : {train_f1:.3f}")
    
    print("\n=== Validation Metrics ===")
    print(f"Accuracy : {val_accuracy:.3f}")
    print(f"Precision: {val_precision:.3f}")
    print(f"Recall   : {val_recall:.3f}")
    print(f"F1 Score : {val_f1:.3f}")
    
    print("\n=== Test Metrics ===")
    print(f"Accuracy : {test_accuracy:.3f}")
    print(f"Precision: {test_precision:.3f}")
    print(f"Recall   : {test_recall:.3f}")
    print(f"F1 Score : {test_f1:.3f}")
    
    
    print("elaped time:", time.time()-start_time)

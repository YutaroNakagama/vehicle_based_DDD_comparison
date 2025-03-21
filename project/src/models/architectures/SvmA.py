import warnings
warnings.filterwarnings("ignore")

# 必要なライブラリのインポート
import time
import pandas as pd
import numpy as np
import joblib
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
        print(params)
        
        if selected_features_train.shape[1] > 0 and selected_features_val.shape[1] > 0:
            svm_model = SVC(kernel='rbf', C=C, gamma=gamma)
            svm_model.fit(selected_features_train, y_train)
            val_accuracy = accuracy_score(y_val, svm_model.predict(selected_features_val))
            return -val_accuracy
        else:
            return 1.0 #float('inf')

    # パラメータの範囲設定
    lb = [0, 0, 0, 0, 0.1, 0.001]
    ub = [1, 1, 1, 1, 10, 1]

    # PSOの実行
    print('run pso')
    best_params, _ = pso(objective, lb, ub, swarmsize=3, maxiter=3)


    return best_params


def SvmA_train(X_train, X_val, y_train, y_val, indices_df, model):
    # 最適化実行
    print('get optiomal paras')
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


    # SVMモデルを保存
    joblib.dump(svm_model_final, f'model/{model}/svm_model_final.pkl')

    # 訓練用の特徴量データを保存
    joblib.dump(selected_features_train, f'model/{model}/selected_features_train.pkl')

    print("SVMモデルと訓練データの特徴量を保存しました。")
    
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
    #train_auc = roc_auc_score(y_train, svm_model_final.predict(selected_features_train))
    #val_auc = roc_auc_score(y_val, svm_model_final.predict(selected_features_val))
    #test_auc = roc_auc_score(y_test, svm_model_final.predict(selected_features_test))
    #print(f"AUC_train   : {train_auc:.3f}")
    #print(f"AUC_val     : {val_auc:.3f}")
    #print(f"AUC_test    : {test_auc:.3f}")
    
    # 結果の表示
    {
        "Best ANFIS Parameters (Weights)": best_anfis_params,
        "Best SVM Parameters (C, gamma)": (best_C, best_gamma),
        "Train Accuracy": train_accuracy,
        "Validation Accuracy": val_accuracy,
        #"Test Accuracy": test_accuracy,
    }
    
    # 結果の表示
    print(
            "results",
            "\nBest ANFIS Parameters (Weights):", best_anfis_params,
            "\nBest SVM Parameters (C, gamma) :", (best_C, best_gamma),
            "\nTrain Accuracy                 :", train_accuracy,
            "\nValidation Accuracy            :", val_accuracy,
            #"\nTest Accuracy                  :", test_accuracy,
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
    
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # 各データセットの予測結果を取得
    train_pred = svm_model_final.predict(selected_features_train)
    val_pred = svm_model_final.predict(selected_features_val)
    #test_pred = svm_model_final.predict(selected_features_test)
    
    # 指標の計算
    def calculate_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
#        precision = precision_score(y_true, y_pred, average='binary')  # 'binary' for binary classification
#        recall = recall_score(y_true, y_pred, average='binary')
#        f1 = f1_score(y_true, y_pred, average='binary')
        precision = precision_score(y_true, y_pred, average=None)  # 'binary' for binary classification
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        return accuracy, precision, recall, f1
    
    # Training metrics
    train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(y_train, train_pred)
    
    # Validation metrics
    val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(y_val, val_pred)
    
    # Test metrics
    #test_accuracy, test_precision, test_recall, test_f1 = calculate_metrics(y_test, test_pred)
    
    # 結果の表示
    print("\n=== Training Metrics ===")
    print(f"Accuracy : {train_accuracy}")
    print(f"Precision: {train_precision}")
    print(f"Recall   : {train_recall}")
    print(f"F1 Score : {train_f1}")
    
    print("\n=== Validation Metrics ===")
    print(f"Accuracy : {val_accuracy}")
    print(f"Precision: {val_precision}")
    print(f"Recall   : {val_recall}")
    print(f"F1 Score : {val_f1}")
    

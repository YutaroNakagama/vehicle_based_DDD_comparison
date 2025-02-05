import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Bidirectional, LSTM, Layer, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from scipy.stats import ttest_ind
import time
import matplotlib.pyplot as plt

# Attention Layerの定義
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# データの読み込みと特徴量選択
X_data = pd.read_csv('../X.csv').drop(columns=['Unnamed: 0'])
y_data = pd.read_csv('../y.csv').drop(columns=['Unnamed: 0']).values.flatten()

# T検定でp値が0.05未満の特徴量を選択
X_data['label'] = y_data
X_label_1 = X_data[X_data['label'] == 1].drop(columns=['label'])
X_label_0 = X_data[X_data['label'] == 0].drop(columns=['label'])

selected_features = []
for feature in X_label_1.columns:
    t_stat, p_val = ttest_ind(X_label_1[feature], X_label_0[feature], equal_var=False)
    if p_val < 0.05:
        selected_features.append(feature)

X_data = X_data[selected_features]  # 選択した特徴量のみで再構成
X_data_3d = np.expand_dims(X_data.values, axis=1)  # 3D変換 (サンプル数, タイムステップ数, 特徴量数)

# 最適化パラメータの範囲
batch_sizes = [16, 32, 64]
epochs = [10, 20, 30]

# グリッドサーチ用の変数
best_score = 0
best_params = {'batch_size': None, 'epochs': None}

for batch_size in batch_sizes:
    for epoch in epochs:
        print(f"\nTesting batch size = {batch_size}, epochs = {epoch}")
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracy_per_fold = []
        precision_per_fold = []
        recall_per_fold = []
        f1_per_fold = []
        auc_per_fold = []
        time_per_fold = []
        
        for train_index, test_index in kf.split(X_data_3d):
            X_train, X_test = X_data_3d[train_index], X_data_3d[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            
            # モデル構築
            inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
            x = Bidirectional(LSTM(50, return_sequences=True))(inputs)
            x = AttentionLayer()(x)
            x = Dense(20, activation='relu')(tf.expand_dims(x, axis=-1))
            x = Flatten()(x)
            x = Dense(1, activation='sigmoid')(x)
            model = Model(inputs, x)
            
            # モデルコンパイル
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # 早期終了設定
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            
            # モデル学習と実行時間計測
            start_time = time.time()
            model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.2, verbose=0, callbacks=[early_stopping])
            end_time = time.time()
            running_time = (end_time - start_time) * 1000
            
            # テストデータでの予測
            y_pred = (model.predict(X_test) > 0.5).astype("int32")
            y_pred_prob = model.predict(X_test)
            
            # 各評価指標の計算
            accuracy = accuracy_score(y_test, y_pred) * 100
            precision = precision_score(y_test, y_pred) * 100
            recall = recall_score(y_test, y_pred) * 100
            f1 = f1_score(y_test, y_pred) * 100
            auc = roc_auc_score(y_test, y_pred_prob) * 100
            
            # 評価結果をリストに追加
            accuracy_per_fold.append(accuracy)
            precision_per_fold.append(precision)
            recall_per_fold.append(recall)
            f1_per_fold.append(f1)
            auc_per_fold.append(auc)
            time_per_fold.append(running_time)
        
        # 各組み合わせの平均指標
        mean_accuracy = np.mean(accuracy_per_fold)
        mean_precision = np.mean(precision_per_fold)
        mean_recall = np.mean(recall_per_fold)
        mean_f1 = np.mean(f1_per_fold)
        mean_auc = np.mean(auc_per_fold)
        mean_running_time = np.mean(time_per_fold)
        
        print(f"Results for batch size = {batch_size}, epochs = {epoch}:")
        print(f"  Accuracy: {mean_accuracy:.2f}%")
        print(f"  Precision: {mean_precision:.2f}%")
        print(f"  Recall: {mean_recall:.2f}%")
        print(f"  F1 Score: {mean_f1:.2f}%")
        print(f"  AUC: {mean_auc:.2f}%")
        print(f"  Running Time: {mean_running_time:.2f} ms")
        
        # ベストスコアの更新
        if mean_accuracy > best_score:
            best_score = mean_accuracy
            best_params['batch_size'] = batch_size
            best_params['epochs'] = epoch

# 最適なバッチサイズとエポック数でモデルを再構築し、ROC曲線を描画
print(f"\nBest batch size: {best_params['batch_size']}, Best epochs: {best_params['epochs']}")
print(f"Best accuracy: {best_score:.2f}%")

# 5-foldクロスバリデーションの最後のモデルでROC曲線を表示
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X_data_3d):
    X_train, X_test = X_data_3d[train_index], X_data_3d[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = Bidirectional(LSTM(50, return_sequences=True))(inputs)
    x = AttentionLayer()(x)
    x = Dense(20, activation='relu')(tf.expand_dims(x, axis=-1))
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    final_model = Model(inputs, x)
    
    final_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    final_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_split=0.2, verbose=0)
    
    y_pred_prob = final_model.predict(X_test).ravel()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    break  # 最初のfoldのみを表示


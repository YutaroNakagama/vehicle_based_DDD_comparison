import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Bidirectional, LSTM, Layer, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

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

# データの読み込み
X_data = pd.read_csv('X.csv').drop(columns=['Unnamed: 0'])
y_data = pd.read_csv('y.csv').drop(columns=['Unnamed: 0'])

# ラベルを2値ラベル（1次元）に変換
y_data = y_data.values.flatten()

# LSTM用にデータを3Dに変換 (サンプル数, タイムステップ数, 特徴量数)
X_data_3d = np.expand_dims(X_data.values, axis=1)

# 5-fold cross-validation の設定
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
accuracy_per_fold = []

for train_index, test_index in kf.split(X_data_3d):
    # 訓練データとテストデータの分割
    X_train, X_test = X_data_3d[train_index], X_data_3d[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    
    # モデルの構築
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = Bidirectional(LSTM(50, return_sequences=True))(inputs)
    x = AttentionLayer()(x)
    x = Dense(20, activation='relu')(tf.expand_dims(x, axis=-1))  # AttentionLayerの出力を2Dに変換
    x = Flatten()(x)  # Flattenで2Dを1Dにする
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, x)
    
    # モデルのコンパイル
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 早期終了を設定
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # モデルの学習
    print(f'--- Starting fold {fold_no} ---')
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2, 
                        verbose=1, callbacks=[early_stopping])
    
    # テストデータでの評価
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Fold {fold_no} - Loss: {scores[0]}, Accuracy: {scores[1]}')
    accuracy_per_fold.append(scores[1])
    fold_no += 1

# 各foldの精度と平均精度の表示
print(f'\nScores per fold: {accuracy_per_fold}')
print(f'Average accuracy: {np.mean(accuracy_per_fold)}')


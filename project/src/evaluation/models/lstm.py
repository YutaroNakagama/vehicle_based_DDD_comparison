import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Bidirectional, LSTM, Layer, Input, Activation
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from src.config import MODEL_PKL_PATH


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

def lstm_eval(X,y,model_name):
    # データの読み込み
    X_test = X
    y_test = y
    
    # ラベルを2値ラベル（1次元）に変換
    y_test = y_test.values.flatten()
    
    # LSTM用にデータを3Dに変換 (サンプル数, タイムステップ数, 特徴量数)
    X_test_3d = np.expand_dims(X_test.values, axis=1)

    # どのfoldのモデルを使うか（例: fold1 のモデルを使用）
    fold_to_test = 1  # 任意のfold番号を設定

    # 保存したモデルをロード
    model = load_model(f'{MODEL_PKL_PATH}/{model_name}/lstm_model_fold{fold_to_test}.keras',
                        custom_objects={'AttentionLayer': AttentionLayer})

    print(f"Model lstm_model_fold{fold_to_test}.keras loaded successfully.")

    # モデルをテストデータで評価
    scores = model.evaluate(X_test_3d, y_test, verbose=0)

    # 結果を表示
    print(f'Loaded Model (Fold {fold_to_test}) - Loss: {scores[0]}, Accuracy: {scores[1]}')
    

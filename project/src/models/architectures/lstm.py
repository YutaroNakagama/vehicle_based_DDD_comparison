import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Bidirectional, LSTM, Layer, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from src.config import MODEL_PKL_PATH


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)


def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(50, return_sequences=True))(inputs)
    x = AttentionLayer()(x)
    x = Dense(20, activation='relu')(tf.expand_dims(x, axis=-1))
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def lstm_train(X, y, model_name, n_splits=5, epochs=10, batch_size=16):
    X_data = np.expand_dims(X.values, axis=1)
    y_data = y.values.flatten()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for fold_no, (train_idx, test_idx) in enumerate(kf.split(X_data), start=1):
        X_train, X_test = X_data[train_idx], X_data[test_idx]
        y_train, y_test = y_data[train_idx], y_data[test_idx]

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        print(f'--- Fold {fold_no} Training Start ---')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_split=0.2, verbose=1, callbacks=[early_stopping])

        scores = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(scores[1])

        model_path = f'{MODEL_PKL_PATH}/{model_name}/lstm_model_fold{fold_no}.keras'
        model.save(model_path)
        print(f'Model for fold {fold_no} saved at {model_path}')
        print(f'Fold {fold_no} - Loss: {scores[0]}, Accuracy: {scores[1]}')

    print(f'\nScores per fold: {accuracies}')
    print(f'Average accuracy: {np.mean(accuracies)}')


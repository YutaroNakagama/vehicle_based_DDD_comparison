"""LSTM-based neural network model with attention mechanism.

This module defines:
- A custom Keras AttentionLayer
- A Bidirectional LSTM model with attention
- A k-fold training routine with early stopping and model saving
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Bidirectional, LSTM, Layer, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from src.config import MODEL_PKL_PATH


class AttentionLayer(Layer):
    """
    Custom attention layer for sequence input.

    Applies learned weights over LSTM outputs to focus on important time steps.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments passed to the base Layer class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Initialize weights and biases for attention."""
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer=tf.keras.initializers.GlorotUniform())
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super().build(input_shape)

    def call(self, x):
        """Forward pass: apply attention to input sequence.

        Args:
            x (tensor): LSTM output tensor of shape (batch, timesteps, features).

        Returns:
            tensor: Aggregated context vector (batch, features).
        """
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)


def build_lstm_model(input_shape: tuple) -> Model:
    """
    Construct the Bidirectional LSTM model with attention.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor in the form ``(timesteps, features)``.

    Returns
    -------
    tensorflow.keras.Model
        Compiled LSTM model with attention mechanism.
    """
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(50, return_sequences=True))(inputs)
    x = AttentionLayer()(x)
    x = Dense(20, activation='relu')(tf.expand_dims(x, axis=-1))
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def lstm_train(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_splits: int = 5,
    epochs: int = 10,
    batch_size: int = 16
) -> None:
    """
    Train a Bidirectional LSTM model with an attention mechanism using k-fold cross-validation.

    This function performs preprocessing, model construction, training, evaluation, and saving of
    an LSTM model using a k-fold strategy. It selects numeric features, handles missing or infinite
    values, and standardizes input before model training. Each fold’s model and scaler are saved.

    Parameters
    ----------
    X : pandas.DataFrame
        Input feature matrix. Only numeric columns are used.
    y : pandas.Series
        Binary class labels (0 or 1) corresponding to ``X``.
    model_name : str
        Name used for saving the model and scaler (e.g., ``"Lstm"``) under ``MODEL_PKL_PATH``.
    n_splits : int, default=5
        Number of folds for K-Fold cross-validation.
    epochs : int, default=10
        Number of training epochs for each fold.
    batch_size : int, default=16
        Batch size used during training.

    Returns
    -------
    None
        Trained models and scalers are saved to disk.
    """
    import joblib

    # Extract numeric columns
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Convert inf/-inf to NaN
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN and align corresponding labels
    X_numeric = X_numeric.dropna()
    y_cleaned = y.loc[X_numeric.index]
    
    # Clip values to fit within float32 range
    X_numeric = X_numeric.clip(lower=np.finfo(np.float32).min, upper=np.finfo(np.float32).max)
    
    # Convert to float32
    X_array = X_numeric.astype(np.float32).values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for fold_no, (train_idx, test_idx) in enumerate(kf.split(X_array), start=1):
        # Standardize (fit per fold)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array[train_idx])
        X_scaled_test = scaler.transform(X_array[test_idx])

        # Reshape for LSTM
        X_train = np.expand_dims(X_scaled, axis=1)
        X_test = np.expand_dims(X_scaled_test, axis=1)
        y_train = y_cleaned.values[train_idx]
        y_test = y_cleaned.values[test_idx]

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        print(f'--- Fold {fold_no} Training Start ---')
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stopping]
        )

        scores = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(scores[1])

        # Save model
        model_path = f'{MODEL_PKL_PATH}/{model_name}/lstm_model_fold{fold_no}.keras'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)

        # Save scaler
        scaler_path = f"{MODEL_PKL_PATH}/{model_name}/scaler_fold{fold_no}.pkl"
        joblib.dump(scaler, scaler_path)

        print(f'Model for fold {fold_no} saved at {model_path}')
        print(f'Scaler for fold {fold_no} saved at {scaler_path}')
        print(f'Fold {fold_no} - Loss: {scores[0]}, Accuracy: {scores[1]}')

    print(f'\nScores per fold: {accuracies}')
    print(f'Average accuracy: {np.mean(accuracies)}')


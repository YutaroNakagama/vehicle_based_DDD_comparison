"""Evaluation module for LSTM model with attention mechanism.

Loads a trained LSTM model and evaluates it on test data,
reporting accuracy, classification report, and confusion matrix.
"""

import logging
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from src.config import MODEL_PKL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AttentionLayer(Layer):
    """Custom attention layer for LSTM model."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Initialize attention weights and bias."""
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        """Apply attention mechanism to input sequence.

        Args:
            x (tensor): Input sequence of shape (batch, timesteps, features)

        Returns:
            tensor: Aggregated output with attention applied
        """
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)


def lstm_eval(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    clf=None,
    scaler=None,
    fold_to_test: int = 1
) -> dict:
    """
    Evaluate a saved LSTM model with attention on test data.

    Loads the model and its scaler if not provided, applies consistent
    preprocessing, and evaluates performance.

    Parameters
    ----------
    X_test : pandas.DataFrame
        Test feature matrix. Only numeric columns are used.
    y_test : pandas.Series
        Ground truth labels corresponding to ``X_test``.
    model_name : str
        Name of the model directory under ``model/``.
    clf : keras.Model, optional
        Preloaded LSTM model. If ``None``, the model is loaded from disk.
    scaler : sklearn.preprocessing.StandardScaler, optional
        Preloaded scaler for feature normalization. If ``None``, the scaler is
        loaded from disk.
    fold_to_test : int, default=1
        Fold index used to load the correct model and scaler from disk.

    Returns
    -------
    dict
        Dictionary containing evaluation results:

        - ``"loss"`` : float
            Test loss.
        - ``"accuracy"`` : float
            Test accuracy.
        - ``"classification_report"`` : dict
            Classification report in scikit-learn format.
        - ``"confusion_matrix"`` : list of list of int
            Confusion matrix as a nested list.
    """

    # Load model and scaler if not provided
    if clf is None or scaler is None:
        # Load from disk based on fold
        scaler_path = f"{MODEL_PKL_PATH}/{model_name}/scaler_fold{fold_to_test}.pkl"
        scaler = joblib.load(scaler_path)
    
        model_path = f"{MODEL_PKL_PATH}/{model_name}/lstm_model_fold{fold_to_test}.keras"
        clf = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})

        model = clf
    else:
        model = clf

    # Preprocessing: numeric only, clean NaN/inf
    X_numeric = X_test.select_dtypes(include=[np.number])
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).dropna()
    y_test = y_test.loc[X_numeric.index].values.flatten()

    X_numeric = X_numeric.clip(np.finfo(np.float32).min, np.finfo(np.float32).max)
    X_array = X_numeric.astype(np.float32).values

    # Apply scaling
    X_scaled = scaler.transform(X_array)

    # Reshape into LSTM input format
    X_test_3d = np.expand_dims(X_scaled, axis=1)

    # Evaluation
    loss, accuracy = model.evaluate(X_test_3d, y_test, verbose=0)
    logging.info(f"Evaluation Results - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Prediction and report
    y_pred_prob = model.predict(X_test_3d)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info("Classification Report:\n" + report)
    logging.info("Confusion Matrix:\n" + str(conf_matrix))

    return {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': conf_matrix.tolist()
    }


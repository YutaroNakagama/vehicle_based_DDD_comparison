"""Evaluation module for LSTM model with attention mechanism.

Loads a trained LSTM model and evaluates it on test data,
reporting accuracy, classification report, and confusion matrix.
"""

import numpy as np
import pandas as pd
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow as tf
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


def lstm_eval(X_test: pd.DataFrame, y_test: pd.Series, model_name: str, fold_to_test: int = 1) -> dict:
    """Evaluate a saved LSTM model with attention on test data.

    Loads the model, reshapes input data, and reports classification metrics.

    Args:
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        model_name (str): Name of model directory under `model/`.
        fold_to_test (int): Fold index to load the correct model file.

    Returns:
        dict: Evaluation results including:
              - 'loss': Test loss
              - 'accuracy': Test accuracy
              - 'classification_report': Text report
              - 'confusion_matrix': Confusion matrix array
    """
    y_test = y_test.values.flatten()
    X_test_3d = np.expand_dims(X_test.values, axis=1)

    model_path = f'{MODEL_PKL_PATH}/{model_name}/lstm_model_fold{fold_to_test}.keras'
    model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})

    logging.info(f"Model '{model_path}' loaded successfully.")

    loss, accuracy = model.evaluate(X_test_3d, y_test, verbose=0)
    logging.info(f"Evaluation Results - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    y_pred_prob = model.predict(X_test_3d)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info("Classification Report:\n" + report)
    logging.info("Confusion Matrix:\n" + str(conf_matrix))

    return {
        'loss': loss,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }


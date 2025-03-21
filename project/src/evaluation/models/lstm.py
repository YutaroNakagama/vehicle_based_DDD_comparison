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

def lstm_eval(X_test, y_test, model_name, fold_to_test=1):
    """
    Evaluate the LSTM model on test data.

    Parameters:
    - X_test: Test features as a DataFrame
    - y_test: Test labels as a Series
    - model_name: Name of the model for loading the saved model
    - fold_to_test: The fold number of the trained model to evaluate
    """

    # Convert labels to a flat numpy array
    y_test = y_test.values.flatten()

    # Reshape features for LSTM input: (samples, time_steps, features)
    X_test_3d = np.expand_dims(X_test.values, axis=1)

    # Load the pre-trained model
    model_path = f'{MODEL_PKL_PATH}/{model_name}/lstm_model_fold{fold_to_test}.keras'
    model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})

    logging.info(f"Model '{model_path}' loaded successfully.")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_3d, y_test, verbose=0)
    logging.info(f"Evaluation Results - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Generate predictions and additional metrics
    y_pred_prob = model.predict(X_test_3d)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Detailed metrics
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


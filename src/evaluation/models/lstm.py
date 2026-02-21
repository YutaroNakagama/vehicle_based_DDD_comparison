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

from src.config import MODEL_PKL_PATH, LSTM_SEGMENT_TIMESTEPS
from src.models.architectures.lstm import create_segments, configure_gpu
from src.evaluation.metrics import (
    calculate_extended_metrics,
    calculate_class_specific_metrics,
    find_optimal_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class AttentionLayer(Layer):
    """Custom attention layer for LSTM model (48-unit Bahdanau attention)."""

    def __init__(self, units: int = 48, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        """Initialize attention weights, bias, and context vector."""
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
        )
        self.b = self.add_weight(
            name="att_bias", shape=(self.units,), initializer="zeros"
        )
        self.v = self.add_weight(
            name="att_context",
            shape=(self.units, 1),
            initializer="glorot_uniform",
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        """Apply attention mechanism to input sequence.

        Parameters
        ----------
        x : tensor
            Input sequence of shape (batch, timesteps, features)

        Returns
        -------
        tensor
            Aggregated output with attention applied
        """
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        score = tf.keras.backend.dot(e, self.v)
        a = tf.keras.backend.softmax(score, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


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
        Test feature matrix.  When called from ``eval_pipeline``, the data
        has already been cleaned, aligned, and scaled by
        ``prepare_evaluation_features()``.  Only when *both* ``clf`` and
        ``scaler`` are ``None`` (standalone mode) does this function load
        artifacts from disk and apply scaling itself.
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

    # ------------------------------------------------------------------
    # Configure GPU if available
    # ------------------------------------------------------------------
    configure_gpu()

    # ------------------------------------------------------------------
    # Standalone mode: load model/scaler from disk & preprocess
    # ------------------------------------------------------------------
    if clf is None or scaler is None:
        scaler_path = f"{MODEL_PKL_PATH}/{model_name}/scaler_fold{fold_to_test}.pkl"
        scaler = joblib.load(scaler_path)

        model_path = f"{MODEL_PKL_PATH}/{model_name}/lstm_model_fold{fold_to_test}.keras"
        clf = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})

        model = clf

        # Preprocessing only needed in standalone mode
        X_numeric = X_test.select_dtypes(include=[np.number])
        X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).dropna()
        y_test = y_test.loc[X_numeric.index].values.flatten()

        X_numeric = X_numeric.clip(np.finfo(np.float32).min, np.finfo(np.float32).max)
        X_array = X_numeric.astype(np.float32).values

        # Apply scaling (only once — data is raw here)
        X_scaled = scaler.transform(X_array)
    else:
        # ------------------------------------------------------------------
        # Pipeline mode: data already cleaned, aligned, and scaled by
        # prepare_evaluation_features() — do NOT re-scale.
        # ------------------------------------------------------------------
        model = clf

        X_array = X_test.values.astype(np.float32)
        # Handle any residual NaN/inf (should be rare after prepare_evaluation_features)
        mask = np.isfinite(X_array).all(axis=1)
        if not mask.all():
            logging.warning(
                f"[LSTM] Dropping {(~mask).sum()} rows with NaN/inf from pre-scaled data"
            )
            X_array = X_array[mask]
            y_test = y_test[mask] if hasattr(y_test, '__getitem__') else y_test

        X_scaled = X_array

    # Flatten y_test to 1-D numpy array
    if hasattr(y_test, 'values'):
        y_test = y_test.values.flatten()
    elif isinstance(y_test, np.ndarray) and y_test.ndim > 1:
        y_test = y_test.flatten()

    # Create fixed-length segments for LSTM input
    seg_len = LSTM_SEGMENT_TIMESTEPS
    X_test_3d, y_test = create_segments(X_scaled, y_test, seg_len)
    logging.info(
        f"[LSTM] Eval segments: {X_test_3d.shape[0]} "
        f"(seg_len={seg_len}, features={X_test_3d.shape[2]})"
    )

    # Evaluation
    eval_scores = model.evaluate(X_test_3d, y_test, verbose=0)
    # eval_scores may be [loss, acc] or [loss, acc, auc] depending on model
    loss = eval_scores[0]
    accuracy = eval_scores[1]
    logging.info(f"Evaluation Results - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Prediction and report
    y_pred_prob = model.predict(X_test_3d)
    y_score = y_pred_prob.flatten()           # sigmoid probabilities

    # --- Optimal threshold via F2-score (emphasize recall for drowsiness) ---
    opt_threshold, opt_f2 = find_optimal_threshold(y_test, y_score, beta=2.0)
    logging.info(
        f"[LSTM] Optimal threshold: {opt_threshold:.4f} (F2={opt_f2:.4f})"
    )
    y_pred = (y_score >= opt_threshold).astype(int)

    report_str = classification_report(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info("Classification Report:\n" + report_str)
    logging.info("Confusion Matrix:\n" + str(conf_matrix))

    # --- Compute extended metrics (ROC-AUC, AUPRC) via shared library ---
    ext = calculate_extended_metrics(
        y_test, y_pred, y_score,
        zero_division=0,
        include_roc=True,
        include_pr=True,
    )

    # Extract positive-class metrics from classification report
    class_metrics = calculate_class_specific_metrics(
        np.array(conf_matrix), report_dict
    )

    result = {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'classification_report': report_dict,
        'confusion_matrix': conf_matrix.tolist(),
        # ROC-AUC and AUPRC
        'roc_auc': ext.get('roc_auc'),
        'auc_pr':  ext.get('auc_pr'),
        'custom_threshold': float(opt_threshold),
    }
    # Merge positive-class metrics (precision_pos, recall_pos, f1_pos, ...)
    result.update(class_metrics)

    logging.info(f"ROC AUC: {result.get('roc_auc'):.4f}" if result.get('roc_auc') is not None else "ROC AUC: N/A")
    logging.info(f"AUPRC:   {result.get('auc_pr'):.4f}" if result.get('auc_pr') is not None else "AUPRC: N/A")

    return result


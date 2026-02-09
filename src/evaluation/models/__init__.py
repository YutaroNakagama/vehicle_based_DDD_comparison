"""Expose evaluation entry functions for all model types."""
from src.evaluation.models.SvmA import SvmA_eval
from src.evaluation.models.common import common_eval

# Lazy import for lstm_eval to avoid TensorFlow dependency at import time
def lstm_eval(*args, **kwargs):
    from src.evaluation.models.lstm import lstm_eval as _lstm_eval
    return _lstm_eval(*args, **kwargs)

__all__ = ["lstm_eval", "SvmA_eval", "common_eval"]

"""Expose evaluation entry functions for all model types."""
from src.evaluation.models.lstm import lstm_eval
from src.evaluation.models.SvmA import SvmA_eval
from src.evaluation.models.common import common_eval
__all__ = ["lstm_eval", "SvmA_eval", "common_eval"]

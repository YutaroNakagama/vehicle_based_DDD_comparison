"""Training module - contains training pipeline, optimization, and evaluation.

Submodules:
- pipeline: Main training pipeline (common_train)
- dispatch: Model-specific training dispatch (train_model)
- model_factory: Classifier instantiation (get_classifier)
- classifiers: Classifier creation and calibration
- optuna_tuning: Hyperparameter optimization with Optuna
- evaluation: Model evaluation helpers
"""

# Lazy imports to avoid circular dependencies and heavy module loading
__all__ = ["common_train", "train_model", "get_classifier"]


def __getattr__(name):
    """Lazy import for heavy modules."""
    if name == "common_train":
        from src.models.training.pipeline import common_train
        return common_train
    elif name == "train_model":
        from src.models.training.dispatch import train_model
        return train_model
    elif name == "get_classifier":
        from src.models.training.model_factory import get_classifier
        return get_classifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Data augmentation module.

This module provides data augmentation techniques for handling class imbalance:

Submodules
----------
oversampling : SMOTE-based oversampling and jitter/scale augmentation
jitter : Time-series jittering augmentation
coral : CORAL domain adaptation
domain_mixup : Domain mixup augmentation
vae_augment : VAE-based augmentation
"""

# Re-export main functions from oversampling for backward compatibility
from .oversampling import (
    augment_minority_class,
    jitter_features,
    scale_features,
    jitter_scale_features,
    estimate_adaptive_sigma,
    analyze_augmentation_quality,
)

__all__ = [
    # Main augmentation functions
    "augment_minority_class",
    "jitter_features",
    "scale_features",
    "jitter_scale_features",
    "estimate_adaptive_sigma",
    "analyze_augmentation_quality",
]

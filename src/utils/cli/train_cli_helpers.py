"""Helper functions for training and evaluation CLI scripts.

This module encapsulates reusable logic for:
- Loading target subject IDs from a text file
- Logging parsed arguments
- Mapping mode → split strategy
- Common argument parser setup

Unified CLI Design:
-------------------
Both train.py and evaluate.py share common arguments:
  --model         : Model architecture (unified from train's --model_name)
  --mode          : Training/evaluation mode (pooled/target_only/source_only/joint_train)
  --target_file   : Path to target subject IDs file
  --seed          : Random seed (default: config.DEFAULT_RANDOM_SEED)
  --tag           : Optional variant tag (e.g., 'coral', 'erm')

Training-specific arguments (train.py only):
  --subject_wise_split   : Enable subject-wise splitting
  --time_stratify_labels : Enable time-stratified splitting
  
Evaluation-specific arguments (evaluate.py only):
  --sample_size   : Number of subjects to evaluate (subset evaluation)
  --fold          : CV fold number (0 = no fold)
  --jobid         : Explicit training job ID (auto-detect if omitted)

Configuration:
--------------
Time stratification parameters are centralized in src/config.py:
  - TIME_STRATIFY_TOLERANCE
  - TIME_STRATIFY_WINDOW
  - TIME_STRATIFY_MIN_CHUNK

Keeping this logic separate makes CLI scripts minimal and easier to reuse
in HPC pipelines or Jupyter environments.
"""

import os
import logging
import argparse
from typing import List, Optional
from src.config import MODEL_CHOICES, DEFAULT_RANDOM_SEED


def load_subjects_from_file(path: str) -> List[str]:
    """Load subject IDs from a text file.

    Supports space, comma, or newline separation.
    Raises clear errors if the file is missing or empty.
    
    Parameters
    ----------
    path : str
        Path to the text file containing subject IDs.
    
    Returns
    -------
    list of str
        List of subject IDs.
    
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If no valid subject IDs are found in the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Target subject file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        subjects = [
            s for s in text.replace(",", " ").replace("\t", " ").split()
            if s.strip()
        ]

    if not subjects:
        raise ValueError(f"[ERROR] No valid subject IDs found in file: {path}")

    logging.info(
        f"[INFO] Loaded {len(subjects)} subjects from {path}: "
        f"{subjects[:5]}{'...' if len(subjects) > 5 else ''}"
    )
    return subjects


def log_train_args(args, target_subjects: List[str]) -> None:
    """Log all parsed training arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    target_subjects : list of str
        List of target subject IDs.
    """
    logging.info(
        "[RUN] model=%s | mode=%s | seed=%s | tag=%s | target_file=%s | subjects=%s",
        args.model,
        args.mode,
        args.seed,
        args.tag,
        args.target_file if args.target_file else "None",
        " ".join(target_subjects) if target_subjects else "None",
    )


def log_eval_args(args, target_subjects: Optional[List[str]] = None) -> None:
    """Log all parsed evaluation arguments.
    
    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    target_subjects : list of str, optional
        List of target subject IDs.
    """
    logging.info(
        "[EVAL] model=%s | mode=%s | tag=%s | seed=%s | jobid=%s | target_file=%s",
        args.model,
        args.mode,
        args.tag or "(none)",
        args.seed,
        args.jobid or "(auto)",
        args.target_file if args.target_file else "None",
    )


def map_mode_to_strategy(mode: str) -> str:
    """Map `--mode` argument to internal split strategy.
    
    Parameters
    ----------
    mode : str
        Mode string (pooled, target_only, source_only, joint_train).
    
    Returns
    -------
    str
        Split strategy name.
    
    Raises
    ------
    ValueError
        If mode is not recognized.
    """
    mapping = {
        "pooled": "random",
        "target_only": "subject_time_split",
        "source_only": "subject_wise_split",
        "joint_train": "finetune_target_subjects",
    }
    if mode not in mapping:
        raise ValueError(f"Unknown mode: {mode}")
    return mapping[mode]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments shared by train and evaluate scripts.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser to add arguments to.
    """
    parser.add_argument(
        "--mode",
        choices=["pooled", "target_only", "source_only", "joint_train"],
        required=True,
        help="Training/evaluation mode (pooled / target_only / source_only / joint_train).",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default=None,
        help="Path to file containing target subject IDs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Random seed (default: {DEFAULT_RANDOM_SEED}).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for saved outputs.",
    )


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Add training-specific arguments.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser to add arguments to.
    """
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        required=True,
        help=f"Model architecture. Choices: {', '.join(MODEL_CHOICES)}",
    )
    parser.add_argument(
        "--subject_wise_split",
        action="store_true",
        help="Use subject-wise splitting to avoid data leakage.",
    )
    
    # Time-stratified splitting flag (parameters now in config.py)
    parser.add_argument(
        "--time_stratify_labels",
        action="store_true",
        help="Enable time-stratified splitting with label balance (uses config.TIME_STRATIFY_*).",
    )
    
    # Oversampling options for class imbalance
    parser.add_argument(
        "--use_oversampling",
        action="store_true",
        help="Enable oversampling of minority class in training data.",
    )
    parser.add_argument(
        "--oversample_method",
        choices=["smote", "adasyn", "borderline", "smote_tomek", "smote_enn", "smote_rus", "jitter", "scale", "jitter_scale", "undersample_rus", "undersample_tomek"],
        default="smote",
        help="Sampling method: SMOTE, ADASYN, BorderlineSMOTE, SMOTE+Tomek, SMOTE+ENN, SMOTE+RUS, Jitter, Scale, Jitter+Scale, Undersample-RUS, or Undersample-Tomek (default: smote).",
    )
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.33,
        help="Target minority/majority ratio for oversampling (default: 0.33).",
    )
    parser.add_argument(
        "--subject_wise_oversampling",
        action="store_true",
        help="Apply oversampling separately for each subject to avoid mixing subject characteristics.",
    )


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    """Add evaluation-specific arguments.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        Argument parser to add arguments to.
    """
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        required=True,
        help=f"Model to evaluate. Choices: {', '.join(MODEL_CHOICES)}",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of subjects to evaluate (for subset evaluation).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number for cross validation (0 = no fold).",
    )
    parser.add_argument(
        "--subject_wise_split",
        action="store_true",
        help="Use subject-wise data splitting to prevent subject leakage.",
    )
    parser.add_argument(
        "--jobid",
        type=str,
        default=None,
        help="Specify training job ID (e.g., 14004123). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom prediction threshold (0.0-1.0). If not specified, uses default 0.5 or optimized threshold.",
    )


def setup_logging(level: int = logging.INFO) -> None:
    """Setup logging configuration with consistent format.
    
    Parameters
    ----------
    level : int, default=logging.INFO
        Logging level.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

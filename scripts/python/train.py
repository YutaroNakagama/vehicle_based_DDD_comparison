"""Minimal training CLI for Driver Drowsiness Detection (DDD).

Used primarily by HPC job scripts:
  - scripts/hpc/train/pbs_train.sh
  - scripts/hpc/domain_analysis/pbs_rank.sh

This script provides a lightweight command-line interface for training models
on the processed DDD dataset. Subject lists are loaded from external text files,
enabling flexible batch execution without direct ID specification.

Supported modes
---------------
- **pooled**: Train/test split over all subjects (domain-agnostic baseline)
- **target_only**: Use only target subjects (within-domain evaluation)
- **source_only**: Use all non-target subjects (exclude target subjects)
- **joint_train**: Combine target and source subjects (mixed-domain training)
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# --- Path setup ---
THIS = Path(__file__).resolve()
PRJ = THIS.parents[1]  # vehicle_based_DDD_comparison/
sys.path.append(str(PRJ))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Thread limiting for deterministic runs ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Imports ---
import src.config
import src.models.model_pipeline as mp


def load_subjects_from_file(path: str):
    """Load subject IDs from a text file.

    Parameters
    ----------
    path : str
        Path to a text file containing subject IDs (space, comma, or newline-separated).

    Returns
    -------
    list of str
        List of subject identifiers read from the file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
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


def log_train_args(args, target_subjects):
    """Log all parsed training arguments."""
    logging.info(
        "[RUN] model=%s | mode=%s | seed=%s | tag=%s | target_file=%s | subjects=%s",
        args.model,
        args.mode,
        args.seed,
        args.tag,
        args.target_file if args.target_file else "None",
        " ".join(target_subjects) if target_subjects else "None",
    )


def main():
    """Parse command-line arguments and execute the training pipeline.

    Parameters
    ----------
    None

    Command-line Options
    --------------------
    --model : str
        Required. Model architecture to train.
        Must be one of ``src.config.MODEL_CHOICES``.
    --mode : {"pooled", "target_only", "source_only", "joint_train"}
        Training mode controlling which subjects are used.
        - ``pooled``       : Train/test over all subjects (domain-agnostic baseline)
        - ``target_only``  : Use only target subjects (within-domain split)
        - ``source_only``  : Use all non-target subjects (exclude targets)
        - ``joint_train``  : Combine both target and source subjects
    --target_file : str, optional
        Path to a text file containing target subject IDs (space, comma, or newline-separated).
        Required for all modes except ``pooled``.
    --subject_wise_split : bool, optional
        If True, prevents data leakage across subjects.
    --seed : int, default=42
        Random seed for reproducibility.
    --tag : str, optional
        Tag name used for model and result file naming.
    --time_stratify_labels : bool, optional
        If True, enforces class ratio consistency across temporal splits.
    --time_stratify_tolerance : float, default=0.02
        Allowed deviation in positive ratio between splits.
    --time_stratify_window : float, default=0.10
        Window proportion (±) for adjusting temporal boundaries.
    --time_stratify_min_chunk : int, default=100
        Minimum number of samples per split.
    """
    parser = argparse.ArgumentParser(
        description="Lightweight training CLI for DDD experiments (file-based subject lists)."
    )

    # --- Core arguments ---
    parser.add_argument("--model", choices=src.config.MODEL_CHOICES, required=True)
    parser.add_argument(
        "--mode",
        choices=["pooled", "target_only", "source_only", "joint_train"],
        required=True,
        help="Training mode (pooled / target_only / source_only / joint_train).",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default=None,
        help="Path to file containing target subject IDs.",
    )
    parser.add_argument(
        "--subject_wise_split",
        action="store_true",
        help="Use subject-wise splitting to avoid data leakage.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag for saved outputs.")

    # --- Time-stratified splitting options ---
    parser.add_argument("--time_stratify_labels", action="store_true")
    parser.add_argument("--time_stratify_tolerance", type=float, default=0.02)
    parser.add_argument("--time_stratify_window", type=float, default=0.10)
    parser.add_argument("--time_stratify_min_chunk", type=int, default=100)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # --- Load target subjects if required ---
    target_subjects = []
    if args.mode != "pooled":
        if not args.target_file:
            raise SystemExit(f"[ERROR] --target_file is required for mode={args.mode}")
        target_subjects = load_subjects_from_file(args.target_file)

    # --- Mode logic mapping ---
    if args.mode == "pooled":
        args.subject_split_strategy = "random"
    elif args.mode == "target_only":
        args.subject_split_strategy = "subject_time_split"
    elif args.mode == "source_only":
        args.subject_split_strategy = "subject_wise_split"
    elif args.mode == "joint_train":
        args.subject_split_strategy = "finetune_target_subjects"

    # --- Execute training pipeline ---
    pipeline_args = {
        "model_name": args.model,
        "mode": args.mode,
        "target_subjects": target_subjects,
        "subject_wise_split": args.subject_wise_split,
        "seed": args.seed,
        "tag": args.tag,
        "subject_split_strategy": args.subject_split_strategy,
        "time_stratify_labels": args.time_stratify_labels,
        "time_stratify_tolerance": args.time_stratify_tolerance,
        "time_stratify_window": args.time_stratify_window,
        "time_stratify_min_chunk": args.time_stratify_min_chunk,
    }

    log_train_args(args, target_subjects)
    mp.train_pipeline(**pipeline_args)


if __name__ == "__main__":
    main()


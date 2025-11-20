"""Minimal training CLI for Driver Drowsiness Detection (DDD).

Used primarily by HPC job scripts:
  - scripts/hpc/train/pbs_train.sh
  - scripts/hpc/domain_analysis/pbs_rank.sh

Provides a lightweight command-line interface for training models on the
processed DDD dataset. Subject lists are loaded from external text files.

CLI Options (summary):
  --model_name <name>     Required. One of src.config.MODEL_CHOICES
  --mode <mode>           pooled | target_only | source_only | joint_train
  --target_file <path>    Path to target subject IDs file (required unless pooled)
  --subject_wise_split    Enable subject-wise splitting (avoid leakage)
  --seed <int>            Random seed (default: 42)
  --tag <str>             Optional artifact suffix
  --time_stratify_labels  Enable time/label stratified splitting
  --time_stratify_tolerance <float>  Positive ratio tolerance (default: 0.02)
  --time_stratify_window <float>     Boundary search window fraction (default: 0.10)
  --time_stratify_min_chunk <int>    Minimum rows per split (default: 100)
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from src.utils.cli.train_cli_helpers import load_subjects_from_file, log_train_args, map_mode_to_strategy

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


def main():
    """Parse command-line arguments and execute the training pipeline.

    Other Parameters
    ----------------
    See module docstring CLI summary for full list. Key options:
    --model_name, --mode, --target_file, stratification flags.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Lightweight training CLI for DDD experiments (file-based subject lists)."
    )

    # --- Core arguments ---
    parser.add_argument("--model_name", choices=src.config.MODEL_CHOICES, required=True)
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname%s - %(message)s")

    # --- Load subjects and map mode ---
    target_subjects = load_subjects_from_file(args.target_file) if args.mode != "pooled" else []
    split_strategy = map_mode_to_strategy(args.mode)

    log_train_args(args, target_subjects)
    mp.train_pipeline(
        model_name=args.model_name,
        mode=args.mode,
        target_subjects=target_subjects,
        subject_wise_split=args.subject_wise_split,
        seed=args.seed,
        tag=args.tag,
        subject_split_strategy=split_strategy,
        time_stratify_labels=args.time_stratify_labels,
        time_stratify_tolerance=args.time_stratify_tolerance,
        time_stratify_window=args.time_stratify_window,
        time_stratify_min_chunk=args.time_stratify_min_chunk,
    )


if __name__ == "__main__":
    main()


"""Minimal training CLI for Driver Drowsiness Detection (DDD).

Used primarily by HPC job scripts:
  - scripts/hpc/train/pbs_train.sh
  - scripts/hpc/domain_analysis/pbs_rank.sh

Provides a lightweight command-line interface for training models on the
processed DDD dataset. Subject lists are loaded from external text files.

CLI Options (summary):
  --model <name>          Required. One of src.config.MODEL_CHOICES
  --mode <mode>           pooled | target_only | source_only | joint_train
  --target_file <path>    Path to target subject IDs file (required unless pooled)
  --subject_wise_split    Enable subject-wise splitting (avoid leakage)
  --seed <int>            Random seed (default: from config.DEFAULT_RANDOM_SEED)
  --tag <str>             Optional artifact suffix
  --time_stratify_labels  Enable time/label stratified splitting

Note:
  Time stratification parameters (tolerance, window, min_chunk) are now
  configured in src.config (TIME_STRATIFY_*).
"""

import sys
import os
import argparse
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
from src import config as cfg
from src.utils.cli.train_cli_helpers import (
    load_subjects_from_file,
    log_train_args,
    map_mode_to_strategy,
    add_common_arguments,
    add_train_arguments,
    setup_logging,
)
import src.models.model_pipeline as mp


def main():
    """Parse command-line arguments and execute the training pipeline.

    Other Parameters
    ----------------
    See module docstring CLI summary for full list. Key options:
    --model, --mode, --target_file, stratification flags.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Lightweight training CLI for DDD experiments (file-based subject lists)."
    )

    # Add common and training-specific arguments
    add_common_arguments(parser)
    add_train_arguments(parser)

    args = parser.parse_args()
    setup_logging()

    # --- Load subjects and map mode ---
    target_subjects = load_subjects_from_file(args.target_file) if args.mode != "pooled" else []
    split_strategy = map_mode_to_strategy(args.mode)

    log_train_args(args, target_subjects)
    mp.train_pipeline(
        model_name=args.model,
        mode=args.mode,
        target_subjects=target_subjects,
        subject_wise_split=args.subject_wise_split,
        seed=args.seed,
        tag=args.tag,
        subject_split_strategy=split_strategy,
        time_stratify_labels=args.time_stratify_labels,
        time_stratify_tolerance=cfg.TIME_STRATIFY_TOLERANCE,
        time_stratify_window=cfg.TIME_STRATIFY_WINDOW,
        time_stratify_min_chunk=cfg.TIME_STRATIFY_MIN_CHUNK,
    )


if __name__ == "__main__":
    main()

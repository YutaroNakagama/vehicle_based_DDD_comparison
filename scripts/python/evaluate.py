"""Evaluate a trained model for driver drowsiness detection.

This script executes the evaluation pipeline for a specified model. It loads the
trained model and corresponding test data, computes evaluation metrics (e.g.,
accuracy, precision, recall, confusion matrix), and prints the results.

Supported options (summary):
  --model <name>              Model architecture (choices: src.config.MODEL_CHOICES)
  --mode <mode>               pooled | target_only | source_only | joint_train
  --tag <str>                 Optional variant tag (e.g., 'erm', 'coral')
  --seed <int>                Random seed (default: from config.DEFAULT_RANDOM_SEED)
  --target_file <path>        Path to target subject list (required for non-pooled modes)
  --fold <int>                Cross-validation fold (0 = no CV)
  --sample_size <int>         Subset number of subjects to evaluate
  --subject_wise_split        Enable subject-wise data partition to avoid leakage
  --jobid <PBS job id>        Explicit training job ID (auto-detect if omitted)

Note:
  --fold is evaluation-only (used for cross-validation evaluation).
  --sample_size and --jobid are also evaluation-specific.

Examples
--------
Evaluate an LSTM model (latest job, pooled mode):
    python evaluate.py --model Lstm --mode pooled

Evaluate a CORAL-tagged RF model on target_only mode with explicit jobid:
    python evaluate.py --model RF --mode target_only --tag coral --jobid 14004123 --target_file config/subjects/target_groups.txt
"""

import sys
import os
import argparse

# Add project root to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.cli.train_cli_helpers import (
    load_subjects_from_file,
    log_eval_args,
    add_common_arguments,
    add_eval_arguments,
    setup_logging,
)
import src.evaluation.eval_pipeline as mp


def main():
    """Parse command-line arguments and run the evaluation pipeline.

    This function parses CLI arguments to determine which model variant and
    experimental mode to evaluate, then calls `eval_pipeline`.

    Other Parameters
    ----------------
    See module docstring for full CLI options summary.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If invalid or missing arguments are provided, or if the evaluation fails.
    """
    parser = argparse.ArgumentParser(
        description="Run the evaluation pipeline for a trained DDD model."
    )

    # Add common and evaluation-specific arguments
    add_common_arguments(parser)
    add_eval_arguments(parser)

    args = parser.parse_args()
    setup_logging()

    # Normalize tag handling: treat None or "default" as no suffix
    eval_tag = args.tag
    if eval_tag is None or eval_tag.strip().lower() == "default":
        eval_tag = ""

    # Load target subjects if specified
    target_subjects = None
    if args.target_file:
        target_subjects = load_subjects_from_file(args.target_file)

    log_eval_args(args, target_subjects)

    try:
        mp.eval_pipeline(
            args.model,
            mode=args.mode,
            tag=eval_tag,
            sample_size=args.sample_size,
            seed=args.seed,
            fold=args.fold,
            subject_wise_split=args.subject_wise_split,
            jobid=args.jobid,
            target_file=args.target_file,
        )
    except Exception as e:
        import logging
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

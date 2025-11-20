"""Evaluate a trained model for driver drowsiness detection.

This script executes the evaluation pipeline for a specified model. It loads the
trained model and corresponding test data, computes evaluation metrics (e.g.,
accuracy, precision, recall, confusion matrix), and prints the results.

Supported options (summary):
  --model <name>              Model architecture (choices: src.config.MODEL_CHOICES)
  --mode <experiment mode>    pooled | target_only | source_only | joint_train
  --tag <str>                 Optional variant tag (erm / coral / etc.)
  --fold <int>                Cross-validation fold (0 = no CV)
  --sample_size <int>         Subset number of subjects to evaluate
  --subject_wise_split        Enable subject-wise data partition to avoid leakage
  --jobid <PBS job id>        Explicit training job ID (auto-detect if omitted)
  --target_file <path>        Path to target subject list (required for non-pooled modes)

Examples
--------
Evaluate an LSTM model (latest job, pooled mode):
    python evaluate.py --model Lstm --mode pooled

Evaluate a CORAL-tagged RF model on target_only mode with explicit jobid:
    python evaluate.py --model RF --mode target_only --tag coral --jobid 14004123 --target_file config/target_groups.txt
"""

import sys
import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)

# Add project root to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MODEL_CHOICES, DEFAULT_RANDOM_SEED
import src.evaluation.eval_pipeline as mp


def main():
    """Parse command-line arguments and run the evaluation pipeline.

    This function parses CLI arguments to determine which model variant and
    experimental mode to evaluate, then calls `eval_pipeline`.

    Other Parameters
    ----------------
    --model : str
        Required. Model to evaluate. Must be one of ``src.config.MODEL_CHOICES``.
    --mode : {"pooled", "target_only", "source_only", "joint_train"}
        Experimental mode controlling subject inclusion.
    --tag : str, optional
        Optional tag to distinguish model variants (e.g., ``erm``, ``coral``).
    --sample_size : int, optional
        Number of subjects to evaluate (subset evaluation).
    --seed : int, default=42
        Random seed for subset sampling.
    --fold : int, default=0
        Cross-validation fold index (0 = no fold subdirectory).
    --subject_wise_split : bool, optional
        If set, perform subject-wise data splitting to prevent leakage.
    --jobid : str, optional
        Explicit training job ID; auto-detected if omitted.
    --target_file : str, optional
        Path to target subject list (required for non-pooled modes / ranking).

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
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        required=True,
        help=f"Model to evaluate. Choices: {', '.join(MODEL_CHOICES)}"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag to distinguish model variants (e.g., 'erm', 'coral')"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of subjects to evaluate (for subset evaluation)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility in subset evaluation"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number for cross validation"
    )
    parser.add_argument(
        "--subject_wise_split",
        action="store_true",
        help="Use subject-wise data splitting to prevent subject leakage"
    )
    parser.add_argument(
        "--mode",
        choices=["pooled", "target_only", "source_only", "joint_train"],
        default="pooled",
        help="Experiment mode: pooled / target_only / source_only / joint_train"
    )
    parser.add_argument(
        "--jobid",
        type=str,
        default=None,
        help="Specify training job ID (e.g., 14004123). "
             "If not provided, the latest training job will be used automatically."
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default=None,
        help="Path to target subject list file (used in ranking experiments)."
    )

    args = parser.parse_args()

    # Normalize tag handling: treat None or "default" as no suffix
    eval_tag = args.tag
    if eval_tag is None or eval_tag.strip().lower() == "default":
        eval_tag = ""

    logging.info(f"Running evaluation for model={args.model}, tag={eval_tag or '(none)'}, mode={args.mode}")

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
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


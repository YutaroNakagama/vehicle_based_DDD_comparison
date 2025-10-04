"""Evaluate a trained model for driver drowsiness detection.

This script executes the evaluation pipeline for a specified model. It loads the
trained model and corresponding test data, computes evaluation metrics (e.g.,
accuracy, precision, recall, confusion matrix), and prints the results to the console.

Examples
--------
Run evaluation for the LSTM model:

    $ python evaluate.py --model Lstm
"""

import sys
import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)

# Add project root to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.config
import src.evaluation.eval_pipeline as mp


def main():
    """Parse command-line arguments and run the evaluation pipeline.

    This function parses CLI arguments to determine which model to evaluate,
    then calls the `eval_pipeline` function from the
    :mod:`src.evaluation.eval_pipeline` module.

    Parameters
    ----------
    None

    Other Parameters
    ----------------
    --model : str
        Required. Model to evaluate. Must be one of the options in
        ``src.config.MODEL_CHOICES``.
    --tag : str, optional
        Optional tag to distinguish model variants (e.g., ``erm``, ``coral``).
    --sample_size : int, optional
        Number of subjects to evaluate (for subset evaluation).
    --seed : int, default=42
        Random seed for reproducibility in subset evaluation.
    --fold : int, default=0
        Fold number for cross validation.
    --subject_wise_split : bool, optional
        If set, perform subject-wise data splitting to prevent subject leakage.

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
        choices=src.config.MODEL_CHOICES,
        required=True,
        help=f"Model to evaluate. Choices: {', '.join(src.config.MODEL_CHOICES)}"
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
        default=42,
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
        choices=["only_target", "only_general", "finetune"],
        default=None,
        help="Experiment mode: only_target / only_general / finetune"
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
            subject_wise_split=args.subject_wise_split
        )
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


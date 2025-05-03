"""Evaluate a trained model for driver drowsiness detection.

This script executes the evaluation pipeline for a specified model. It loads the
trained model and corresponding test data, computes evaluation metrics (e.g.,
accuracy, precision, recall, confusion matrix), and prints the results to the console.

Example:
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

    This function parses the `--model` argument from the command line to determine
    which model's evaluation pipeline to run. It then calls the `eval_pipeline`
    function from the `src.evaluation.eval_pipeline` module.

    Args:
        None

    Command-line Arguments:
        --model (str): Required. Specifies which model to evaluate. Must be one of
            the options listed in `src.config.MODEL_CHOICES`.

    Raises:
        SystemExit: Raised automatically by argparse if invalid or missing arguments are provided.
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
        "--subject_wise_split",
        action="store_true",
        help="Use subject-wise data splitting to prevent subject leakage"
    )


    args = parser.parse_args()

    logging.info(f"Running '{args.model}' model...")

    try:
        mp.eval_pipeline(
            args.model,
            tag=args.tag,
            sample_size=args.sample_size,
            seed=args.seed,
            subject_wise_split=args.subject_wise_split
        )
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


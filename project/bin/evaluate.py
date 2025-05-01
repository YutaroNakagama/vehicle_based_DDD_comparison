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

    args = parser.parse_args()

    print(f"Running '{args.model}' model...")

    mp.eval_pipeline(args.model)


if __name__ == '__main__':
    main()


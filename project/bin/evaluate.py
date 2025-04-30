"""Evaluate a trained model for driver drowsiness detection.

This script runs the evaluation pipeline for a specified model, printing metrics
or saving results as configured within the evaluation pipeline.

Example:
    python evaluate.py --model Lstm
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.config
import src.evaluation.eval_pipeline as mp

def main():
    """Parse arguments and run the evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Select a model to run.")
    parser.add_argument(
        "--model",
        choices=src.config.MODEL_CHOICES,
        required=True,
        help="Choose a model from: {', '.join(config.MODEL_CHOICES)}"
    )

    args = parser.parse_args()

    print(f"Running '{args.model}' model...")

    mp.eval_pipeline(args.model)

if __name__ == '__main__':
    main()


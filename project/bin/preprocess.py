"""Preprocess data for training or evaluation in the DDD pipeline.

This script dispatches a model-specific preprocessing routine, optionally applying
jittering-based data augmentation. It is intended to be used as a command-line interface
and invokes `main_pipeline()` from `src.data_pipeline.processing_pipeline`.

Examples:
    Run preprocessing for the LSTM model with jittering augmentation:
        $ python preprocess.py --model Lstm --jittering

    Run preprocessing for the Random Forest model without augmentation:
        $ python preprocess.py --model RF
"""

import sys
import os
import argparse

# Ensure the project root is included in the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.config
import src.data_pipeline.processing_pipeline as dp
import src.data_pipeline.processing_pipeline_mp as dp_mp


def main():
    """Parse command-line arguments and execute the preprocessing pipeline.

    This function uses argparse to parse CLI arguments specifying the model type and
    whether jittering should be applied. It then calls the `main_pipeline` function
    with the selected configuration.

    Args:
        None

    Command-line Arguments:
        --model (str): Required. Must be one of the supported model types defined in
            `src.config.DATA_PROCESS_CHOICES`. Specifies which model's preprocessing
            pipeline to run.
        --jittering (bool): Optional. If provided, applies jittering augmentation to
            the input data.

    Raises:
        SystemExit: If argument parsing fails due to invalid or missing input.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess data for a selected model in the DDD pipeline."
    )
    parser.add_argument(
        "--model",
        choices=src.config.DATA_PROCESS_CHOICES,
        required=True,
        help=f"Model to preprocess for. Choices: {', '.join(src.config.DATA_PROCESS_CHOICES)}"
    )
    parser.add_argument(
        "--jittering",
        action="store_true",
        help="Apply jittering augmentation to the input features."
    )
    parser.add_argument(
        "--multi_process",
        action="store_true",
        help="Apply multi process."
    )

    args = parser.parse_args()

    print(f"Running '{args.model}' model with jittering={'enabled' if args.jittering else 'disabled'}...")

    if args.multi_process:
        dp_mp.main_pipeline(args.model, use_jittering=args.jittering)
    else:
        dp.main_pipeline(args.model, use_jittering=args.jittering)


if __name__ == '__main__':
    main()


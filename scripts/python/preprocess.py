"""Preprocess data for training or evaluation in the DDD pipeline.

This script is the command-line entry point to the preprocessing stage. It selects a
model-specific preprocessing routine from :mod:`src.data_pipeline.processing_pipeline`
or the multiprocessing variant :mod:`src.data_pipeline.processing_pipeline_mp`.

CLI Options (summary):
  --model <name>          Required. One of src.config.DATA_PROCESS_CHOICES
  --jittering             Enable jittering augmentation
  --multi_process         Use multiprocessing pipeline implementation

Examples
--------
Run preprocessing for the LSTM model with jittering augmentation:
    python preprocess.py --model Lstm --jittering

Run preprocessing for the Random Forest model without augmentation:
    python preprocess.py --model RF
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root (one level up from scripts/)
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Ensure the project root is included in the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.config
import src.data_pipeline.processing_pipeline as dp
import src.data_pipeline.processing_pipeline_mp as dp_mp


def main():
    """Parse CLI arguments and execute the preprocessing pipeline.

    Delegates to either the single-process or multiprocessing pipeline depending
    on ``--multi_process``.

    Other Parameters
    ----------------
    --model : str
        Required. One of ``src.config.DATA_PROCESS_CHOICES``.
    --jittering : bool
        Apply jittering augmentation if passed.
    --multi_process : bool
        Use multiprocessing variant if set.

    Returns
    -------
    None
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


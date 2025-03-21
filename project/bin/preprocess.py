import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.config
import src.data_pipeline.processing_pipeline as dp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select a model to run.")
    parser.add_argument(
        "--model", 
        choices=src.config.DATA_PROCESS_CHOICES, 
        required=True, 
        help="Choose a model from: {', '.join(config.MODEL_CHOICES)}"
        )

    args = parser.parse_args()

    print(f"Running '{args.model}' model...")

    dp.main_pipeline(args.model)

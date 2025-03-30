import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.config
import src.models.model_pipeline as mp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select a model to run.")
    parser.add_argument(
        "--model", 
        choices=src.config.MODEL_CHOICES, 
        required=True, 
        help="Choose a model from: {', '.join(config.MODEL_CHOICES)}"
    )
    parser.add_argument(
        "--domain_mixup",
        action="store_true",
        help="Enable domain mixup augmentation during training."
    )

    args = parser.parse_args()

    print(f"Running '{args.model}' model with domain_mixup={'enabled' if args.domain_mixup else 'disabled'}...")

    mp.train_pipeline(args.model, domain_generalize=args.domain_mixup)

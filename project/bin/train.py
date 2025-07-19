"""Train a machine learning model for driver drowsiness detection.

This script executes the training pipeline for a specified model architecture.
Users can optionally enable data augmentation or domain generalization techniques
such as Domain Mixup, CORAL (Correlation Alignment), and VAE-based feature augmentation.

Examples:
    Train an LSTM model with all augmentation methods enabled:
        $ python train.py --model Lstm --domain_mixup --coral --vae

    Train a Random Forest model without any augmentation:
        $ python train.py --model RF
"""

import sys
import os
import argparse
import logging

# Add project root to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.config
import src.models.model_pipeline as mp

def log_train_args(args):
    logging.info(
        "[RUN] model=%s | domain_mixup=%s | coral=%s | vae=%s | sample_size=%s | seed=%s | fold=%s | tag=%s | subject_wise_split=%s | feature_selection=%s | data_leak=%s",
        args.model,
        "enabled" if args.domain_mixup else "disabled",
        "enabled" if args.coral else "disabled",
        "enabled" if args.vae else "disabled",
        str(args.sample_size) if args.sample_size is not None else "None",
        str(args.seed) if args.seed is not None else "None",
        str(args.fold) if args.fold is not None else "None",
        args.tag if args.tag else "None",
        "enabled" if args.subject_wise_split else "disabled",
        args.feature_selection,
        "enabled" if args.data_leak else "disabled",
        )


def main():
    """Parse command-line arguments and invoke the training pipeline.

    This function handles the parsing of command-line arguments for model selection
    and optional data augmentation strategies. Based on user input, it triggers
    the appropriate training routine.

    Command-line Arguments:
        --model (str): Required. Specifies the model architecture to train.
            Must be one of the model names defined in `src.config.MODEL_CHOICES`.
        --domain_mixup (bool): Optional. If set, applies Domain Mixup-based
            feature interpolation.
        --coral (bool): Optional. If set, applies CORAL-based domain alignment.
        --vae (bool): Optional. If set, applies VAE-based feature augmentation.

    Raises:
        SystemExit: Raised by argparse if invalid or missing arguments are given.
    """
    parser = argparse.ArgumentParser(
        description="Train a model for driver drowsiness detection with optional augmentation."
    )
    parser.add_argument(
        "--model",
        choices=src.config.MODEL_CHOICES,
        required=True,
        help=f"Model architecture to train. Choices: {', '.join(src.config.MODEL_CHOICES)}"
    )
    parser.add_argument(
        "--domain_mixup",
        action="store_true",
        help="Apply Domain Mixup augmentation during training."
    )
    parser.add_argument(
        "--coral",
        action="store_true",
        help="Apply CORAL domain alignment during training."
    )
    parser.add_argument(
        "--vae",
        action="store_true",
        help="Apply VAE-based data augmentation during training."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of subjects to randomly sample from all available subjects (for small-scale experiments)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subject sampling."
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=None,
        help="Number of folds for cross-validation (if specified, runs fold=1,...,n_folds sequentially)."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold number for single-fold training. If n_folds is set, this is ignored."
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag suffix for model saving (e.g., 'erm', 'coral')."
    )
    parser.add_argument(
        "--subject_wise_split",
        action="store_true",
        help="Use subject-wise data splitting to prevent data leakage across subjects."
    )
    parser.add_argument(
        "--feature_selection",
        choices=["rf", "mi", "anova"],
        default="rf",
        help="Feature selection method: 'rf' (RandomForest importance), 'mi' (Mutual Information), 'anova' (ANOVA F-test). Default: rf."
    )
    parser.add_argument(
        "--data_leak",
        action="store_true",
        help="If set, intentionally allow data leakage for feature selection (for ablation/demonstration).",
    )
    parser.add_argument(
        "--subject_split_strategy",
        choices=["random", "leave-one-out", "custom", "isolate_target_subjects", "finetune_target_subjects", "single_subject_data_split","subject_time_split"],
        default="random",
        help="Strategy for splitting subjects into train, validation, and test sets."
    )
    parser.add_argument(
        "--target_subjects",
        nargs='+',
        default=[],
        help="List of subject IDs for the training set (used with 'custom' or target-based strategies)."
    )
    parser.add_argument(
        "--train_subjects",
        nargs='+',
        default=[],
        help="List of subject IDs for the training set (used with 'custom' strategy)."
    )
    parser.add_argument(
        "--val_subjects",
        nargs='+',
        default=[],
        help="List of subject IDs for the validation set (used with 'custom' strategy)."
    )
    parser.add_argument(
        "--test_subjects",
        nargs='+',
        default=[],
        help="List of subject IDs for the test set (used with 'custom' strategy)."
    )
    parser.add_argument(
        "--general_subjects",
        nargs='+',
        default=[],
        help="List of subject IDs for general training data (used with 'finetune_target_subjects' strategy)."
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    tag_msg = f", tag={args.tag}" if args.tag else ""

    # Centralize the call to the training pipeline
    pipeline_args = {
        "model_name": args.model,
        "use_domain_mixup": args.domain_mixup,
        "use_coral": args.coral,
        "use_vae": args.vae,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "tag": args.tag,
        "subject_wise_split": args.subject_wise_split,
        "feature_selection_method": args.feature_selection,
        "data_leak": False,
        "subject_split_strategy": args.subject_split_strategy,
        "target_subjects": args.target_subjects,
        "train_subjects": args.train_subjects,
        "val_subjects": args.val_subjects,
        "test_subjects": args.test_subjects,
        "general_subjects": args.general_subjects
    }

    if args.n_folds is not None:
        for fold in range(1, args.n_folds + 1):
            logging.info(f"==== Running fold {fold} / {args.n_folds} ====")
            pipeline_args["fold"] = fold
            pipeline_args["n_folds"] = args.n_folds
            log_train_args(args)
            mp.train_pipeline(**pipeline_args)
    else:
        pipeline_args["fold"] = args.fold
        pipeline_args["n_folds"] = args.n_folds
        log_train_args(args)
        mp.train_pipeline(**pipeline_args)

if __name__ == '__main__':
    main()


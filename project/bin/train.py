"""Train a machine learning model for driver drowsiness detection.

This script executes the training pipeline for a specified model architecture.
Users can optionally enable data augmentation or domain generalization techniques
such as Domain Mixup, CORAL (Correlation Alignment), and VAE-based feature augmentation.

Examples
--------
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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import src.config
import src.models.model_pipeline as mp

def log_train_args(args):
    """Log training arguments in a consistent format.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing training settings.
    """
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

    This function parses CLI arguments for model selection, augmentation
    options, cross-validation, and subject split strategies. It then calls
    the centralized training pipeline.

    Parameters
    ----------
    None

    Other Parameters
    ----------------
    --model : str
        Required. Model architecture to train. Must be one of
        ``src.config.MODEL_CHOICES``.
    --domain_mixup : bool, optional
        Apply Domain Mixup augmentation during training.
    --coral : bool, optional
        Apply CORAL domain alignment during training.
    --vae : bool, optional
        Apply VAE-based feature augmentation during training.
    --sample_size : int, optional
        Number of subjects to randomly sample for small-scale experiments.
    --seed : int, optional
        Random seed for reproducibility. Default is 42.
    --n_folds : int, optional
        Number of folds for cross-validation. If set, runs sequentially.
    --fold : int, optional
        Fold number for single-fold training. Ignored if ``n_folds`` is set.
    --tag : str, optional
        Optional tag suffix for model saving.
    --subject_wise_split : bool, optional
        Prevent data leakage across subjects by using subject-wise split.
    --feature_selection : {"rf", "mi", "anova"}, default="rf"
        Feature selection method: RandomForest importance, Mutual Information, or ANOVA F-test.
    --data_leak : bool, optional
        Intentionally allow data leakage for feature selection (for ablation).
    --subject_split_strategy : {"random", "leave-one-out", "custom", \
"isolate_target_subjects", "finetune_target_subjects", "single_subject_data_split", "subject_time_split"}, default="random"
        Strategy for splitting subjects into train/val/test.
    --target_subjects, --train_subjects, --val_subjects, --test_subjects, --general_subjects : list of str, optional
        Lists of subject IDs used with specific split strategies.
    --finetune_setting : str, optional
        Path to pretrained settings (pickle file).
    --save_pretrain : str, optional
        Path to save pretrain settings (feature list, scaler, model params).
    --eval_only_pretrained : bool, optional
        Skip fine-tuning and directly evaluate a pretrained model.
    --time_stratify_labels : bool, optional
        Adjust time-ordered split boundaries so each split matches the global
        positive/negative ratio.
    --time_stratify_tolerance : float, default=0.02
        Allowed deviation in class ratio per split.
    --time_stratify_window : float, default=0.10
        Window size (fraction of N) to adjust split boundaries.
    --time_stratify_min_chunk : int, default=100
        Minimum rows per split to avoid degenerate segments.
    --balance_labels : bool, optional
        Rebalance each split to 50/50 class distribution.
    --balance_method : {"undersample", "oversample"}, default="undersample"
        Method for label balancing.
    --mode : {"only_target", "only_general", "eval_only", "finetune"}, optional
        Experiment mode controlling subject splits.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If invalid or missing arguments are provided.
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

    parser.add_argument(
        "--finetune_setting",
        type=str,
        default=None,
        help="Path to pickle file with pretrained feature/param settings for finetune."
    )

    parser.add_argument(
        "--eval_only_pretrained",
        action="store_true",
        help="Skip fine-tuning on target subjects and directly evaluate using a model pretrained on non-target subjects."
    )

    parser.add_argument(
        "--save_pretrain", type=str, default=None,
        help="Path to save pretrain settings (feature list, scaler, model params)."
    )
    parser.add_argument(
        "--time_stratify_labels",
        action="store_true",
        help="Adjust time-ordered split boundaries so that each split's pos/neg ratio matches the global ratio (no resampling)."
    )
    parser.add_argument(
        "--time_stratify_tolerance",
        type=float,
        default=0.02,
        help="Allowed absolute deviation of positive ratio per split from the global ratio (e.g., 0.02 = ±2%)."
    )
    parser.add_argument(
        "--time_stratify_window",
        type=float,
        default=0.10,
        help="Search window size around nominal cut positions as a fraction of N (e.g., 0.10 = ±10%)."
    )
    parser.add_argument(
        "--time_stratify_min_chunk",
        type=int,
        default=100,
        help="Minimum number of rows per split to avoid degenerate segments."
    )
    parser.add_argument(
        "--balance_labels", 
        action="store_true",
        help="Rebalance each split (train/val/test) to 50/50 positive/negative."
    )
    parser.add_argument(
        "--balance_method", 
        choices=["undersample", "oversample"],
        default="undersample", 
        help="How to balance labels within each split."
    )
    parser.add_argument(
        "--mode",
        choices=["only_target", "only_general", "eval_only", "finetune", "train_only"],
        default=None,
        help="Exp mode: only_target / only_general / eval_only / finetune / train_only"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="If set, skip training and only evaluate using an already saved model."
    )
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="If set, save model and scaler but skip evaluation (no metrics/plots are generated)."
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    tag_msg = f", tag={args.tag}" if args.tag else ""

    if args.mode in ("only_general", "eval_only"):
        args.subject_split_strategy = "finetune_target_subjects"
        # Do not force eval_only when train_only mode is set
        if not args.train_only and args.mode != "train_only":
            args.eval_only_pretrained = True
        if not args.target_subjects:
            raise SystemExit("[ERROR] --mode=only_general(eval_only) では --target_subjects が必須です。")
    
    elif args.mode == "finetune":
        args.subject_split_strategy = "finetune_target_subjects"
        if not args.target_subjects:
            raise SystemExit("[ERROR] --mode=finetune では --target_subjects が必須です。")
        if (args.finetune_setting is None) and (args.save_pretrain is None):
            raise SystemExit("[ERROR] --mode=finetune では --finetune_setting か --save_pretrain のどちらかが必要です。")
    
    elif args.mode == "only_target":
        args.subject_split_strategy = "subject_time_split"
        if not args.target_subjects:
            raise SystemExit("[ERROR] --mode=only_target では --target_subjects が必須です。")

    elif args.mode == "train_only":
        if args.target_subjects:
            args.subject_split_strategy = "subject_time_split"
        else:
            args.subject_split_strategy = "finetune_target_subjects"

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
        "data_leak": args.data_leak,
        "subject_split_strategy": args.subject_split_strategy,
        "target_subjects": args.target_subjects,
        "train_subjects": args.train_subjects,
        "val_subjects": args.val_subjects,
        "test_subjects": args.test_subjects,
        "general_subjects": args.general_subjects,
        "finetune_setting": args.finetune_setting,
        "save_pretrain": args.save_pretrain,   
        "eval_only_pretrained": args.eval_only_pretrained,
        "balance_labels": args.balance_labels,          # if you kept previous balancing option
        "balance_method": args.balance_method,
        "time_stratify_labels": args.time_stratify_labels,
        "time_stratify_tolerance": args.time_stratify_tolerance,
        "time_stratify_window": args.time_stratify_window,
        "time_stratify_min_chunk": args.time_stratify_min_chunk,
        "eval_only": args.eval_only,   
        "train_only": args.train_only or args.mode == "train_only",
        "mode": args.mode,   
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


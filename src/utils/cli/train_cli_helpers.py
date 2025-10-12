"""Helper functions for `scripts/python/train.py` CLI.

This module encapsulates reusable logic for:
- Loading target subject IDs from a text file
- Logging parsed arguments
- Mapping mode → split strategy

Keeping this logic separate makes `train.py` minimal and easier to reuse
in HPC pipelines or Jupyter environments.
"""

import os
import logging


def load_subjects_from_file(path: str):
    """Load subject IDs from a text file.

    Supports space, comma, or newline separation.
    Raises clear errors if the file is missing or empty.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Target subject file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        subjects = [
            s for s in text.replace(",", " ").replace("\t", " ").split()
            if s.strip()
        ]

    if not subjects:
        raise ValueError(f"[ERROR] No valid subject IDs found in file: {path}")

    logging.info(
        f"[INFO] Loaded {len(subjects)} subjects from {path}: "
        f"{subjects[:5]}{'...' if len(subjects) > 5 else ''}"
    )
    return subjects


def log_train_args(args, target_subjects):
    """Log all parsed training arguments."""
    logging.info(
        "[RUN] model=%s | mode=%s | seed=%s | tag=%s | target_file=%s | subjects=%s",
        args.model,
        args.mode,
        args.seed,
        args.tag,
        args.target_file if args.target_file else "None",
        " ".join(target_subjects) if target_subjects else "None",
    )


def map_mode_to_strategy(mode: str) -> str:
    """Map `--mode` argument to internal split strategy."""
    mapping = {
        "pooled": "random",
        "target_only": "subject_time_split",
        "source_only": "subject_wise_split",
        "joint_train": "finetune_target_subjects",
    }
    if mode not in mapping:
        raise ValueError(f"Unknown mode: {mode}")
    return mapping[mode]

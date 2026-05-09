#!/usr/bin/env python3
"""
Rebuild submitted_exp2_v2.txt to contain ONLY successfully completed configs.

Why: the auto-resubmit daemon skips any config already listed in
submitted_exp2_v2.txt, but currently that list contains TIMEOUT/CANCELLED
configs that produced no output. Resetting the list to "completed only"
allows the daemon to resubmit failed configs.

Source of truth = presence of BOTH train_results JSON in
results/outputs/training/RF/manual_2026*/ AND matching eval_results JSON
in results/outputs/evaluation/RF/manual_2026*/.

Filename → config key mapping:
  baseline:    train_results_RF_{mode}_baseline_domain_knn_{dist}_{dom}_{mode}_split2_s{seed}.json
  smote_plain: train_results_RF_{mode}_smote_plain_knn_{dist}_{dom}_{mode}_split2_ratio{R}_s{seed}.json
  smote:       train_results_RF_{mode}_imbalv3_knn_{dist}_{dom}_{mode}_split2_subjectwise_ratio{R}_s{seed}.json
  undersample: train_results_RF_{mode}_undersample_rus_knn_{dist}_{dom}_{mode}_split2_ratio{R}_s{seed}.json

Output key (matches submit_exp2_multiqueue.sh KEY format):
  baseline:                {condition}:{distance}:{domain}:{mode}:{seed}
  smote/smote_plain/under: {condition}:{distance}:{domain}:{mode}:{ratio}:{seed}
"""

import re
import sys
import glob
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/s2240011/git/ddd/vehicle_based_DDD_comparison")
TRAIN_DIR = ROOT / "results/outputs/training/RF"
EVAL_DIR = ROOT / "results/outputs/evaluation/RF"
SUBMITTED_FILE = ROOT / "scripts/hpc/logs/domain/submitted_exp2_v2.txt"

SEEDS = [0, 1, 3, 7, 13, 42, 123, 256, 512, 999, 1337, 2024]
RATIOS = ["0.1", "0.5"]
DISTANCES = ["mmd", "dtw", "wasserstein"]
DOMAINS = ["out_domain", "in_domain"]
MODES = ["source_only", "target_only", "mixed"]

# Maps result-file condition string → submit-script condition string
COND_MAP = {
    "baseline_domain": "baseline",
    "smote_plain":     "smote_plain",
    "imbalv3":         "smote",
    "undersample_rus": "undersample",
}

# Regex per condition. Anchors after RF_{mode}_
PATTERNS = [
    # baseline (no ratio): ..._baseline_domain_knn_{dist}_{dom}_{mode}_split2_s{seed}
    (
        "baseline",
        re.compile(
            r"^train_results_RF_(?P<mode>source_only|target_only|mixed)_"
            r"baseline_domain_knn_"
            r"(?P<dist>mmd|dtw|wasserstein)_"
            r"(?P<dom>out_domain|in_domain)_"
            r"(?:source_only|target_only|mixed)_split2_"
            r"s(?P<seed>-?\d+)\.json$"
        ),
    ),
    # smote_plain (ratio): ..._smote_plain_knn_{dist}_{dom}_{mode}_split2_ratio{R}_s{seed}
    (
        "smote_plain",
        re.compile(
            r"^train_results_RF_(?P<mode>source_only|target_only|mixed)_"
            r"smote_plain_knn_"
            r"(?P<dist>mmd|dtw|wasserstein)_"
            r"(?P<dom>out_domain|in_domain)_"
            r"(?:source_only|target_only|mixed)_split2_"
            r"ratio(?P<ratio>[\d.]+)_"
            r"s(?P<seed>-?\d+)\.json$"
        ),
    ),
    # smote = imbalv3: ..._imbalv3_knn_{dist}_{dom}_{mode}_split2_subjectwise_ratio{R}_s{seed}
    (
        "smote",
        re.compile(
            r"^train_results_RF_(?P<mode>source_only|target_only|mixed)_"
            r"imbalv3_knn_"
            r"(?P<dist>mmd|dtw|wasserstein)_"
            r"(?P<dom>out_domain|in_domain)_"
            r"(?:source_only|target_only|mixed)_split2_subjectwise_"
            r"ratio(?P<ratio>[\d.]+)_"
            r"s(?P<seed>-?\d+)\.json$"
        ),
    ),
    # undersample = undersample_rus: ..._undersample_rus_knn_{dist}_{dom}_{mode}_split2_ratio{R}_s{seed}
    (
        "undersample",
        re.compile(
            r"^train_results_RF_(?P<mode>source_only|target_only|mixed)_"
            r"undersample_rus_knn_"
            r"(?P<dist>mmd|dtw|wasserstein)_"
            r"(?P<dom>out_domain|in_domain)_"
            r"(?:source_only|target_only|mixed)_split2_"
            r"ratio(?P<ratio>[\d.]+)_"
            r"s(?P<seed>-?\d+)\.json$"
        ),
    ),
]


def parse_train_filename(name: str):
    """Return (key, eval_filename) or (None, None) if not exp2 RF result."""
    for cond, pat in PATTERNS:
        m = pat.match(name)
        if not m:
            continue
        d = m.groupdict()
        if cond == "baseline":
            key = f"baseline:{d['dist']}:{d['dom']}:{d['mode']}:{d['seed']}"
        else:
            ratio = d["ratio"]
            # Ratio might be "0.5" or "0.50" — submit script uses "0.5" / "0.1"
            try:
                rf = float(ratio)
                ratio_norm = ("%g" % rf)
            except ValueError:
                ratio_norm = ratio
            key = f"{cond}:{d['dist']}:{d['dom']}:{d['mode']}:{ratio_norm}:{d['seed']}"
        eval_name = "eval_results_" + name[len("train_results_"):]
        return key, eval_name
    return None, None


def expected_full_set():
    keys = set()
    for dist in DISTANCES:
        for dom in DOMAINS:
            for mode in MODES:
                for seed in SEEDS:
                    keys.add(f"baseline:{dist}:{dom}:{mode}:{seed}")
                    for ratio in RATIOS:
                        for cond in ("smote_plain", "smote", "undersample"):
                            keys.add(f"{cond}:{dist}:{dom}:{mode}:{ratio}:{seed}")
    return keys


def main():
    dry_run = "--dry-run" in sys.argv

    # Build set of completed configs (require both train + eval JSON)
    completed = set()
    train_keys = {}  # key -> train file path
    train_files = glob.glob(str(TRAIN_DIR / "manual_2026*/**/train_results_*.json"),
                            recursive=True)
    eval_files_set = set()
    for ef in glob.glob(str(EVAL_DIR / "manual_2026*/**/eval_results_*.json"),
                       recursive=True):
        eval_files_set.add(Path(ef).name)

    unparsed = 0
    for tf in train_files:
        name = Path(tf).name
        key, eval_name = parse_train_filename(name)
        if key is None:
            unparsed += 1
            continue
        if eval_name in eval_files_set:
            completed.add(key)
            train_keys.setdefault(key, tf)

    full = expected_full_set()
    extra = completed - full   # keys present but not in expected (shouldn't happen)
    missing = full - completed

    print(f"Expected total configs: {len(full)}")
    print(f"Train result files scanned: {len(train_files)} (unparsed: {unparsed})")
    print(f"Completed (train+eval present, matching expected set): {len(completed & full)}")
    print(f"Extra (parsed but not in expected set): {len(extra)}")
    print(f"Missing (need to (re)submit): {len(missing)}")

    if extra:
        print("\nSample extras (first 5):")
        for k in sorted(extra)[:5]:
            print(f"  {k}")

    print("\nMissing breakdown by condition:")
    by_cond = {}
    for k in missing:
        c = k.split(":", 1)[0]
        by_cond[c] = by_cond.get(c, 0) + 1
    for c, n in sorted(by_cond.items()):
        print(f"  {c}: {n}")

    print("\nMissing breakdown by mode:")
    by_mode = {}
    for k in missing:
        parts = k.split(":")
        # baseline: cond:dist:dom:mode:seed (5 parts) → mode at idx 3
        # else:     cond:dist:dom:mode:ratio:seed (6 parts) → mode at idx 3
        mode = parts[3]
        by_mode[mode] = by_mode.get(mode, 0) + 1
    for m, n in sorted(by_mode.items()):
        print(f"  {m}: {n}")

    # Backup current submitted file
    if dry_run:
        print("\n[DRY-RUN] No changes written.")
        return

    if SUBMITTED_FILE.exists():
        backup = SUBMITTED_FILE.with_suffix(
            f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        shutil.copy2(SUBMITTED_FILE, backup)
        print(f"\nBacked up old submitted file → {backup.name}")

    # Write only completed configs (intersected with expected set to drop bogus extras)
    keep = sorted(completed & full)
    SUBMITTED_FILE.write_text("\n".join(keep) + ("\n" if keep else ""))
    print(f"Wrote {len(keep)} completed-only entries to {SUBMITTED_FILE.name}")
    print(f"Daemon will now (re)submit the {len(missing)} missing configs.")


if __name__ == "__main__":
    main()

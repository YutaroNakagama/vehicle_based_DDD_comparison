#!/usr/bin/env python3
"""Plot aggregated LSTM ROC curves (mean +/- std) for mixed-in/out and pooled."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from src.config import LSTM_SEGMENT_TIMESTEPS
from src.models.architectures.lstm import create_segments
from src.utils.io.loaders import load_model_and_scaler, load_subjects_and_data
from src.utils.io.preprocessing import prepare_evaluation_features
from src.utils.io.split_helpers import split_data
from src.utils.io.target_resolution import resolve_target_subjects_from_tag


def _fallback_target_subjects_from_file() -> list[str]:
    """Load default target subjects when rank-based mapping is unavailable."""
    p = Path("config/subjects/target_groups.txt")
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8")
    # File is whitespace-delimited; keep order and remove duplicates.
    ids = re.findall(r"S\d{4}_\d", txt)
    return list(dict.fromkeys(ids))


def _collect_latest_eval_files_by_seed() -> dict[str, list[tuple[int, Path]]]:
    root = Path("results/outputs/evaluation/Lstm")
    if not root.exists():
        raise FileNotFoundError(f"Missing evaluation root: {root}")

    prefixes = {
        "mixed_in": "eval_results_Lstm_mixed_imbalv3_knn_wasserstein_in_domain_mixed_split2_subjectwise_ratio0.5_s",
        "mixed_out": "eval_results_Lstm_mixed_imbalv3_knn_wasserstein_out_domain_mixed_split2_subjectwise_ratio0.5_s",
        "pooled": "eval_results_Lstm_pooled_iv25smote_Lstm_pooled_swsmote_s",
    }

    out: dict[str, list[tuple[int, Path]]] = {}
    for case, pfx in prefixes.items():
        latest_by_seed: dict[int, Path] = {}
        for p in root.rglob("eval_results_Lstm_*.csv"):
            n = p.name
            if not (n.startswith(pfx) and n.endswith(".csv")):
                continue
            seed_str = n[len(pfx) : -4]
            if not seed_str.isdigit():
                continue
            seed = int(seed_str)
            prev = latest_by_seed.get(seed)
            if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
                latest_by_seed[seed] = p

        out[case] = sorted(latest_by_seed.items(), key=lambda x: x[0])
    return out


def build_test_split(subjects, model_name: str, mode: str, tag: str, seed: int):
    target_subjects = resolve_target_subjects_from_tag(tag=tag, mode=mode, cli_target_subjects=None)
    if mode == "mixed" and len(target_subjects) == 0:
        target_subjects = _fallback_target_subjects_from_file()

    if mode in ["source_only", "target_only", "mixed"] and len(target_subjects) > 0:
        _, _, x_test, _, _, y_test = split_data(
            subject_split_strategy="subject_time_split",
            subject_list=subjects,
            target_subjects=target_subjects,
            model_name=model_name,
            seed=seed,
            time_stratify_labels=False,
            time_stratify_tolerance=0.02,
            time_stratify_window=0.10,
            time_stratify_min_chunk=100,
        )
    else:
        # pooled evaluation path in current pipeline
        _, _, x_test, _, _, y_test = split_data(
            subject_split_strategy="random",
            subject_list=subjects,
            target_subjects=[],
            model_name=model_name,
            seed=seed,
            time_stratify_labels=False,
            time_stratify_tolerance=0.02,
            time_stratify_window=0.10,
            time_stratify_min_chunk=100,
        )

    return x_test, y_test


def compute_curve(
    subjects,
    model_name: str,
    mode: str,
    tag: str,
    seed: int,
    fold: int = 0,
    jobid: str | None = None,
):
    x_test, y_test = build_test_split(subjects, model_name, mode, tag, seed)

    # Prefer explicit jobid when provided so we bind to the exact artifact that
    # produced the saved eval CSV/JSON row for this tag.
    if jobid is None:
        raise ValueError(f"jobid is required for deterministic loading: mode={mode}, tag={tag}")
    clf, scaler, features = load_model_and_scaler(model_name, mode, tag, fold, jobid)
    if clf is None:
        raise RuntimeError(f"Model load failed: mode={mode}, tag={tag}")

    x_test_prepared = prepare_evaluation_features(x_test, scaler, features)
    x_3d, y_seg = create_segments(
        np.asarray(x_test_prepared, dtype=np.float32),
        np.asarray(y_test),
        LSTM_SEGMENT_TIMESTEPS,
    )
    y_score = clf.predict(x_3d, verbose=0).flatten()

    fpr, tpr, _ = roc_curve(y_seg, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc, int(len(y_seg)), int(np.sum(y_seg))


def _interpolate_tpr(fpr: np.ndarray, tpr: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # Ensure endpoints are present for stable interpolation at [0, 1].
    fpr2 = np.concatenate(([0.0], fpr, [1.0]))
    tpr2 = np.concatenate(([0.0], tpr, [1.0]))
    return np.interp(grid, fpr2, tpr2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42, help="Fallback seed for legacy/single-seed usage")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--outdir", default="results/analysis/exp3_verification")
    ap.add_argument("--max-seeds", type=int, default=0, help="Optional cap per condition (0=all)")
    args = ap.parse_args()

    model = "Lstm"
    subjects, model_name, _ = load_subjects_and_data(model, args.fold, None, args.seed, False)

    case_modes = {
        "mixed_in": "mixed",
        "mixed_out": "mixed",
        "pooled": "pooled",
    }

    eval_files_by_seed = _collect_latest_eval_files_by_seed()

    fpr_grid = np.linspace(0.0, 1.0, 501)
    agg: dict[str, dict] = {}

    for name in ["mixed_in", "mixed_out", "pooled"]:
        mode = case_modes[name]
        entries = eval_files_by_seed[name]
        if args.max_seeds > 0:
            entries = entries[: args.max_seeds]

        if not entries:
            raise RuntimeError(f"No eval files found for case={name}")

        tprs = []
        aucs = []
        n_samples = []
        n_pos = []
        used_seeds = []

        for seed, eval_csv in entries:
            tag = eval_csv.name[len("eval_results_Lstm_") : -4]
            jobid = eval_csv.parent.parent.name
            try:
                fpr, tpr, roc_auc, n_s, n_p = compute_curve(
                    subjects,
                    model_name,
                    mode,
                    tag,
                    seed,
                    args.fold,
                    jobid=jobid,
                )
            except Exception as e:
                print(f"[WARN] skip case={name} seed={seed} jobid={jobid}: {e}")
                continue

            tprs.append(_interpolate_tpr(fpr, tpr, fpr_grid))
            aucs.append(float(roc_auc))
            n_samples.append(int(n_s))
            n_pos.append(int(n_p))
            used_seeds.append(int(seed))

        if not tprs:
            raise RuntimeError(f"All seeds failed for case={name}")

        tprs_arr = np.asarray(tprs)
        agg[name] = {
            "fpr": fpr_grid,
            "tpr_mean": tprs_arr.mean(axis=0),
            "tpr_std": tprs_arr.std(axis=0),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "n_seeds": len(used_seeds),
            "seeds": used_seeds,
            "n_samples_mean": float(np.mean(n_samples)),
            "n_pos_mean": float(np.mean(n_pos)),
        }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / "lstm_mixed_in_out_pooled_roc_mean_std.png"
    out_csv = outdir / "lstm_mixed_in_out_pooled_roc_mean_std_points.csv"
    out_json = outdir / "lstm_mixed_in_out_pooled_roc_mean_std_summary.json"

    plt.figure(figsize=(8, 6))
    color_map = {"mixed_in": "#1f77b4", "mixed_out": "#ff7f0e", "pooled": "#2ca02c"}
    for name in ["mixed_in", "mixed_out", "pooled"]:
        d = agg[name]
        fpr = d["fpr"]
        tpr_m = d["tpr_mean"]
        tpr_s = d["tpr_std"]
        lo = np.clip(tpr_m - tpr_s, 0.0, 1.0)
        hi = np.clip(tpr_m + tpr_s, 0.0, 1.0)
        plt.fill_between(fpr, lo, hi, color=color_map[name], alpha=0.15)
        plt.plot(
            fpr,
            tpr_m,
            color=color_map[name],
            lw=2,
            label=(
                f"{name} AUC={d['auc_mean']:.3f}±{d['auc_std']:.3f} "
                f"(seeds={d['n_seeds']})"
            ),
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LSTM ROC Curves (mean +/- std across seeds)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("case,fpr,tpr_mean,tpr_std,auc_mean,auc_std,n_seeds\n")
        for name in ["mixed_in", "mixed_out", "pooled"]:
            d = agg[name]
            for x, y, s in zip(d["fpr"], d["tpr_mean"], d["tpr_std"]):
                f.write(f"{name},{x},{y},{s},{d['auc_mean']},{d['auc_std']},{d['n_seeds']}\n")

    summary = {
        k: {
            "auc_mean": v["auc_mean"],
            "auc_std": v["auc_std"],
            "n_seeds": v["n_seeds"],
            "seeds": v["seeds"],
            "n_samples_mean": v["n_samples_mean"],
            "n_pos_mean": v["n_pos_mean"],
        }
        for k, v in agg.items()
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"PNG: {out_png.as_posix()}")
    print(f"CSV: {out_csv.as_posix()}")
    print(f"JSON: {out_json.as_posix()}")
    for name in ["mixed_in", "mixed_out", "pooled"]:
        d = agg[name]
        print(
            f"{name}: AUC_mean={d['auc_mean']:.6f}, AUC_std={d['auc_std']:.6f}, "
            f"seeds={d['n_seeds']}, seed_list={d['seeds']}"
        )


if __name__ == "__main__":
    main()

# src/analysis/summary_groups.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List
import json
import numpy as np
import pandas as pd

METRICS = ["accuracy", "f1", "auc", "precision", "recall"]

def _read_test_metrics(csv_path: Path, split: Optional[str]) -> Optional[Dict[str, float]]:
    """Read evaluation metrics from a metrics CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the metrics CSV file (e.g., ``metrics_RF.csv``).
    split : str, optional
        Dataset split to filter (e.g., ``"test"``). If provided, only rows
        where ``split`` matches are used. If no ``split`` column exists,
        the first row is used.

    Returns
    -------
    dict of {str: float} or None
        Dictionary of metric values for the selected row.
        Keys are from ``METRICS`` = ["accuracy", "f1", "auc", "precision", "recall"].
        Returns ``None`` if the file does not exist, cannot be read, or no rows match.
    """
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "split" in df.columns and split:
        df = df[df["split"] == split].copy()
        if df.empty:
            return None
    row = df.iloc[0]
    return {m: float(row.get(m, np.nan)) for m in METRICS}

def _fmt(v):
    return f"{v:.6f}" if isinstance(v, float) else str(v)

def run_summarize_only10_vs_finetune(
    names_file: Path,
    model_dir: Path,
    out_prefix: str,
    model: str = "RF",
    split: Optional[str] = "test",
    make_radar: bool = False,
    only10_pattern: str = "metrics_{model}_only10_{group}.csv",
    finetune_pattern: str = "metrics_{model}_finetune_{group}_finetune.csv",
) -> Dict[str, Path]:
    """Summarize only10 vs finetune results across groups.

    This function collects per-group metrics from the specified model directory,
    compares ``only10`` and ``finetune`` schemes, and generates multiple outputs:
    long-form CSV, wide-form CSV with deltas, improvement summary, and Markdown.
    Optionally, radar plots are generated.

    Parameters
    ----------
    names_file : Path
        Path to a text file listing group names, one per line.
    model_dir : Path
        Directory containing metrics CSV files.
    out_prefix : str
        Prefix for output filenames.
    model : str, default="RF"
        Model identifier used in metrics filenames.
    split : str, optional, default="test"
        Dataset split to filter (e.g., ``"test"``). If ``None``, use all rows.
    make_radar : bool, default=False
        Whether to generate radar plots for all groups.
    only10_pattern : str, default="metrics_{model}_only10_{group}.csv"
        Filename pattern for only10 metrics.
    finetune_pattern : str, default="metrics_{model}_finetune_{group}_finetune.csv"
        Filename pattern for finetune metrics.

    Returns
    -------
    dict of {str: Path}
        Dictionary containing paths to generated output files:
        - ``"long"`` : long-form CSV
        - ``"wide"`` : wide-form CSV with deltas
        - ``"markdown"`` : Markdown summary
        - ``"improvements"`` : improvement summary CSV

    Raises
    ------
    FileNotFoundError
        If the group names file is missing.
    RuntimeError
        If no metrics are found in the given directory.
    """
    names_file = Path(names_file)
    model_dir  = Path(model_dir)
    out_long   = model_dir / f"{out_prefix}_long.csv"
    out_wide   = model_dir / f"{out_prefix}_wide.csv"
    out_md     = model_dir / f"{out_prefix}.md"
    imp_csv    = model_dir / f"{out_prefix}_improvement_summary.csv"

    if not names_file.exists():
        raise FileNotFoundError(f"Group names file not found: {names_file}")

    names: List[str] = [
        ln.strip() for ln in names_file.read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("group_names")
    ]

    rows: List[Dict] = []

    base_path = model_dir / f"metrics_{model}.csv"
    base_m = _read_test_metrics(base_path, split)
    if base_m is not None:
        rows.append({"group": "baseline", "scheme": "baseline", **base_m, "source": base_path.name})

    for name in names:
        only10_path   = model_dir / only10_pattern.format(model=model, group=name)
        finetune_path = model_dir / finetune_pattern.format(model=model, group=name)

        m_only = _read_test_metrics(only10_path, split)
        m_fine = _read_test_metrics(finetune_path, split)

        if m_only is not None:
            rows.append({"group": name, "scheme": "only10", **m_only, "source": only10_path.name})
        else:
            print(f"[WARN] missing only10: {only10_path.name}")

        if m_fine is not None:
            rows.append({"group": name, "scheme": "finetune", **m_fine, "source": finetune_path.name})
        else:
            print(f"[WARN] missing finetune: {finetune_path.name}")

    if not rows:
        raise RuntimeError("No metrics found. Check files/paths and patterns.")

    # long
    long_df = pd.DataFrame(rows)
    cat_order = ["baseline"] + names if "baseline" in long_df["group"].values else names
    long_df["group"] = pd.Categorical(long_df["group"], categories=cat_order, ordered=True)
    long_df = long_df.sort_values(["group", "scheme"]).reset_index(drop=True)
    long_df.to_csv(out_long, index=False)
    print(f"Saved: {out_long}")

    # wide (+ delta)
    wide_src = long_df[long_df["scheme"].isin(["only10", "finetune"])].copy()
    wide = wide_src.pivot_table(index="group", columns="scheme", values=METRICS, aggfunc="first", observed=False)
    wide.columns = [f"{m}_{sch}" for m, sch in wide.columns]
    wide = wide.reset_index()
    for m in METRICS:
        co, cf = f"{m}_only10", f"{m}_finetune"
        if co in wide.columns and cf in wide.columns:
            wide[f"{m}_delta"] = wide[cf] - wide[co]
    wide.to_csv(out_wide, index=False)
    print(f"Saved: {out_wide}")

    # 改善サマリ
    imp_rows = []
    for m in METRICS:
        dcol = f"{m}_delta"
        if dcol not in wide.columns:
            continue
        vals = wide.loc[wide["group"] != "baseline", dcol] if "baseline" in wide["group"].values else wide[dcol]
        imp_rows.append({
            "metric": m,
            "n_groups": int(vals.notna().sum()),
            "n_improved": int((vals >  0).sum()),
            "n_tied":     int((vals == 0).sum()),
            "n_worse":    int((vals <  0).sum()),
            "mean_delta": float(vals.mean()),
            "median_delta": float(vals.median()),
        })
    improve_df = pd.DataFrame(imp_rows)
    improve_df.to_csv(imp_csv, index=False)
    print(f"Saved: {imp_csv}")

    # Markdown
    lines = []
    lines.append(f"# Summary: only10 vs finetune (groups={len(names)}, split={split or 'ALL'})\n")
    lines.append("## Improvements summary\n")
    head = ["metric","n_groups","n_improved","n_tied","n_worse","mean_delta","median_delta"]
    lines += [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join(["---"]*len(head)) + " |",
    ]
    for _, r in improve_df[head].iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in head) + " |")

    lines.append("\n## Wide (excerpt)\n")
    show_cols = ["group"]
    for m in METRICS:
        for suff in ["only10","finetune","delta"]:
            col = f"{m}_{suff}"
            if col in wide.columns:
                show_cols.append(col)
    lines += [
        "| " + " | ".join(show_cols) + " |",
        "| " + " | ".join(["---"]*len(show_cols)) + " |",
    ]
    for _, r in wide[show_cols].iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in show_cols) + " |")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_md}")

    if make_radar:
        try:
            from .radar import make_radar
        except Exception:
            from src.analysis.radar import make_radar  # type: ignore
        make_radar(wide, model_dir / "radar_allgroups", metrics=METRICS, ylim=(0,1))

    return {"long": out_long, "wide": out_wide, "markdown": out_md, "improvements": imp_csv}


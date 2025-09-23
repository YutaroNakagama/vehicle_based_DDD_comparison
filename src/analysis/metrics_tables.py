# src/analysis/metrics_tables.py
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd
from typing import Optional, Tuple, Dict, List, Any

METRIC_COLS_DEFAULT = ["accuracy", "precision", "recall", "f1", "auc", "ap"]

def _safe_get(d: Dict[str, Any], k: str, default=float("nan")) -> Any:
    """Safely retrieve a value from a dictionary.

    Parameters
    ----------
    d : dict of {str: Any}
        Dictionary from which to retrieve the value.
    k : str
        Key to look up in the dictionary.
    default : Any, default=NaN
        Value to return if the key is not found.

    Returns
    -------
    Any
        The value corresponding to the key if it exists,
        otherwise the default value.
    """
    return d[k] if k in d else default

def _read_test_row(csv_path: Path, split: str = "test") -> Dict[str, Any] | None:
    """Read a metrics row from a CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file containing evaluation metrics.
    split : str, default="test"
        Dataset split to filter (e.g., ``test``).

    Returns
    -------
    dict of {str: Any} or None
        Dictionary of the last row matching the split.  
        If the split column does not exist, the last row of the file is used.  
        Returns ``None`` if the file is missing or cannot be read.
    """
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    use = df[df["split"] == split] if "split" in df.columns else df
    if use.empty:
        use = df.tail(1)
    return use.iloc[-1].to_dict()

def _parse_name(fname: str, model_tag: str) -> Tuple[str, Optional[str]]:
    """Parse the scheme and group name from a metrics filename.

    Parameters
    ----------
    fname : str
        Filename to parse.
    model_tag : str
        Model identifier used in the filename.

    Returns
    -------
    tuple of (str, str or None)
        A tuple where the first element is the scheme
        (``baseline``, ``only10``, or ``finetune``),
        and the second element is the group name, or ``None`` if not applicable.
    """
    # baseline
    if re.fullmatch(rf"metrics_{re.escape(model_tag)}\.csv", fname):
        return "baseline", None

    # only10 (group name free-form)
    m = re.fullmatch(rf"metrics_{re.escape(model_tag)}_only10_(.+)\.csv", fname)
    if m:
        return "only10", m.group(1)

    # finetune (free-form group name between the two tokens)
    m = re.fullmatch(rf"metrics_{re.escape(model_tag)}_finetune_(.+?)_finetune\.csv", fname)
    if m:
        return "finetune", m.group(1)

    # legacy: groupN
    m = re.fullmatch(rf"metrics_{re.escape(model_tag)}_finetune_group(\d+)_finetune\.csv", fname)
    if m:
        return "finetune", f"group{m.group(1)}"

    return "", None

def summarize_metrics(
    model_dir: Path,
    model_tag: str = "RF",
    split: str = "test",
    out_csv: Optional[Path] = None,
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Summarize evaluation metrics across models.

    This function scans a directory for metrics CSVs and compiles them into a
    long-form DataFrame.

    Parameters
    ----------
    model_dir : Path
        Directory containing metrics CSV files.
    model_tag : str, default="RF"
        Model identifier used in filenames.
    split : str, default="test"
        Dataset split to filter (e.g., ``test``).
    out_csv : Path, optional
        If provided, save the summary table to this path.
    metric_cols : list of str, optional
        Metrics to include. Defaults to
        ``["accuracy", "precision", "recall", "f1", "auc", "ap"]``.

    Returns
    -------
    pandas.DataFrame
        Long-form DataFrame with columns: ``group``, ``scheme``, metrics, ``source``.
    """
    metric_cols = metric_cols or METRIC_COLS_DEFAULT
    rows: List[Dict[str, Any]] = []
    for p in model_dir.glob("metrics_*.csv"):
        scheme, group = _parse_name(p.name, model_tag=model_tag)
        if not scheme:  # skip non-matching files
            continue
        rec = _read_test_row(p, split=split)
        if rec is None:
            continue
        row = {
            "group": group,
            "scheme": scheme,
            "source": p.name,
        }
        for m in metric_cols:
            row[m] = _safe_get(rec, m)
        rows.append(row)

    df = pd.DataFrame(rows)
    # sort if columns exist
    sk = [c for c in ["group", "scheme"] if c in df.columns]
    if sk:
        df = df.sort_values(sk).reset_index(drop=True)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df

def make_comparison_table(
    summary_df_or_path: pd.DataFrame | Path,
    out_csv: Optional[Path] = None,
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate a wide comparison table from summarized metrics.

    Converts the long-form summary into a wide table with metrics
    for ``only10`` and ``finetune`` schemes, and their differences (delta).

    Parameters
    ----------
    summary_df_or_path : pandas.DataFrame or Path
        Summary DataFrame or path to CSV (from :func:`summarize_metrics`).
    out_csv : Path, optional
        If provided, save the wide table to this path.
    metric_cols : list of str, optional
        Metrics to include. Defaults to
        ``["accuracy", "precision", "recall", "f1", "auc", "ap"]``.

    Returns
    -------
    pandas.DataFrame
        Wide-form DataFrame with columns for each metric:
        ``{metric}_only10``, ``{metric}_finetune``, and ``{metric}_delta``.
    """
    metric_cols = metric_cols or METRIC_COLS_DEFAULT

    if isinstance(summary_df_or_path, Path):
        df = pd.read_csv(summary_df_or_path)
    else:
        df = summary_df_or_path.copy()

    # normalize scheme labels just in case
    scheme_map = {
        "with_pretrain(finetune)": "finetune",
        "without_pretrain(only10)": "only10",
        "baseline": "baseline",
    }
    if "scheme" in df.columns:
        df["scheme"] = df["scheme"].map(lambda x: scheme_map.get(x, x))

    # keep only only10/finetune for pivot
    df2 = df[df["scheme"].isin(["only10", "finetune"])].copy()
    # keep group as str (free-form)
    if "group" in df2.columns:
        df2["group"] = df2["group"].astype(str)

    wide = df2.pivot_table(
        index="group",
        columns="scheme",
        values=[m for m in metric_cols if m in df2.columns],
        aggfunc="first",
        observed=False,  # future-proof for pandas
    )

    # build output columns in the order: only10, finetune, delta
    out = pd.DataFrame(index=wide.index)
    for m in metric_cols:
        c_only = (m, "only10")
        c_fine = (m, "finetune")
        if c_only in wide.columns:
            out[f"{m}_only10"] = wide[c_only]
        if c_fine in wide.columns:
            out[f"{m}_finetune"] = wide[c_fine]
        if c_only in wide.columns and c_fine in wide.columns:
            out[f"{m}_delta"] = wide[c_fine] - wide[c_only]

    out = out.sort_index()

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv)
    return out


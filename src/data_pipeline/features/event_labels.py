"""Event-Based Labeling for Driver Cognitive State Detection.

This module implements task-event-based binary labeling following Wang et al.
(2022)'s methodology, adapted for the Scheutz et al. (2023) Tufts driving
simulator dataset.

Labeling Scheme (Wang et al. 2022)
----------------------------------
- **Class 1 (Task/Distracted)**: Time windows during driving events
  (EventTime[i, 0] to EventTime[i, 1]).
- **Class 0 (Baseline/Focused)**: 20 s *before* each event start and
  10 s *after* each event end.
- **Discarded**: All other time intervals (not within any event or baseline
  window).

When baseline windows overlap between consecutive events, boundaries are
clipped to the midpoint of the inter-event gap so that each time point
receives a unique label.

References
----------
Wang, X. et al. (2022). "Driver distraction detection based on vehicle
dynamics using naturalistic driving data." *Transportation Research Part C*,
136, 103561.

Scheutz, M. et al. (2023). "Estimating Systemic Cognitive States from a
Mixture of Physiological and Brain Signals." *Topics in Cognitive Science*,
16, 485–526.
"""

from __future__ import annotations

import os
import logging

import numpy as np
import pandas as pd
import scipy.io as sio

from src.config import DATASET_PATH, INTRIM_CSV_PATH, PROCESS_CSV_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ── Wang et al. 2022 baseline parameters ───────────────────────────────────
PRE_EVENT_BASELINE_SEC: float = 20.0
"""Seconds before each event start to label as baseline (class 0)."""

POST_EVENT_BASELINE_SEC: float = 10.0
"""Seconds after each event end to label as baseline (class 0)."""

# Column name for the binary event label
EVENT_LABEL_COL: str = "event_label"
"""Name of the column storing the binary event label (0 = baseline, 1 = task)."""


# ── Helper: load EventTimes ────────────────────────────────────────────────
def _load_event_times(subject_id: str, version: str) -> dict | None:
    """Load ``EventTimes_<id>_<ver>.mat`` and return parsed arrays.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., ``"S0113"``).
    version : str
        Trial version (``"1"`` or ``"2"``).

    Returns
    -------
    dict or None
        Dictionary with keys ``"event_time"`` (N×2 float array of
        [start, end] pairs) and ``"event_desc"`` (N×2 uint8 array where
        col-0 = brake flag, col-1 = DRT flag).  Returns ``None`` if the
        file is missing.
    """
    mat_path = os.path.join(
        DATASET_PATH, subject_id,
        f"EventTimes_{subject_id}_{version}.mat",
    )
    if not os.path.isfile(mat_path):
        logging.warning("EventTimes file not found: %s", mat_path)
        return None

    mat = sio.loadmat(mat_path)
    return {
        "event_time": mat["EventTime"].astype(np.float64),   # (N, 2)
        "event_desc": mat["EventDiscription"].astype(np.uint8),  # (N, 2)
    }


# ── Helper: build label ranges ─────────────────────────────────────────────
def _build_label_ranges(
    event_times: np.ndarray,
    t_min: float,
    t_max: float,
    pre_sec: float = PRE_EVENT_BASELINE_SEC,
    post_sec: float = POST_EVENT_BASELINE_SEC,
) -> list[tuple[float, float, int]]:
    """Create non-overlapping (start, end, label) intervals.

    For each event window the label is **1**.  Pre-event and post-event
    baselines receive label **0**.  When adjacent baselines would overlap,
    the boundary is placed at the midpoint of the gap.

    Parameters
    ----------
    event_times : ndarray, shape (N, 2)
        ``event_times[i] = [start_i, end_i]`` in seconds.
    t_min, t_max : float
        Valid time range of the recording.
    pre_sec : float
        Baseline duration before each event start.
    post_sec : float
        Baseline duration after each event end.

    Returns
    -------
    list of (start, end, label)
        Sorted, non-overlapping intervals covering only the labelled
        portions of the recording.
    """
    n_events = event_times.shape[0]
    ranges: list[tuple[float, float, int]] = []

    for i in range(n_events):
        ev_start = event_times[i, 0]
        ev_end = event_times[i, 1]

        # ── Pre-event baseline (class 0) ───────────────────────────────
        pre_start = max(ev_start - pre_sec, t_min)
        # Clip against previous event's post-baseline
        if i > 0:
            prev_ev_end = event_times[i - 1, 1]
            prev_post_end = prev_ev_end + post_sec
            gap_mid = (prev_post_end + (ev_start - pre_sec)) / 2.0
            # If gap is tight, split at midpoint; never overlap
            pre_start = max(pre_start, min(prev_post_end, gap_mid))

        if pre_start < ev_start:
            ranges.append((pre_start, ev_start, 0))

        # ── Event window (class 1) ─────────────────────────────────────
        ranges.append((ev_start, ev_end, 1))

        # ── Post-event baseline (class 0) ──────────────────────────────
        post_end = min(ev_end + post_sec, t_max)
        # Clip against next event's pre-baseline
        if i < n_events - 1:
            next_ev_start = event_times[i + 1, 0]
            next_pre_start = next_ev_start - pre_sec
            gap_mid = (post_end + next_pre_start) / 2.0
            post_end = min(post_end, max(next_pre_start, gap_mid))

        if ev_end < post_end:
            ranges.append((ev_end, post_end, 0))

    return ranges


def _assign_labels(
    timestamps: np.ndarray,
    ranges: list[tuple[float, float, int]],
) -> np.ndarray:
    """Assign a label to each timestamp based on intervals.

    Parameters
    ----------
    timestamps : ndarray, shape (M,)
        Sorted timestamp array from the feature CSV.
    ranges : list of (start, end, label)
        Non-overlapping labelled intervals.

    Returns
    -------
    ndarray, shape (M,)
        Label array.  ``-1`` indicates "outside all labelled windows"
        (to be dropped).
    """
    labels = np.full(len(timestamps), -1, dtype=np.int8)
    for r_start, r_end, lbl in ranges:
        mask = (timestamps >= r_start) & (timestamps < r_end)
        labels[mask] = lbl
    return labels


# ── Public API ──────────────────────────────────────────────────────────────
def event_label_process(subject: str, model_name: str) -> None:
    """Add event-based labels to a merged feature CSV and save to processed/.

    This replaces KSS-based labeling for the Lstm model.  The merged CSV
    is loaded from ``data/interim/merged/<model_name>/``, event labels are
    assigned from ``EventTimes_<id>_<ver>.mat``, unlabelled rows are
    dropped, and the result is saved to ``data/processed/<model_name>/``.

    Parameters
    ----------
    subject : str
        Subject in ``"<id>_<version>"`` format (e.g., ``"S0113_1"``).
    model_name : str
        Model architecture name (e.g., ``"Lstm"``).
    """
    parts = subject.split("_")
    if len(parts) != 2:
        logging.error("Unexpected subject format: %s", subject)
        return
    subject_id, version = parts

    # ── Load merged features ────────────────────────────────────────────
    merged_path = os.path.join(
        INTRIM_CSV_PATH, "merged", model_name,
        f"merged_{subject_id}_{version}.csv",
    )
    if not os.path.isfile(merged_path):
        logging.error("Merged file not found: %s", merged_path)
        return

    data = pd.read_csv(merged_path)
    if "Timestamp" not in data.columns:
        logging.error("No Timestamp column in %s", merged_path)
        return

    timestamps = data["Timestamp"].values

    # ── Load EventTimes ────────────────────────────────────────────────
    et = _load_event_times(subject_id, version)
    if et is None:
        logging.error(
            "Cannot label %s_%s — EventTimes not available.", subject_id, version,
        )
        return

    event_times = et["event_time"]  # (N, 2)
    t_min = timestamps.min()
    t_max = timestamps.max()

    # ── Build label intervals ──────────────────────────────────────────
    ranges = _build_label_ranges(event_times, t_min, t_max)

    # ── Assign labels ──────────────────────────────────────────────────
    labels = _assign_labels(timestamps, ranges)
    data[EVENT_LABEL_COL] = labels

    # ── Drop unlabelled rows (label == -1) ─────────────────────────────
    n_before = len(data)
    data = data[data[EVENT_LABEL_COL] >= 0].copy()
    n_after = len(data)

    n_event = (data[EVENT_LABEL_COL] == 1).sum()
    n_baseline = (data[EVENT_LABEL_COL] == 0).sum()
    logging.info(
        "%s_%s: %d→%d rows kept (event=%d, baseline=%d, dropped=%d)",
        subject_id, version, n_before, n_after,
        n_event, n_baseline, n_before - n_after,
    )

    # ── Save ───────────────────────────────────────────────────────────
    out_dir = os.path.join(PROCESS_CSV_PATH, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"processed_{subject_id}_{version}.csv")
    data.to_csv(out_path, index=False)
    logging.info("Event-labelled data saved: %s", out_path)

"""One-off: regenerate Lstm processed CSVs with event_label column.

Re-runs `event_label_process` for every subject in the subject list, using the
already-existing merged Lstm CSVs at data/interim/merged/Lstm/.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from src.data_pipeline.features.event_labels import event_label_process
from src.utils.io.loaders import read_subject_list


def _worker(args):
    subject, idx, total = args
    try:
        logging.info(f"[{idx+1}/{total}] {subject}")
        event_label_process(subject, "Lstm")
    except Exception as e:
        logging.error(f"Error for {subject}: {e}")


def main() -> None:
    subjects = read_subject_list()
    total = len(subjects)
    n_proc = int(os.environ.get("N_PROC", min(mp.cpu_count(), total, 16)))
    logging.info(f"Running event_label_process for {total} subjects with {n_proc} workers.")
    args = [(s, i, total) for i, s in enumerate(subjects)]
    with mp.Pool(processes=n_proc) as pool:
        pool.map(_worker, args)
    logging.info("Done.")


if __name__ == "__main__":
    sys.exit(main())

"""WSL2/CUDA launcher for Lstm exp3 non-SW-SMOTE conditions (NVIDIA RTX 3060).

Covers the 3 conditions that local Phase 2 (imbalv3 SW-SMOTE) did NOT run:
  - baseline       (no rebalancing)
  - smote_plain    r=0.5  (global SMOTE, not per-subject)
  - undersample_rus r=0.5  (random undersampling)

r=0.1 is N/A for Lstm with DRT labels (natural minority ≈27% > target 0.1).

Run from WSL2 Ubuntu:
    /home/ynakagama/.venv_tf_gpu/bin/python \\
        /mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison/\\
        scripts/python/train/local_exp3_lstm_wsl2_other_launcher.py

Dry-run to preview pending jobs:
    ... local_exp3_lstm_wsl2_other_launcher.py --dry-run

Design:
  Conditions: baseline / smote_plain r=0.5 / undersample_rus r=0.5
  Distance:   mmd, wasserstein, dtw (3)
  Domain:     in_domain, out_domain (2)
  Seeds:      same 15 as Phase 2
  Total:      (1 + 1 + 1) conditions × 3 × 2 × 15 = 270 jobs
  GPU:        CUDA_VISIBLE_DEVICES=0 (NVIDIA RTX 3060, sole CUDA GPU in WSL2)
  Workers:    8 (shared GPU via TF_FORCE_GPU_ALLOW_GROWTH)
  ETA:        ~270 / 8 × 6 min ≈ 3.4 h
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
)

REPO = Path("/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison")
VENV_PYTHON = Path("/home/ynakagama/.venv_tf_gpu/bin/python")

if not REPO.exists():
    raise FileNotFoundError(f"Repo not found at WSL2 mount: {REPO}")
if not VENV_PYTHON.exists():
    raise FileNotFoundError(f"WSL2 GPU venv not found: {VENV_PYTHON}")

RANKINGS_DIR = (
    REPO / "results" / "analysis" / "exp2_domain_shift"
    / "distance" / "rankings" / "split2" / "knn"
)
EVAL_DIR = REPO / "results" / "outputs" / "evaluation"
LOG_DIR = REPO / "logs" / "exp3_lstm_wsl2"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DISTANCES = ["mmd", "wasserstein", "dtw"]
DOMAINS = ["in_domain", "out_domain"]
SEEDS = [0, 7, 42, 99, 123, 256, 512, 777, 1000, 1337, 2024, 2025, 3407, 9999, 31415]

# Each entry: (condition_name, ratio_or_None)
# Naming matches PBS wrapper: baseline→no ratio, others→ratio in tag
CONDITIONS: list[tuple[str, Optional[float]]] = [
    ("baseline",        None),
    ("smote_plain",     0.5),
    ("undersample_rus", 0.5),
]

N_WORKERS = 8


# ---------------------------------------------------------------------------

@dataclass
class Job:
    condition: str          # "baseline" | "smote_plain" | "undersample_rus"
    ratio: Optional[float]  # None for baseline
    distance: str
    domain: str
    seed: int

    @property
    def tag(self) -> str:
        if self.condition == "baseline":
            return (
                f"prior_Lstm_baseline_knn_{self.distance}_{self.domain}"
                f"_domain_train_split2_s{self.seed}"
            )
        return (
            f"prior_Lstm_{self.condition}_knn_{self.distance}_{self.domain}"
            f"_domain_train_split2_ratio{self.ratio}_s{self.seed}"
        )

    @property
    def target_file(self) -> Path:
        return RANKINGS_DIR / f"{self.distance}_{self.domain}.txt"

    @property
    def cross_target_file(self) -> Path:
        cross = "out_domain" if self.domain == "in_domain" else "in_domain"
        return RANKINGS_DIR / f"{self.distance}_{cross}.txt"

    @property
    def eval_within_json_name(self) -> str:
        return f"eval_results_Lstm_domain_train_{self.tag}_within.json"

    def already_done(self, done_set: set) -> bool:
        return self.eval_within_json_name in done_set

    def train_oversampling_args(self) -> list[str]:
        """Return train.py args for this condition's rebalancing."""
        if self.condition == "baseline":
            return []
        if self.condition == "smote_plain":
            return [
                "--use_oversampling",
                "--oversample_method", "smote",
                "--target_ratio", str(self.ratio),
            ]
        if self.condition == "undersample_rus":
            return [
                "--use_oversampling",
                "--oversample_method", "undersample_rus",
                "--target_ratio", str(self.ratio),
            ]
        raise ValueError(f"Unknown condition: {self.condition}")


def build_done_set() -> set:
    base = EVAL_DIR / "Lstm"
    if not base.exists():
        return set()
    return {f.name for f in base.rglob("*.json")}


def build_jobs() -> List[Job]:
    jobs = []
    for cond, ratio in CONDITIONS:
        for dist in DISTANCES:
            for dom in DOMAINS:
                for s in SEEDS:
                    jobs.append(Job(condition=cond, ratio=ratio,
                                    distance=dist, domain=dom, seed=s))
    return jobs


# ---------------------------------------------------------------------------

def run_one(job: Job) -> int:
    tag = job.tag
    log_path = LOG_DIR / f"{tag}.log"
    jobid = f"{int(time.time() * 1000)}{os.getpid()}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PBS_JOBID"] = jobid
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env["OMP_NUM_THREADS"] = "2"
    env["TF_NUM_INTRAOP_THREADS"] = "2"
    env["TF_NUM_INTEROP_THREADS"] = "1"

    train_cmd = [
        str(VENV_PYTHON), "scripts/python/train/train.py",
        "--model", "Lstm",
        "--mode", "domain_train",
        "--seed", str(job.seed),
        "--target_file", str(job.target_file.relative_to(REPO)),
        "--tag", tag,
        "--time_stratify_labels",
        *job.train_oversampling_args(),
    ]
    eval_within_cmd = [
        str(VENV_PYTHON), "scripts/python/evaluation/evaluate.py",
        "--model", "Lstm",
        "--tag", tag,
        "--mode", "domain_train",
        "--target_file", str(job.target_file.relative_to(REPO)),
        "--eval_type", "within",
        "--jobid", jobid,
    ]
    eval_cross_cmd = [
        str(VENV_PYTHON), "scripts/python/evaluation/evaluate.py",
        "--model", "Lstm",
        "--tag", tag,
        "--mode", "domain_train",
        "--target_file", str(job.cross_target_file.relative_to(REPO)),
        "--eval_type", "cross",
        "--jobid", jobid,
    ]

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# JOBID={jobid}  TAG={tag}  COND={job.condition}\n")
        lf.write(f"# Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lf.flush()

        def _run(cmd, label):
            lf.write(f"\n=== [{label}] {' '.join(str(c) for c in cmd)} ===\n")
            lf.flush()
            p = subprocess.run(
                cmd, cwd=str(REPO), env=env,
                stdout=lf, stderr=subprocess.STDOUT,
            )
            return p.returncode

        rc = _run(train_cmd, "TRAIN")
        if rc != 0:
            logging.error("Train FAILED for %s (rc=%d)", tag, rc)
            return rc
        rc1 = _run(eval_within_cmd, "EVAL within")
        rc2 = _run(eval_cross_cmd, "EVAL cross")
        elapsed = time.time() - start
        lf.write(f"\n# Finished in {elapsed:.1f}s | CUDA:0 | rc1={rc1} rc2={rc2}\n")

    logging.info(
        "DONE %s CUDA:0 in %.1fs (rc1=%d rc2=%d)", tag, elapsed, rc1, rc2
    )
    return 0


def worker_loop(name: str, q: "queue.Queue[Job]") -> None:
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        try:
            logging.info("START %s | cond=%s", job.tag, job.condition)
            run_one(job)
        except Exception:
            logging.exception("Job crashed: %s", job.tag)
        finally:
            q.task_done()


# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print pending job tags and exit (no training).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run at most N jobs (for testing).")
    args = ap.parse_args()

    done_set = build_done_set()
    all_jobs = build_jobs()
    pending = [j for j in all_jobs if not j.already_done(done_set)]
    done_count = len(all_jobs) - len(pending)

    logging.info(
        "Total=%d  Done=%d  Pending=%d  Seeds=%d  Workers(CUDA:0)=%d",
        len(all_jobs), done_count, len(pending), len(SEEDS), N_WORKERS,
    )

    # Condition breakdown
    from collections import Counter
    cond_counts = Counter(j.condition for j in pending)
    for cond, cnt in sorted(cond_counts.items()):
        logging.info("  pending %-20s %d jobs", cond, cnt)

    if args.dry_run:
        for j in pending:
            print(j.tag)
        return

    if args.limit:
        pending = pending[: args.limit]
        logging.info("Limiting to first %d pending jobs.", args.limit)

    q: "queue.Queue[Job]" = queue.Queue()
    for j in pending:
        q.put(j)

    threads = []
    for i in range(min(N_WORKERS, len(pending))):
        t = threading.Thread(
            target=worker_loop, args=(f"CUDA-{i}", q), name=f"CUDA-{i}",
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    logging.info("All Lstm other-condition CUDA workers finished.")


if __name__ == "__main__":
    main()

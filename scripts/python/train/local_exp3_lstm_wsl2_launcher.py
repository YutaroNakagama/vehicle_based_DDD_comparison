"""WSL2/CUDA launcher for Lstm exp3 seed expansion (NVIDIA RTX 3060, CUDA).

Run this script from inside WSL2 Ubuntu:
    /home/ynakagama/.venv_tf_gpu/bin/python \
        /mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison/\
        scripts/python/train/local_exp3_lstm_wsl2_launcher.py

Continues from where the DirectML run left off — already_done() skips
any jobs whose eval JSON already exists in the Windows filesystem via
/mnt/c/... paths.

Design:
  Models:    Lstm only
  Distance:  mmd, wasserstein, dtw
  Domain:    in_domain, out_domain
  Ratios:    0.3, 0.5
  Seeds:     15 total (includes existing [42, 123, 2025])
  => 3 x 2 x 2 x 15 = 180 jobs total

GPU assignment (CUDA):
  CUDA_VISIBLE_DEVICES=0 -> NVIDIA RTX 3060 (sole CUDA-visible GPU in WSL2)
  N_WORKERS = 8 (CUDA handles parallel jobs efficiently on a single GPU)
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
)

# Linux mount point of the Windows repo (WSL2 path)
REPO = Path("/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison")
VENV_PYTHON = Path("/home/ynakagama/.venv_tf_gpu/bin/python")

if not REPO.exists():
    raise FileNotFoundError(f"Repo not found at WSL2 mount: {REPO}")
if not VENV_PYTHON.exists():
    raise FileNotFoundError(f"WSL2 GPU venv not found: {VENV_PYTHON}")

RANKINGS_DIR = REPO / "results" / "analysis" / "exp2_domain_shift" / "distance" / "rankings" / "split2" / "knn"
EVAL_DIR = REPO / "results" / "outputs" / "evaluation"
LOG_DIR = REPO / "logs" / "exp3_lstm_wsl2"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DISTANCES = ["mmd", "wasserstein", "dtw"]
DOMAINS = ["in_domain", "out_domain"]
RATIOS = [0.3, 0.5]

# 15 seeds: includes existing [42, 123, 2025] (skipped via already_done())
SEEDS = [0, 7, 42, 99, 123, 256, 512, 777, 1000, 1337, 2024, 2025, 3407, 9999, 31415]

# CUDA: 1 GPU (RTX 3060), 8 workers share it via CUDA multi-process
N_WORKERS = 8


@dataclass
class Job:
    distance: str
    domain: str
    ratio: float
    seed: int

    @property
    def tag(self) -> str:
        return (
            f"prior_Lstm_imbalv3_knn_{self.distance}_{self.domain}"
            f"_domain_train_split2_subjectwise_ratio{self.ratio}_s{self.seed}"
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


def build_done_set() -> set:
    """Single scan of eval dir — O(n_files) instead of O(n_jobs * n_files)."""
    base = EVAL_DIR / "Lstm"
    if not base.exists():
        return set()
    return {f.name for f in base.rglob("*.json")}


def build_jobs() -> List[Job]:
    jobs = []
    for dist in DISTANCES:
        for dom in DOMAINS:
            for r in RATIOS:
                for s in SEEDS:
                    jobs.append(Job(distance=dist, domain=dom, ratio=r, seed=s))
    return jobs


def run_one(job: Job) -> int:
    """Run train + 2 evals for one Lstm job on CUDA GPU 0."""
    tag = job.tag
    log_path = LOG_DIR / f"{tag}.log"
    jobid = f"{int(time.time() * 1000)}{os.getpid()}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PBS_JOBID"] = jobid
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # CUDA GPU selection: RTX 3060 is GPU 0 inside WSL2
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # Allow TF to grow GPU memory gradually rather than pre-allocating all
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # Threading: 2 intra-op threads per job
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
        "--use_oversampling",
        "--oversample_method", "smote",
        "--target_ratio", str(job.ratio),
        "--subject_wise_oversampling",
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
        lf.write(f"# JOBID={jobid}  TAG={tag}  GPU=CUDA:0\n")
        lf.write(f"# Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lf.flush()

        def _run(cmd, label):
            lf.write(f"\n=== [{label}] {' '.join(str(c) for c in cmd)} ===\n")
            lf.flush()
            p = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT)
            return p.returncode

        rc = _run(train_cmd, "TRAIN")
        if rc != 0:
            logging.error("Train FAILED for %s (rc=%d)", tag, rc)
            return rc
        rc1 = _run(eval_within_cmd, "EVAL within")
        rc2 = _run(eval_cross_cmd, "EVAL cross")
        elapsed = time.time() - start
        lf.write(f"\n# Finished in {elapsed:.1f}s | CUDA:0 | rc1={rc1} rc2={rc2}\n")

    logging.info("DONE %s CUDA:0 in %.1fs (rc1=%d rc2=%d)", tag, elapsed, rc1, rc2)
    return 0


def worker_loop(name: str, q: "queue.Queue[Job]") -> None:
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        try:
            logging.info("START %s on CUDA:0", job.tag)
            run_one(job)
        except Exception:
            logging.exception("Job crashed: %s", job.tag)
        finally:
            q.task_done()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print pending job tags and exit.")
    ap.add_argument("--limit", type=int, default=None, help="Run at most N jobs (for testing).")
    args = ap.parse_args()

    done_set = build_done_set()
    all_jobs = build_jobs()
    pending = [j for j in all_jobs if not j.already_done(done_set)]
    done = len(all_jobs) - len(pending)
    logging.info(
        "Total=%d  Done=%d  Pending=%d  Seeds=%d  Workers(CUDA:0)=%d",
        len(all_jobs), done, len(pending), len(SEEDS), N_WORKERS,
    )

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
    for i in range(N_WORKERS):
        t = threading.Thread(target=worker_loop, args=(f"CUDA-{i}", q), name=f"CUDA-{i}")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    logging.info("All WSL2/CUDA Lstm workers finished.")


if __name__ == "__main__":
    main()

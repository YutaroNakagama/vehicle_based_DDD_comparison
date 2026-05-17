"""GPU launcher for Lstm exp3 seed expansion (DirectML: NVIDIA RTX 3060 + Intel Iris Xe).

Extends the local 3-seed SW-SMOTE results to 15 seeds by running 12 additional
seeds on the two GPU adapters available via tensorflow-directml-plugin.

Virtual env: .venv_tf210_gpu (TF 2.10.1 + tensorflow-directml-plugin)

Design:
  Models:    Lstm only
  Distance:  mmd, wasserstein, dtw
  Domain:    in_domain, out_domain
  Ratios:    0.3, 0.5
  Seeds:     15 total (includes existing [42, 123, 2025]; already_done() skips them)
  => 3 x 2 x 2 x 15 = 180 jobs (144 new + 36 existing/skipped)

GPU assignment (DML_VISIBLE_DEVICES):
  Workers 0-3 -> GPU:1 (NVIDIA RTX 3060, faster)
  Workers 4-5 -> GPU:0 (Intel Iris Xe, slower but available)
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

REPO = Path(__file__).resolve().parents[3]
# GPU venv Python (TF 2.10.1 + DirectML)
VENV_PYTHON = REPO / ".venv_tf210_gpu" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    raise FileNotFoundError(f"GPU venv not found: {VENV_PYTHON}")

RANKINGS_DIR = REPO / "results" / "analysis" / "exp2_domain_shift" / "distance" / "rankings" / "split2" / "knn"
EVAL_DIR = REPO / "results" / "outputs" / "evaluation"
LOG_DIR = REPO / "logs" / "exp3_lstm_gpu"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DISTANCES = ["mmd", "wasserstein", "dtw"]
DOMAINS = ["in_domain", "out_domain"]
RATIOS = [0.3, 0.5]

# 15 seeds: includes existing [42, 123, 2025] (skipped via already_done())
SEEDS = [0, 7, 42, 99, 123, 256, 512, 777, 1000, 1337, 2024, 2025, 3407, 9999, 31415]

# GPU assignment (DML adapter order is reversed from Windows Task Manager):
#   DML_VISIBLE_DEVICES=0 -> NVIDIA RTX 3060 (Windows GPU 1, VideoController2)
#   DML_VISIBLE_DEVICES=1 -> Intel Iris Xe   (Windows GPU 0, VideoController1)
N_WORKERS_NVIDIA = 8   # DML adapter 0 = NVIDIA RTX 3060
N_WORKERS_INTEL = 2    # DML adapter 1 = Intel Iris Xe
N_WORKERS_TOTAL = N_WORKERS_NVIDIA + N_WORKERS_INTEL


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

    def already_done(self) -> bool:
        base = EVAL_DIR / "Lstm"
        if not base.exists():
            return False
        return any(base.rglob(self.eval_within_json_name))


def build_jobs() -> List[Job]:
    jobs = []
    for dist in DISTANCES:
        for dom in DOMAINS:
            for r in RATIOS:
                for s in SEEDS:
                    jobs.append(Job(distance=dist, domain=dom, ratio=r, seed=s))
    return jobs


def run_one(job: Job, gpu_id: int) -> int:
    """Run train + 2 evals for one Lstm job on specified GPU."""
    tag = job.tag
    log_path = LOG_DIR / f"{tag}.log"
    jobid = f"{int(time.time() * 1000)}{os.getpid()}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PBS_JOBID"] = jobid
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # DirectML GPU selection
    env["DML_VISIBLE_DEVICES"] = str(gpu_id)
    # TF threading: each job gets 2 intra-op threads
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
        lf.write(f"# JOBID={jobid}  TAG={tag}  GPU={gpu_id}\n")
        lf.write(f"# Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lf.flush()

        def _run(cmd, label):
            lf.write(f"\n=== [{label}] {' '.join(str(c) for c in cmd)} ===\n")
            lf.flush()
            p = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT)
            return p.returncode

        rc = _run(train_cmd, "TRAIN")
        if rc != 0:
            logging.error("Train FAILED for %s on GPU:%d (rc=%d)", tag, gpu_id, rc)
            return rc
        rc1 = _run(eval_within_cmd, "EVAL within")
        rc2 = _run(eval_cross_cmd, "EVAL cross")
        elapsed = time.time() - start
        lf.write(f"\n# Finished in {elapsed:.1f}s | GPU={gpu_id} | rc1={rc1} rc2={rc2}\n")

    logging.info("DONE %s GPU:%d in %.1fs (rc1=%d rc2=%d)", tag, gpu_id, elapsed, rc1, rc2)
    return 0


def worker_loop(name: str, q: "queue.Queue[Job]", gpu_id: int) -> None:
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        try:
            if job.already_done():
                logging.info("SKIP %s (already done)", job.tag)
            else:
                logging.info("START %s on GPU:%d", job.tag, gpu_id)
                run_one(job, gpu_id)
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

    all_jobs = build_jobs()
    pending = [j for j in all_jobs if not j.already_done()]
    done = len(all_jobs) - len(pending)
    logging.info(
        "Total=%d  Done=%d  Pending=%d  Seeds=%d  "
        "Workers: NVIDIA(GPU:1)=%d + Intel(GPU:0)=%d",
        len(all_jobs), done, len(pending), len(SEEDS),
        N_WORKERS_NVIDIA, N_WORKERS_INTEL,
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
    # NVIDIA workers (DML adapter 0 = NVIDIA RTX 3060, faster)
    for i in range(N_WORKERS_NVIDIA):
        t = threading.Thread(target=worker_loop, args=(f"NVIDIA-{i}", q, 0), name=f"NVIDIA-{i}")
        t.start()
        threads.append(t)
    # Intel workers (DML adapter 1 = Intel Iris Xe, slower)
    for i in range(N_WORKERS_INTEL):
        t = threading.Thread(target=worker_loop, args=(f"Intel-{i}", q, 1), name=f"Intel-{i}")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    logging.info("All GPU Lstm workers finished.")


if __name__ == "__main__":
    main()

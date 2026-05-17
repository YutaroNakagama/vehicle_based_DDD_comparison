"""Force-run specific Lstm jobs regardless of already_done status.

Used to re-run the 2 mmd_in_domain/ratio0.3 jobs (s777, s1000) whose
cross-eval JSON was missing after the DirectML launcher was killed.

Run from WSL2:
    /home/ynakagama/.venv_tf_gpu/bin/python \
        /mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison/\
        scripts/python/train/run_forced_lstm_jobs.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
)

REPO = Path("/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison")
VENV_PYTHON = Path("/home/ynakagama/.venv_tf_gpu/bin/python")

RANKINGS_DIR = REPO / "results" / "analysis" / "exp2_domain_shift" / "distance" / "rankings" / "split2" / "knn"
LOG_DIR = REPO / "logs" / "exp3_lstm_wsl2"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Jobs to force-run: (distance, domain, ratio, seed)
FORCED_JOBS = [
    ("mmd", "in_domain", 0.3, 777),
    ("mmd", "in_domain", 0.3, 1000),
]


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
        "--use_oversampling",
        "--oversample_method", "smote",
        "--target_ratio", str(job.ratio),
        "--subject_wise_oversampling",
    ]
    eval_within_cmd = [
        str(VENV_PYTHON), "scripts/python/evaluation/evaluate.py",
        "--model", "Lstm", "--tag", tag, "--mode", "domain_train",
        "--target_file", str(job.target_file.relative_to(REPO)),
        "--eval_type", "within", "--jobid", jobid,
    ]
    eval_cross_cmd = [
        str(VENV_PYTHON), "scripts/python/evaluation/evaluate.py",
        "--model", "Lstm", "--tag", tag, "--mode", "domain_train",
        "--target_file", str(job.cross_target_file.relative_to(REPO)),
        "--eval_type", "cross", "--jobid", jobid,
    ]

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# FORCED RE-RUN  JOBID={jobid}  TAG={tag}\n")
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
        lf.write(f"\n# Finished in {elapsed:.1f}s | rc1={rc1} rc2={rc2}\n")

    logging.info("DONE %s in %.1fs (rc1=%d rc2=%d)", tag, elapsed, rc1, rc2)
    return 0


def worker_loop(name: str, q: "queue.Queue[Job]") -> None:
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        try:
            logging.info("START %s", job.tag)
            run_one(job)
        except Exception:
            logging.exception("Job crashed: %s", job.tag)
        finally:
            q.task_done()


def main():
    jobs = [Job(d, dom, r, s) for d, dom, r, s in FORCED_JOBS]
    logging.info("Force-running %d jobs", len(jobs))
    for j in jobs:
        logging.info("  %s", j.tag)

    q: "queue.Queue[Job]" = queue.Queue()
    for j in jobs:
        q.put(j)

    threads = []
    for i in range(len(jobs)):
        t = threading.Thread(target=worker_loop, args=(f"W-{i}", q), name=f"W-{i}")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    logging.info("Forced re-run complete.")


if __name__ == "__main__":
    main()

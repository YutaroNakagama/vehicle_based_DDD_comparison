"""Local launcher for 108-job prior-research exp3 subset.

Subset: SW-SMOTE + domain_train only.
- Models:   SvmW, SvmA, Lstm
- Distance: mmd, wasserstein, dtw
- Domain:   in_domain, out_domain
- Ratios:   0.3, 0.5
- Seeds:    42, 123, 2025
=> 3 x 3 x 2 x 2 x 3 = 108 jobs.

Per job: 1 training + 2 evaluations (within + cross).

Parallelism: SvmW x 2, SvmA x 1, Lstm x 1 (queues separated per model).
Skips jobs whose within-domain eval JSON already exists.
"""
from __future__ import annotations

import argparse
import json
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
PYTHON = sys.executable
RANKINGS_DIR = REPO / "results" / "analysis" / "exp2_domain_shift" / "distance" / "rankings" / "split2" / "knn"
EVAL_DIR = REPO / "results" / "outputs" / "evaluation"
LOG_DIR = REPO / "logs" / "exp3_local"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["SvmW", "SvmA", "Lstm"]
DISTANCES = ["mmd", "wasserstein", "dtw"]
DOMAINS = ["in_domain", "out_domain"]
RATIOS = [0.3, 0.5]
SEEDS = [42, 123, 2025]

PARALLELISM = {"SvmW": 4, "SvmA": 6, "Lstm": 3}


@dataclass
class Job:
    model: str
    distance: str
    domain: str
    ratio: float
    seed: int

    @property
    def tag(self) -> str:
        return (
            f"prior_{self.model}_imbalv3_knn_{self.distance}_{self.domain}"
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
        return f"eval_results_{self.model}_domain_train_{self.tag}_within.json"

    def already_done(self) -> bool:
        # Search recursively under EVAL_DIR/<model>/
        base = EVAL_DIR / self.model
        if not base.exists():
            return False
        return any(base.rglob(self.eval_within_json_name))


def build_jobs() -> List[Job]:
    jobs: List[Job] = []
    for model in MODELS:
        for dist in DISTANCES:
            for dom in DOMAINS:
                for r in RATIOS:
                    for s in SEEDS:
                        jobs.append(Job(model=model, distance=dist, domain=dom, ratio=r, seed=s))
    return jobs


def run_one(job: Job) -> int:
    """Run train + 2 evals for one job. Returns exit code."""
    tag = job.tag
    log_path = LOG_DIR / f"{tag}.log"
    jobid = f"{int(time.time() * 1000)}{os.getpid()}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PBS_JOBID"] = jobid
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    # GPU only for Lstm
    if job.model != "Lstm":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        # TF 2.13 on Windows is CPU-only. Limit threads so concurrent
        # Lstm workers don't oversubscribe cores.
        env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
        env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    # SvmW Optuna trials (default 100). Keep default if env not set explicitly.
    env.setdefault("N_TRIALS_OVERRIDE", os.environ.get("N_TRIALS_OVERRIDE", "100"))

    train_cmd = [
        PYTHON, "scripts/python/train/train.py",
        "--model", job.model,
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
        PYTHON, "scripts/python/evaluation/evaluate.py",
        "--model", job.model,
        "--tag", tag,
        "--mode", "domain_train",
        "--target_file", str(job.target_file.relative_to(REPO)),
        "--eval_type", "within",
        "--jobid", jobid,
    ]
    eval_cross_cmd = [
        PYTHON, "scripts/python/evaluation/evaluate.py",
        "--model", job.model,
        "--tag", tag,
        "--mode", "domain_train",
        "--target_file", str(job.cross_target_file.relative_to(REPO)),
        "--eval_type", "cross",
        "--jobid", jobid,
    ]

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# JOBID={jobid}  TAG={tag}\n")
        lf.write(f"# Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lf.flush()

        def _run(cmd, label):
            lf.write(f"\n=== [{label}] {' '.join(cmd)} ===\n")
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
            if job.already_done():
                logging.info("SKIP %s (already done)", job.tag)
            else:
                logging.info("START %s", job.tag)
                run_one(job)
        except Exception:
            logging.exception("Job crashed: %s", job.tag)
        finally:
            q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Run at most N jobs (for testing).")
    args = ap.parse_args()

    all_jobs = [j for j in build_jobs() if j.model in args.models]
    pending = [j for j in all_jobs if not j.already_done()]
    done = len(all_jobs) - len(pending)
    logging.info("Total=%d  Done=%d  Pending=%d", len(all_jobs), done, len(pending))

    if args.limit:
        pending = pending[: args.limit]
        logging.info("Limiting to first %d pending jobs.", args.limit)

    if args.dry_run:
        for j in pending:
            print(j.tag)
        return

    # Per-model queues + workers
    threads = []
    queues = {}
    for model in args.models:
        q: "queue.Queue[Job]" = queue.Queue()
        for j in pending:
            if j.model == model:
                q.put(j)
        queues[model] = q
        n_workers = PARALLELISM.get(model, 1)
        for i in range(n_workers):
            t = threading.Thread(target=worker_loop, args=(f"{model}-{i}", q), name=f"{model}-{i}")
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    logging.info("All workers finished.")


if __name__ == "__main__":
    main()

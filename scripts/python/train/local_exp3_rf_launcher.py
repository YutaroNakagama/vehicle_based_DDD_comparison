"""Local CPU launcher for RF full 12-seed exp3 distance-comparison run.

Purpose: provide RF (RandomForest) results across all three distance metrics
(mmd / wasserstein / dtw) under the exact exp3 SW-SMOTE domain_train setup, so
the best distance metric can be chosen before committing SvmA's remaining
(expensive, GPU-bound) dtw / wasserstein jobs.

RF is CPU-only and cheap, so this runs on Windows-native Python with several
parallel workers. It does NOT use the GPU and therefore runs concurrently with
the SvmA cuML GPU launcher without contention.

Run from PowerShell (repo root):
    python scripts/python/train/local_exp3_rf_launcher.py

Scope (mirrors local_exp3_svma_cuml_launcher.py):
  Model:    RF
  Distance: mmd, wasserstein, dtw
  Domain:   in_domain, out_domain
  Ratios:   0.3, 0.5
  Seeds:    12 (Phase 1 [42,123,2025] + Phase 2 [0,7,99,256,512,777,1000,1337,2024])
  => 3 x 2 x 2 x 12 = 144 jobs

Per-job: 1 training + 2 evaluations (within + cross). Skips jobs whose
within-domain eval JSON already exists (idempotent / resumable).

Worker count via env LOCAL_PARALLEL_RF (default 8).
"""
from __future__ import annotations

import argparse
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
LOG_DIR = REPO / "logs" / "exp3_rf"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "RF"
DISTANCES = ["mmd", "wasserstein", "dtw"]
DOMAINS = ["in_domain", "out_domain"]
RATIOS = [0.3, 0.5]
SEEDS = [0, 7, 42, 99, 123, 256, 512, 777, 1000, 1337, 2024, 2025]

N_WORKERS = int(os.environ.get("LOCAL_PARALLEL_RF", "8"))


@dataclass
class Job:
    distance: str
    domain: str
    ratio: float
    seed: int

    @property
    def tag(self) -> str:
        return (
            f"prior_{MODEL}_imbalv3_knn_{self.distance}_{self.domain}"
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
        return f"eval_results_{MODEL}_domain_train_{self.tag}_within.json"

    def already_done(self, done_set: set) -> bool:
        return self.eval_within_json_name in done_set


def build_done_set() -> set:
    base = EVAL_DIR / MODEL
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
    tag = job.tag
    log_path = LOG_DIR / f"{tag}.log"
    jobid = f"{int(time.time() * 1000)}{os.getpid()}"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PBS_JOBID"] = jobid
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # RF is CPU-only — keep off the GPU so it never contends with SvmA cuML.
    env["CUDA_VISIBLE_DEVICES"] = ""
    # Bound per-worker BLAS threads to avoid oversubscription across workers.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    train_cmd = [
        PYTHON, "scripts/python/train/train.py",
        "--model", MODEL,
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
        "--model", MODEL,
        "--tag", tag,
        "--mode", "domain_train",
        "--target_file", str(job.target_file.relative_to(REPO)),
        "--eval_type", "within",
        "--jobid", jobid,
    ]
    eval_cross_cmd = [
        PYTHON, "scripts/python/evaluation/evaluate.py",
        "--model", MODEL,
        "--tag", tag,
        "--mode", "domain_train",
        "--target_file", str(job.cross_target_file.relative_to(REPO)),
        "--eval_type", "cross",
        "--jobid", jobid,
    ]

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# JOBID={jobid}  TAG={tag}  CPU RF\n")
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
        lf.write(f"\n# Finished in {elapsed:.1f}s | CPU RF | rc1={rc1} rc2={rc2}\n")

    logging.info("DONE %s in %.1fs (rc1=%d rc2=%d)", tag, elapsed, rc1, rc2)
    return 0


def worker_loop(name: str, q: "queue.Queue[Job]", done_set: set) -> None:
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        try:
            if job.already_done(done_set):
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
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--tag-filter", type=str, default=None,
                    help="Substring match — only run jobs whose tag contains this string.")
    args = ap.parse_args()

    done_set = build_done_set()
    all_jobs = build_jobs()
    pending = [j for j in all_jobs if not j.already_done(done_set)]
    if args.tag_filter:
        pending = [j for j in pending if args.tag_filter in j.tag]
    if args.reverse:
        pending.reverse()
    if args.skip > 0:
        pending = pending[args.skip:]
    if args.limit:
        pending = pending[: args.limit]
    done = len(all_jobs) - len([j for j in all_jobs if not j.already_done(done_set)])
    logging.info(
        "RF | Total=%d Done=%d Selected=%d Workers=%d Reverse=%s Skip=%d Filter=%s",
        len(all_jobs), done, len(pending), N_WORKERS, args.reverse, args.skip, args.tag_filter,
    )

    if args.dry_run:
        for j in pending:
            print(j.tag)
        return

    q: "queue.Queue[Job]" = queue.Queue()
    for j in pending:
        q.put(j)

    threads = []
    for i in range(N_WORKERS):
        t = threading.Thread(target=worker_loop, args=(f"RF-{i}", q, done_set), name=f"RF-{i}")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    logging.info("All RF workers finished.")


if __name__ == "__main__":
    main()

"""B1 comparison launcher — prior methods under exp2's target_only framework.

Reproduces exp2's factor design (target_only / SW-SMOTE / within-domain /
wasserstein) for the prior methods SvmW, SvmA, Lstm (and RF as reference),
each using its OWN paper-faithful features + label + classifier (the pipeline
is model-aware; this launcher does NOT homogenise feature selection).

Scope (per user, 2026-06-24):
  Models:   RF, SvmW, SvmA, Lstm  (run one model per invocation via --model)
  Mode:     target_only           (exp2's within-domain training protocol)
  Distance: wasserstein           (exp2 Sobol: distance negligible -> narrowed)
  Domain:   in_domain             (within-domain)
  Imbalance: SW-SMOTE             (smote + --subject_wise_oversampling)
  Ratios:   0.3, 0.5
  Seeds:    42, 123, 2025
  => 2 x 3 = 6 cells per model.

Labels (paper-faithful, handled by the pipeline):
  RF/SvmW/SvmA -> KSS (SvmA uses its own 1-6/8-9 mapping); Lstm -> event_label.

Data:
  RF/SvmW/SvmA read data/processed/common (local, corrected kss labels).
  Lstm reads data/processed/Lstm (event_label + Wang features).

Backend (set by the CALLER via the python exe + env vars; this script uses
sys.executable and inherits the environment):
  RF/SvmW : Windows python, CPU  (CUDA_VISIBLE_DEVICES="")
  Lstm    : WSL2 .venv_tf_gpu,   GPU
  SvmA    : WSL2 .venv_svma_cuml, GPU (SVMA_USE_CUML=1, CUDA_VISIBLE_DEVICES=0)

Resume-safe: skips any cell whose within-domain eval JSON already exists.
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
RANK = REPO / "results" / "analysis" / "exp2_domain_shift" / "distance" / "rankings" / "split2" / "knn"
EVAL_DIR = REPO / "results" / "outputs" / "evaluation"
LOG_DIR = REPO / "logs" / "exp3_b1cmp"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DISTANCE = "wasserstein"
DOMAIN = "in_domain"
RATIOS = ["0.3", "0.5"]
# Per-method seed counts chosen by a convergence criterion (95% CI half-width):
#   RF/SvmW/Lstm -> 11 seeds (TIV2026's 10 + original 2025); reuses prior cells.
#   SvmA -> 6 seeds: it is robustly chance (~0.52), and 6 seeds give a 95% CI
#   of ~[0.48,0.56] (upper bound < 0.60), academically sufficient to conclude
#   "no benefit". (Saves ~3 GPU-days; SvmA is the compute bottleneck.) The
#   running-mean convergence plot (analysis/seed_convergence) justifies it.
SEEDS = [0, 1, 7, 13, 42, 123, 256, 512, 1337, 2024, 2025]
SEEDS_SVMA = [0, 1, 7, 42, 123, 2025]  # 6 (incl. the 3 already-run: 42,123,2025)

def _seeds(model: str):
    return SEEDS_SVMA if model == "SvmA" else SEEDS

DEFAULT_WORKERS = {"RF": 6, "SvmW": 6, "SvmA": 1, "Lstm": 4}


@dataclass
class Cell:
    model: str
    ratio: str
    seed: int

    @property
    def tag(self) -> str:
        return (
            f"b1cmp_{self.model}_target_only_knn_{DISTANCE}_{DOMAIN}"
            f"_swsmote_ratio{self.ratio}_s{self.seed}"
        )

    @property
    def target_file(self) -> Path:
        return RANK / f"{DISTANCE}_{DOMAIN}.txt"

    @property
    def within_json(self) -> str:
        return f"eval_results_{self.model}_target_only_{self.tag}_within.json"

    def already_done(self) -> bool:
        base = EVAL_DIR / self.model
        return base.exists() and any(base.rglob(self.within_json))


def build_cells(model: str) -> List[Cell]:
    return [Cell(model, r, s) for r in RATIOS for s in _seeds(model)]


def run_cell(cell: Cell) -> int:
    tag = cell.tag
    jobid = f"{int(time.time() * 1000)}{os.getpid()}{cell.seed}"
    log_path = LOG_DIR / f"{tag}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PBS_JOBID"] = jobid
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    tf = str(cell.target_file.relative_to(REPO))
    train_cmd = [
        PYTHON, "scripts/python/train/train.py",
        "--model", cell.model, "--mode", "target_only", "--seed", str(cell.seed),
        "--target_file", tf, "--tag", tag,
        "--time_stratify_labels", "--use_oversampling", "--oversample_method", "smote",
        "--target_ratio", cell.ratio, "--subject_wise_oversampling",
    ]
    eval_cmd = [
        PYTHON, "scripts/python/evaluation/evaluate.py",
        "--model", cell.model, "--tag", tag, "--mode", "target_only",
        "--target_file", tf, "--eval_type", "within", "--jobid", jobid,
    ]

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# JOBID={jobid} TAG={tag}\n# Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        lf.flush()

        def _run(cmd, label):
            lf.write(f"\n=== [{label}] {' '.join(cmd)} ===\n"); lf.flush()
            return subprocess.run(cmd, cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode

        rc = _run(train_cmd, "TRAIN")
        if rc != 0:
            logging.error("TRAIN FAILED %s rc=%d", tag, rc)
            return rc
        rc1 = _run(eval_cmd, "EVAL within")
        lf.write(f"\n# Finished in {time.time()-start:.1f}s rc1={rc1}\n")
    logging.info("DONE %s in %.1fs", tag, time.time() - start)
    return 0


def worker(name: str, q: "queue.Queue[Cell]") -> None:
    while True:
        try:
            cell = q.get_nowait()
        except queue.Empty:
            return
        try:
            if cell.already_done():
                logging.info("SKIP %s (done)", cell.tag)
            else:
                logging.info("START %s", cell.tag)
                run_cell(cell)
        except Exception:
            logging.exception("crash %s", cell.tag)
        finally:
            q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["RF", "SvmW", "SvmA", "Lstm"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = build_cells(args.model)
    pending = [c for c in cells if not c.already_done()]
    n_workers = args.workers or DEFAULT_WORKERS.get(args.model, 1)
    logging.info("B1 %s | python=%s | total=%d done=%d pending=%d workers=%d",
                 args.model, PYTHON, len(cells), len(cells) - len(pending), len(pending), n_workers)

    if args.dry_run:
        for c in pending:
            print(c.tag)
        return

    q: "queue.Queue[Cell]" = queue.Queue()
    for c in pending:
        q.put(c)
    threads = [threading.Thread(target=worker, args=(f"{args.model}-{i}", q), name=f"{args.model}-{i}")
               for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    logging.info("B1 %s: all workers finished.", args.model)


if __name__ == "__main__":
    main()

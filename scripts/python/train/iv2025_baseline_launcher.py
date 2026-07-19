"""IV2025-baseline launcher — reproduce the IV2025 setting (NO domain, NO imbalance).

IV2025 (nakagama2025iv) compared the four methods in a *pooled* split with NO
class-imbalance handling — the setting where RF dominated and the prior methods
collapsed to ~chance (C2/proper config: SvmA AUC 0.53, SvmW 51%, Lstm AUC 0.52).
Reproducing it locally lets us quantify the gain from adding domain + SW-SMOTE
(the B1/exp3 setting).

Config (mirrors scripts/hpc/jobs/train/pbs_array_svma_pooled.sh, baseline arm):
  Models: RF, SvmW, SvmA, Lstm  (one per invocation via --model)
  Mode:   pooled  (original IV2025 config: train --subject_wise_split, eval without;
          reproduced as-published to stay comparable to IV2025 / TIV2026)
  Imbalance: NONE (no --use_oversampling)
  Seeds:  42, 123, 2025  (match B1 for paired comparison)
  => 3 cells per model.

Each model keeps its OWN paper-faithful features + label + classifier (pipeline
is model-aware). Resume-safe. Distinct tag prefix iv25base_.
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s")

REPO = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
EVAL_DIR = REPO / "results" / "outputs" / "evaluation"
LOG_DIR = REPO / "logs" / "iv2025_base"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Academically-appropriate seed counts (convergence criterion, same as B1):
#   RF (discriminating "winner") already has 11 seeds completed.
#   SvmW/SvmA/Lstm are chance (~0.52) in the uncontrolled baseline -> 6 seeds
#   give a 95% CI whose upper bound stays < 0.60 (excludes weak signal),
#   justified by the running-mean convergence plot. 6 = TIV2026's first six.
SEEDS = [0, 1, 7, 42, 123, 2025]
# RF is high-variance (seed-adequacy: req_n=15 for 95% CI half-width <=0.05); chance
# methods (SvmW/Lstm/SvmA) are low-variance so 6 seeds already exceed adequacy.
SEEDS_RF = [0, 1, 7, 13, 42, 123, 256, 512, 1337, 2024, 2025, 3, 5, 9, 11]  # 15

def _seeds(model: str):
    return SEEDS_RF if model == "RF" else SEEDS

DEFAULT_WORKERS = {"RF": 3, "SvmW": 3, "SvmA": 1, "Lstm": 3}


@dataclass
class Cell:
    model: str
    seed: int
    smote: bool = False  # pooled + SW-SMOTE(0.5, subject-wise) arm (user request 2026-07-04)
    nofs: bool = False   # feature-selection-OFF arm (RF all-features, user 2026-07-11)

    @property
    def tag(self) -> str:
        fs = "_nofs" if self.nofs else ""
        if self.smote:
            return f"iv25smote_{self.model}_pooled_swsmote{fs}_s{self.seed}"
        return f"iv25base_{self.model}_pooled_baseline{fs}_s{self.seed}"

    def already_done(self) -> bool:
        base = EVAL_DIR / self.model
        if not base.exists():
            return False
        # EXACT name (no trailing '*': 's1*' would wrongly match s13/s123/s1337).
        return any(base.rglob(f"eval_results_{self.model}_pooled_{self.tag}.json"))


def build_cells(model: str, smote: bool = False, nofs: bool = False, seeds: List[int] = None) -> List[Cell]:
    use = seeds if seeds is not None else _seeds(model)
    return [Cell(model, s, smote, nofs) for s in use]


def run_cell(cell: Cell) -> int:
    tag = cell.tag
    jobid = f"{int(time.time() * 1000)}{os.getpid()}{cell.seed}"
    log_path = LOG_DIR / f"{tag}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PBS_JOBID"] = jobid
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # IV2025 baseline: reproduced with the ORIGINAL IV2025 config for comparability
    # (train --subject_wise_split; eval --mode pooled without the flag). This matches the
    # published IV2025 "before" values. (Kept as-published to respect IV2025/TIV2026.)
    train_cmd = [
        PYTHON, "scripts/python/train/train.py",
        "--model", cell.model, "--mode", "pooled", "--subject_wise_split",
        "--seed", str(cell.seed), "--tag", tag,
    ]
    if cell.smote:
        # Pooled + imbalance handling: SAME SW-SMOTE recipe as c1 (ratio 0.5,
        # subject-wise) so the "Pooled(handled)" column is directly comparable to
        # both the IV25 baseline (identical pooled train/eval protocol) and the
        # c1 Mixed/Within cells (identical oversampling method).
        train_cmd += ["--use_oversampling", "--oversample_method", "smote",
                      "--target_ratio", "0.5", "--subject_wise_oversampling"]
    if cell.nofs:  # feature-selection-off arm: keep all features (RF all-features ablation)
        train_cmd += ["--feature_selection", "none"]
    eval_cmd = [
        PYTHON, "scripts/python/evaluation/evaluate.py",
        "--model", cell.model, "--tag", tag, "--mode", "pooled", "--jobid", jobid,
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
        rc1 = _run(eval_cmd, "EVAL pooled")
        lf.write(f"\n# Finished in {time.time()-start:.1f}s rc1={rc1}\n")
    if rc1 != 0:
        # eval was interrupted/killed (e.g. resource starvation) BEFORE writing the result
        # JSON — the old code logged DONE regardless, masking this. Surface it; cell stays
        # pending (already_done is JSON-based) so it is retried on the next launch.
        logging.error("EVAL FAILED %s rc1=%d (no result JSON; stays pending)", tag, rc1)
        return rc1
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
    ap.add_argument("--limit", type=int, default=None,
                    help="Run at most N pending cells then exit (for round-robin GPU interleaving).")
    ap.add_argument("--smote", action="store_true",
                    help="Pooled + SW-SMOTE(0.5, subject-wise) arm (tag prefix iv25smote_).")
    ap.add_argument("--no-fs", dest="no_fs", action="store_true",
                    help="Feature-selection-OFF arm (keep all features; RF all-features ablation). Tags get _nofs.")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seed override (default: model's built-in seed set).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    cells = build_cells(args.model, args.smote, nofs=args.no_fs, seeds=seeds)
    pending = [c for c in cells if not c.already_done()]
    if args.limit:
        pending = pending[: args.limit]
    n_workers = args.workers or DEFAULT_WORKERS.get(args.model, 1)
    logging.info("IV25base %s | python=%s | total=%d done=%d pending=%d workers=%d",
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
    logging.info("IV25base %s: done.", args.model)


if __name__ == "__main__":
    main()

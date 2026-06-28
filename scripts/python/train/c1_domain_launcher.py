"""c1 domain-comparison launcher — Within vs Cross domain, in/out, 4 models.

Revised exp3 plan (advisor, 2026-06-27): for RF/SvmW/SvmA/Lstm, compare
Within-domain (target_only) and Cross-domain (source_only) on the in/out groups
to demonstrate RF's superiority (esp. cross-domain robustness).

Mirrors the exp2 precedent (scripts/hpc/jobs/domain_analysis/pbs_domain_comparison.sh):
  Train: train.py --model M --mode MODE --seed S --target_file <split2/knn>/<dist>_<DOMAIN>.txt
         --tag TAG --time_stratify_labels --use_oversampling --oversample_method smote
         --target_ratio 0.5 --subject_wise_oversampling
  Eval:  evaluate.py --model M --mode MODE --seed S --target_file <same> --tag TAG --subject_wise_split

Tag (exp2-compatible so source_only resolves the source group via split2/knn):
  imbalv3_knn_<dist>_<DOMAIN>_<MODE>_split2_subjectwise_ratio0.5_s<SEED>
  (must start with 'imbalv3_' and contain 'split2' -> target_resolution.py uses
   rankings/split2/knn/<dist>_<oppositeDOMAIN>.txt as the cross-domain source group.)

Conditions (within-IN is reused from B1's target_only/in_domain cells, identical
target_timewise eval -> not re-run here):
  (target_only, out_domain)  = Within-out
  (source_only, in_domain)   = Cross-in   (train out, eval in)
  (source_only, out_domain)  = Cross-out  (train in,  eval out)
SW-SMOTE fixed, wasserstein, ratio 0.5, seeds 42/123/2025.
=> 3 conditions x 4 models x 3 seeds = 36 cells.

Backends (set by caller python exe + env): RF/SvmW = Windows CPU; Lstm/SvmA = WSL2 GPU.
Resume-safe; --limit N for round-robin interleaving.
"""
from __future__ import annotations
import argparse, logging, os, queue, subprocess, sys, threading, time
from dataclasses import dataclass
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s")

REPO = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
RANK = REPO / "results" / "analysis" / "exp2_domain_shift" / "distance" / "rankings" / "split2" / "knn"
EVAL_DIR = REPO / "results" / "outputs" / "evaluation"
LOG_DIR = REPO / "logs" / "exp3_c1dom"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DISTANCE = "wasserstein"
RATIO = "0.5"
SEEDS = [42, 123, 2025]
# (mode, domain). The 6-case grid (advisor 2026-06-28) = {Within, Cross, Mixed} x {in,out}:
#   target_only = Within (train target domain), source_only = Cross (train other domain),
#   mixed = Mixed (train ALL subjects), all evaluated on the target domain (in/out).
# Within-in (target_only, in_domain) is reused from B1 for RF/SvmW/Lstm (features
# unaffected by the SvmA fix); SvmA ALSO re-runs Within-in here because the T1
# feature-faithfulness fix (SvmA.py, 2026-06-27) invalidates B1's SvmA cells.
CONDITIONS = [("target_only", "out_domain"),
              ("source_only", "in_domain"), ("source_only", "out_domain"),
              ("mixed", "in_domain"), ("mixed", "out_domain")]
WITHIN_IN = ("target_only", "in_domain")
DEFAULT_WORKERS = {"RF": 4, "SvmW": 4, "SvmA": 1, "Lstm": 3}


@dataclass
class Cell:
    model: str
    mode: str
    domain: str
    seed: int

    @property
    def tag(self) -> str:
        return f"imbalv3_knn_{DISTANCE}_{self.domain}_{self.mode}_split2_subjectwise_ratio{RATIO}_s{self.seed}"

    @property
    def target_file(self) -> Path:
        return RANK / f"{DISTANCE}_{self.domain}.txt"

    def already_done(self) -> bool:
        base = EVAL_DIR / self.model
        if not base.exists():
            return False
        # eval name = eval_results_<M>_<mode>_<tag>.json ; tag ends with _s<seed> (no
        # seed-prefix collision among 42/123/2025). Anchor on the tag to be safe.
        return any(base.rglob(f"*{self.tag}.json")) or any(base.rglob(f"*{self.tag}_*.json"))


def build_cells(model: str, seeds: List[int] = None) -> List[Cell]:
    conds = list(CONDITIONS)
    if model == "SvmA":  # T1 fix invalidates B1's SvmA Within-in -> re-run it here too
        conds = [WITHIN_IN] + conds
    return [Cell(model, m, d, s) for (m, d) in conds for s in (seeds or SEEDS)]


def run_cell(cell: Cell) -> int:
    tag = cell.tag
    jobid = f"{int(time.time() * 1000)}{os.getpid()}{cell.seed}"
    log_path = LOG_DIR / f"{cell.model}_{tag}.log"  # model in name: tag itself omits model
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO); env["PBS_JOBID"] = jobid; env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    tf = str(cell.target_file.relative_to(REPO))
    train_cmd = [PYTHON, "scripts/python/train/train.py", "--model", cell.model, "--mode", cell.mode,
                 "--seed", str(cell.seed), "--target_file", tf, "--tag", tag, "--time_stratify_labels",
                 "--use_oversampling", "--oversample_method", "smote", "--target_ratio", RATIO,
                 "--subject_wise_oversampling"]
    eval_cmd = [PYTHON, "scripts/python/evaluation/evaluate.py", "--model", cell.model, "--mode", cell.mode,
                "--seed", str(cell.seed), "--target_file", tf, "--tag", tag, "--subject_wise_split"]
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# JOBID={jobid} TAG={tag}\n# Started {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"); lf.flush()
        def _run(cmd, label):
            lf.write(f"\n=== [{label}] {' '.join(cmd)} ===\n"); lf.flush()
            return subprocess.run(cmd, cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        rc = _run(train_cmd, "TRAIN")
        if rc != 0:
            logging.error("TRAIN FAILED %s rc=%d", tag, rc); return rc
        rc1 = _run(eval_cmd, "EVAL")
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
                logging.info("START %s", cell.tag); run_cell(cell)
        except Exception:
            logging.exception("crash %s", cell.tag)
        finally:
            q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["RF", "SvmW", "SvmA", "Lstm"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Run at most N pending cells then exit.")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seed override (e.g. seed-increase run). Default: 42,123,2025.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    cells = build_cells(args.model, seeds)
    pending = [c for c in cells if not c.already_done()]
    if args.limit:
        pending = pending[: args.limit]
    n_workers = args.workers or DEFAULT_WORKERS.get(args.model, 1)
    logging.info("c1dom %s | total=%d done=%d pending=%d workers=%d",
                 args.model, len(cells), len(cells) - len([c for c in cells if not c.already_done()]),
                 len(pending), n_workers)
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
    logging.info("c1dom %s: done.", args.model)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys, os, argparse
from pathlib import Path
PRJ = Path(__file__).resolve().parents[1]
sys.path.append(str(PRJ))  # import src.***

from src.analysis.summary_groups import run_summarize_only10_vs_finetune

def main():
    ap = argparse.ArgumentParser(description="Summarize only10 vs finetune (6 groups)")
    ap.add_argument("--names_file", default=str(PRJ / "misc" / "pretrain_groups" / "group_names.txt"))
    ap.add_argument("--model_dir",  default=str(PRJ / "model" / "common"))
    ap.add_argument("--out_prefix", default="summary_6groups_only10_vs_finetune")
    ap.add_argument("--model", default="RF")
    ap.add_argument("--split", default="test")
    ap.add_argument("--make_radar", action="store_true")
    ap.add_argument("--only10_pattern",   default="metrics_{model}_only10_{group}.csv")
    ap.add_argument("--finetune_pattern", default="metrics_{model}_finetune_{group}_finetune.csv")
    args = ap.parse_args()

    run_summarize_only10_vs_finetune(
        names_file=Path(args.names_file),
        model_dir=Path(args.model_dir),
        out_prefix=args.out_prefix,
        model=args.model,
        split=args.split,
        make_radar=args.make_radar,
        only10_pattern=args.only10_pattern,
        finetune_pattern=args.finetune_pattern,
    )

if __name__ == "__main__":
    main()


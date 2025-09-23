#!/usr/bin/env python3
import glob
import os
import pandas as pd

def compare_metrics(tag):
    eval_file = f"model/common/metrics_RF_{tag}_evalonly_on_targets.csv"
    only_file = f"model/common/metrics_RF_{tag}_only_target.csv"
    if not (os.path.exists(eval_file) and os.path.exists(only_file)):
        print(f"[SKIP] Missing files for tag={tag}")
        return

    df_eval = pd.read_csv(eval_file, index_col=0)
    df_only = pd.read_csv(only_file, index_col=0)

    print("="*80)
    print(f"Tag: {tag}")
    print("EvalOnly_on_targets vs Only_target")
    for split in ["val", "test"]:
        if split in df_eval.index and split in df_only.index:
            row_eval = df_eval.loc[split]
            row_only = df_only.loc[split]
            print(f"--- {split.upper()} ---")
            for metric in ["accuracy", "precision", "recall", "f1", "auc", "ap"]:
                v1, v2 = row_eval[metric], row_only[metric]
                print(f"{metric:>10}: evalonly={v1:.3f}, only={v2:.3f}, diff={v2-v1:+.3f}")
    print()

if __name__ == "__main__":
    # evalonly ファイルを基準にタグ一覧を取得
    BASE = os.path.dirname(os.path.abspath(__file__))
    eval_files = glob.glob(os.path.join(BASE, "../model/common/metrics_RF_*_evalonly_on_targets.csv"))
    tags = [os.path.basename(f).replace("metrics_RF_", "").replace("_evalonly_on_targets.csv", "") for f in eval_files]

    if not tags:
        print("[WARN] No evalonly_on_targets files found")
    else:
        for tag in tags:
            compare_metrics(tag)


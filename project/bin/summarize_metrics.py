# bin/summarize_metrics.py
import re
import os
import glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
OUTDIR = os.path.join(ROOT, "model", "common")

# 対象ファイルの正規表現（必要に応じて追加）
PATTERNS = [
    # pretrainあり（finetune）
    (re.compile(r"metrics_RF_finetune_group(\d+)_finetune\.csv$"), "pretrainあり(finetune)"),
    # pretrainなし（only10）
    (re.compile(r"metrics_RF_only10_group(\d+)\.csv$"), "pretrainなし(only10)"),
    # 参考: ベースライン（全体学習など）。groupなし
    (re.compile(r"metrics_RF\.csv$"), "baseline"),
]

def parse_file(path):
    fname = os.path.basename(path)
    for pat, scheme in PATTERNS:
        m = pat.search(fname)
        if m:
            grp = int(m.group(1)) if m.groups() else None
            return scheme, grp
    return None, None

def safe_get(d, k, default=float("nan")):
    return d[k] if k in d else default

rows = []
for fp in glob.glob(os.path.join(OUTDIR, "metrics_*.csv")):
    scheme, group = parse_file(fp)
    if scheme is None:
        # 想定外のファイル名はスキップ
        continue
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        print(f"[WARN] read fail: {fp}: {e}")
        continue

    # split=='test' を優先。なければ一番下の行を使う安全策
    use = df[df["split"] == "test"] if "split" in df.columns else df
    if use.empty:
        use = df.tail(1)

    rec = use.iloc[-1].to_dict()
    rows.append({
        "group": group,
        "scheme": scheme,
        "accuracy": safe_get(rec, "accuracy"),
        "precision": safe_get(rec, "precision"),
        "recall": safe_get(rec, "recall"),
        "f1": safe_get(rec, "f1"),
        "auc": safe_get(rec, "auc"),
        "ap": safe_get(rec, "ap"),
        "source": os.path.basename(fp),
    })

summary = pd.DataFrame(rows)

# カラムがあるときだけソート（今回のエラー回避）
sort_keys = [c for c in ["group", "scheme"] if c in summary.columns]
if sort_keys:
    summary = summary.sort_values(sort_keys)

# 出力
out_csv = os.path.join(OUTDIR, "summary_only10_vs_finetune.csv")
summary.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
print(summary)

# bin/make_comparison_table.py
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
OUTDIR = os.path.join(ROOT, "model", "common")
SRC = os.path.join(OUTDIR, "summary_only10_vs_finetune.csv")

df = pd.read_csv(SRC)

# スキーム名の正規化
scheme_map = {
    "pretrainあり(finetune)": "finetune",
    "pretrainなし(only10)": "only10",
    "baseline": "baseline",
}
df["scheme"] = df["scheme"].map(lambda x: scheme_map.get(x, x))

# 比較対象（only10 vs finetune）のみ抽出
df2 = df[df["scheme"].isin(["only10", "finetune"])].copy()

metrics = ["accuracy", "precision", "recall", "f1", "auc", "ap"]

# group が float になっていたら int に
if "group" in df2.columns:
    try:
        df2["group"] = df2["group"].astype("Int64")
    except Exception:
        pass

# ピボット（行: group, 列: scheme, 値: 各指標）
wide = df2.pivot_table(index="group", columns="scheme", values=metrics, aggfunc="first")

# 列順と見栄え（metric_only10, metric_finetune, metric_delta の順）
cols = []
out = pd.DataFrame(index=wide.index)
for m in metrics:
    c_only = (m, "only10")
    c_fine = (m, "finetune")
    if c_only in wide.columns and c_fine in wide.columns:
        out[f"{m}_only10"]   = wide[c_only]
        out[f"{m}_finetune"] = wide[c_fine]
        out[f"{m}_delta"]    = wide[c_fine] - wide[c_only]  # finetune - only10
    else:
        # 片方しかない場合も落ちないように
        if c_only in wide.columns:
            out[f"{m}_only10"] = wide[c_only]
        if c_fine in wide.columns:
            out[f"{m}_finetune"] = wide[c_fine]

# 丸め（見やすさ用）
out = out.round(3)
out = out.sort_index()

# CSV と Markdown を保存
csv_path = os.path.join(OUTDIR, "table_only10_vs_finetune_wide.csv")
md_path  = os.path.join(OUTDIR, "table_only10_vs_finetune_wide.md")

out.to_csv(csv_path)
#out.reset_index().to_markdown(md_path, index=False)

print(f"Saved CSV: {csv_path}")
print(f"Saved MD : {md_path}")
print(out)


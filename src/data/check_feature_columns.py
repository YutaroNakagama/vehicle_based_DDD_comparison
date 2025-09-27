# misc/check_feature_columns.py
import pandas as pd
import glob, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
DATA_DIR = os.path.join(ROOT, "data/processed/common")
RANK_LIST = os.path.join(ROOT, "misc/pretrain_groups/rank_names.txt")

# IDを正規化（余分な記号除去・大文字化）
def normalize_id(tok: str) -> str:
    tok = tok.strip().strip(",;")
    return tok

# グループファイル群からターゲットIDを収集
target_ids = []
with open(RANK_LIST) as f:
    for line in f:
        path = line.strip()
        if not path:
            continue
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        if not os.path.isfile(path):
            print(f"[WARN] group file not found: {path}")
            continue
        with open(path) as g:
            for raw in g:
                if not raw.strip():
                    continue
                # 空白区切りでも改行でも拾う
                for tok in raw.strip().split():
                    tid = normalize_id(tok)
                    if tid: 
                        target_ids.append(tid)

# 重複除去
target_ids = sorted(set(target_ids))

# CSVパスを組み立て
def id_to_csv(id_):
    return os.path.join(DATA_DIR, f"processed_{id_}.csv")

target_files = []
missing_target = []
for tid in target_ids:
    fp = id_to_csv(tid)
    if os.path.isfile(fp):
        target_files.append(fp)
    else:
        missing_target.append((tid, fp))

# すべてのCSV（S*.csv）からターゲットを除外して「一般（事前学習）側」ファイル集合を作る
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "processed_S*.csv")))
target_set = set(os.path.basename(p) for p in target_files)
general_files = [p for p in all_files if os.path.basename(p) not in target_set]

def cols(files):
    s = set()
    for f in files:
        try:
            s.update(pd.read_csv(f, nrows=0).columns)
        except Exception as e:
            print(f"[WARN] cannot read header: {f} ({e})")
    return s

gen_cols = cols(general_files)
tgt_cols = cols(target_files)

print("=== Summary ===")
print(f"#all_csv         : {len(all_files)}")
print(f"#target_ids      : {len(target_ids)}")
print(f"#target_files    : {len(target_files)} (missing {len(missing_target)})")
print(f"#general_files   : {len(general_files)}")
if missing_target:
    print("Missing target CSVs (ID -> expected path):")
    for tid, fp in missing_target[:10]:
        print("  ", tid, "->", fp)
    if len(missing_target) > 10:
        print(f"  ... and {len(missing_target)-10} more")

only_in_general = sorted(gen_cols - tgt_cols)
only_in_target  = sorted(tgt_cols - gen_cols)

print("\n=== Column diff ===")
print("Only in general:", only_in_general)
print("Only in target :", only_in_target)

# 終了コード：差分があれば1を返すようにしてCI等でも使いやすく
if only_in_general or only_in_target or missing_target:
    sys.exit(1)


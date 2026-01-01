# misc/check_feature_columns.py
import pandas as pd
import glob, os, re, sys

from src import config as cfg

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
DATA_DIR = os.path.join(ROOT, cfg.PROCESS_CSV_COMMON_PATH)
RANK_LIST = os.path.join(ROOT, "misc/pretrain_groups/rank_names.txt")

# Normalize subject ID (remove extra symbols, uppercase)
def normalize_id(tok: str) -> str:
    tok = tok.strip().strip(",;")
    return tok

# Collect target subject IDs from group files
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
                # Handle whitespace or newline delimiters
                for tok in raw.strip().split():
                    tid = normalize_id(tok)
                    if tid: 
                        target_ids.append(tid)

# Remove duplicatesve duplicates
target_ids = sorted(set(target_ids))

# Build CSV file path from subject ID
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

# Create general (pretrain) file set by excluding targets from all CSVs (S*.csv)
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

# Exit code: return 1 if differences found (useful for CI/CD)
if only_in_general or only_in_target or missing_target:
    sys.exit(1)


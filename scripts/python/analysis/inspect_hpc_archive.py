#!/usr/bin/env python3
"""Inspect HPC exp3_snapshot archive: count by condition and sample JSON content."""
import json
import sys
import tarfile
from collections import Counter, defaultdict

try:
    import zstandard
except ImportError:
    print("ERROR: zstandard not installed. Run: pip install zstandard")
    sys.exit(1)

ARCHIVE = "/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison/results/outputs/hpc/exp3_snapshot_2026-05-19.tar.zst"
MODELS = ["Lstm", "SvmW", "SvmA"]

# Condition detection
def detect_condition(name):
    if "prior_Lstm_baseline" in name or "prior_SvmW_baseline" in name or "prior_SvmA_baseline" in name:
        return "baseline"
    if "imbalv3" in name:
        if "ratio0.1" in name:
            return "imbalv3_r0.1"
        if "ratio0.3" in name:
            return "imbalv3_r0.3"
        if "ratio0.5" in name:
            return "imbalv3_r0.5"
        return "imbalv3_other"
    if "smote_plain" in name:
        return "smote_plain"
    if "undersample" in name:
        return "undersample_rus"
    return "other"

def detect_distance(name):
    for d in ("mmd", "dtw", "wasserstein"):
        if f"_knn_{d}_" in name:
            return d
    return "unknown"

def detect_domain(name):
    if "in_domain" in name:
        return "in_domain"
    if "out_domain" in name:
        return "out_domain"
    return "unknown"

# Counts
counts = {m: Counter() for m in MODELS}
seed_sets = {m: defaultdict(set) for m in MODELS}
sample_jsons = {}  # model -> {condition -> json_content}

print(f"Reading archive: {ARCHIVE}")
dctx = zstandard.ZstdDecompressor()
with open(ARCHIVE, "rb") as fh:
    with dctx.stream_reader(fh) as reader:
        tf = tarfile.open(fileobj=reader, mode="r|")
        for member in tf:
            name = member.name
            if not name.endswith("_within.json"):
                continue
            for model in MODELS:
                if f"evaluation/{model}/" not in name:
                    continue
                cond = detect_condition(name)
                dist = detect_distance(name)
                dom = detect_domain(name)
                key = f"{cond}|{dist}|{dom}"
                counts[model][key] += 1

                # Seed extraction
                import re
                m = re.search(r"_s(\d+)_within", name)
                if m:
                    seed_sets[model][cond].add(m.group(1))

                # Sample JSON for new/interesting conditions
                sample_key = f"{model}:{cond}"
                if sample_key not in sample_jsons and member.size > 0:
                    f = tf.extractfile(member)
                    if f:
                        try:
                            d = json.load(f)
                            sample_jsons[sample_key] = {
                                "path": name,
                                "keys": list(d.keys()),
                                "roc_auc": d.get("roc_auc"),
                                "auc_pr": d.get("auc_pr"),
                                "f2_thr": d.get("f2_thr"),
                                "recall": d.get("recall"),
                                "precision": d.get("precision"),
                                "timestamp": d.get("timestamp", ""),
                            }
                        except Exception:
                            pass

print("\n=== Within JSON counts (per model, condition|distance|domain) ===")
for model in MODELS:
    total = sum(counts[model].values())
    print(f"\n--- {model} (total: {total}) ---")
    for key in sorted(counts[model]):
        print(f"  {key:<40s}: {counts[model][key]}")

print("\n=== Seeds per condition ===")
for model in MODELS:
    print(f"\n{model}:")
    for cond, seeds in sorted(seed_sets[model].items()):
        print(f"  {cond:<25s}: {len(seeds)} seeds -> {sorted(seeds, key=int)}")

print("\n=== Sample JSON fields (first hit per model:condition) ===")
for key, info in sorted(sample_jsons.items()):
    print(f"\n[{key}]")
    print(f"  path      : {info['path'][-80:]}")
    print(f"  keys      : {info['keys'][:15]}")
    print(f"  roc_auc   : {info['roc_auc']}")
    print(f"  auc_pr    : {info['auc_pr']}")
    print(f"  f2_thr    : {info['f2_thr']}")
    print(f"  recall    : {info['recall']}")
    print(f"  precision : {info['precision']}")
    print(f"  timestamp : {info['timestamp']}")

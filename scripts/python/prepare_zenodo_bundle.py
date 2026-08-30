#!/usr/bin/env python3
"""Build the Zenodo publication bundle from an explicit allowlist.

Why this exists
---------------
``.gitignore`` protects nothing when files are collected manually: a naive
``zip -r`` of the working tree would publish ~115 GB of model binaries, the
execution logs, both virtualenvs, and the confidential write-ups under
``docs/experiments/results/``. This script therefore never walks the repository
as a whole. It collects ONLY the paths named in :data:`ALLOW`, then re-checks
every collected path against :data:`DENY_*` before a single byte is written.

Safety model (both layers must pass)
------------------------------------
1. **Allowlist** — only the trees in :data:`ALLOW` are ever walked.
2. **Denylist** — the final staged path list is re-screened; any hit aborts.

Plus: byte-level verification of the deduplication assumption, a PII / local-path
scan, symlink refusal, a clean-worktree check, and a SHA-256 manifest.

Nothing is ever written inside the repository working tree.

Usage
-----
    python scripts/python/prepare_zenodo_bundle.py --dry-run
    python scripts/python/prepare_zenodo_bundle.py --format zip
    python scripts/python/prepare_zenodo_bundle.py --only results/analysis --format zip

Exit codes
----------
    0 success           3 deduplication assumption violated
    1 usage / config     4 PII scan hit
    2 denylist hit       5 dirty working tree
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

TOOL_VERSION = "1.0.0"
BUNDLE_VERSION = "1.0.0"

# --------------------------------------------------------------------------
# LAYER 1 — the allowlist. Nothing outside these trees is ever walked.
# --------------------------------------------------------------------------
# (repo-relative path, description, include-globs or None for "every file")
ALLOW: list[tuple[str, str, tuple[str, ...] | None]] = [
    ("results/outputs/training",   "Per-run training records (JSON + flattened CSV)", ("*.json", "*.csv")),
    ("results/outputs/evaluation", "Per-run evaluation records (JSON + flattened CSV)", ("*.json", "*.csv")),
    ("results/analysis",           "Aggregated tables, Sobol indices, journal figures, subject rankings", None),
    ("data/processed/common",      "Canonical per-subject-session feature tables (210 columns)", ("*.csv",)),
    ("data/processed/Lstm",        "Per-subject-session feature tables, LSTM windowing", ("*.csv",)),
    # Small provenance files that must travel with the data.
    ("DATADICTIONARY.md",          "Data dictionary", None),
    ("LICENSE",                    "Dual licence (MIT code / CC BY 4.0 data)", None),
    ("CITATION.cff",               "Machine-readable citation", None),
]

# Deduplicated away: byte-identical to data/processed/common (re-verified below).
DEDUP_AGAINST_COMMON = ("data/processed/SvmA", "data/processed/SvmW")

# --------------------------------------------------------------------------
# LAYER 2 — the denylist. Screened against the FINAL staged path list.
# --------------------------------------------------------------------------
DENY_PREFIXES: tuple[str, ...] = (
    "models/",                              # ~115 GB of pickles / Keras checkpoints
    "logs/",
    "scripts/hpc/logs/",
    "docs/",                                # confidential write-ups + IEEE class files
    "data/interim/",
    "data/processed/svma/",                 # dedup duplicates
    "data/processed/svmw/",
    "data/processed/_hpc_common_pull/",     # HPC-derived: reference-only
    "data/processed/common_fixedlabels_bak/",   # canonicity undetermined
    "results/outputs/hpc/",                 # HPC-derived: reference-only
    # Local re-runs that reproduce specific HPC jobs (tags like `repro_hpc88c`).
    # They are local output and would pass the allowlist, but "HPC results are
    # reference-only" under project policy, so shipping reproductions of them
    # invites a reviewer to read them as reported results. Matching is done on
    # the lowercased path, hence `rf` rather than `RF`.
    "results/outputs/evaluation/rf/reprojob",
    "results/outputs/training/rf/reprojob",
    "results/_archived",
    # results/analysis/exp1/ is NOT analysis data: it is a complete manuscript
    # source tree (IV2025 preprint) that also carries confidential peer-review
    # comments and an iThenticate originality report. It sits inside an allowlisted
    # tree, which is precisely why this second layer exists.
    "results/analysis/exp1/",
    ".git/",
    ".github/",
    ".venv",
    ".claude/",
    "memory/",
    "echo/",
    "tests/",
    "src/",
    "scripts/",                             # code ships via the GitHub release, not here
)

DENY_SEGMENTS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".ipynb_checkpoints", ".pytest_cache",
    "node_modules", "hpc", "_hpc_common_pull", "common_fixedlabels_bak",
    "review",           # peer-review correspondence must never be republished
})

DENY_GLOBS: tuple[str, ...] = (
    "*.pkl", "*.joblib", "*.keras", "*.h5", "*.hdf5", "*.pt", "*.pth", "*.ckpt",
    "*.pb", "*.onnx",                       # model artefacts of any flavour
    "*.log", "*.bak", "*.tmp", "*.swp", "*.pyc", "*.pyo",
    "*.zst", "*.tar.gz", "*.tgz", "*.7z", "*.rar",   # nested archives (hpc snapshots)
    "*.env", "*.pem", "*.key", "id_rsa*", "*.p12",
    ".DS_Store", "Thumbs.db", "desktop.ini",
    # Manuscript sources and third-party LaTeX machinery: not data, and IEEE's
    # class/style files may not be redistributed under this repository's licence.
    "*.tex", "*.cls", "*.bst", "*.bbl", "*.blg", "*.aux", "*.pptx", "*.docx",
    "*ithenticate*", "review*.txt", "*commentstoauthor*",
)

# Allow-list exception: DENY_PREFIXES bans "scripts/" and "src/" wholesale, but the
# three top-level provenance files must survive the screen.
DENY_EXEMPT_EXACT: frozenset[str] = frozenset({
    "DATADICTIONARY.md", "LICENSE", "CITATION.cff",
})

# --------------------------------------------------------------------------
# PII / local-path scan
# --------------------------------------------------------------------------
PII_PATTERNS: tuple[tuple[str, str], ...] = (
    ("local username",   r"ynakagama|nkgm1003"),
    ("Windows home path", r"[A-Za-z]:[\\/]+Users[\\/]+"),
    ("POSIX home path",  r"/home/[A-Za-z0-9_.-]+/"),
    ("cluster hostname", r"kagayaki|spcc-adm"),
    ("e-mail address",   r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:com|jp|org|net|edu)"),
    ("credential-like",  r"(?i)\b(?:api[_-]?key|password|passwd|secret|access[_-]?token)\b\s*[:=]"),
)
PII_TEXT_SUFFIXES = frozenset({".json", ".csv", ".md", ".txt", ".cff", ".yaml", ".yml", ".tex"})
PII_SCAN_BYTES = 1 << 20  # first 1 MiB of each text file; reported, never silent


# ==========================================================================
# Helpers
# ==========================================================================
def fail(code: int, message: str) -> "NoReturn":  # type: ignore[valid-type]
    print(f"\n[ABORT] {message}", file=sys.stderr)
    sys.exit(code)


def find_repo_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).resolve()
    else:
        root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        fail(1, f"{root} is not a git repository (no .git). Pass --repo explicitly.")
    return root


def git(root: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, check=False
        )
        return out.stdout.strip()
    except FileNotFoundError:
        return ""


def rel_posix(path: Path, root: Path) -> str:
    return PurePosixPath(path.relative_to(root).as_posix()).as_posix()


def is_denied(relpath: str) -> str | None:
    """Return the denylist rule that rejects *relpath*, or None if it passes."""
    if relpath in DENY_EXEMPT_EXACT:
        return None
    low = relpath.lower()
    parts = low.split("/")

    if any(not ch.isprintable() or ord(ch) > 126 for ch in relpath):
        return "non-ASCII/non-printable filename"
    if ".." in parts or relpath.startswith("/") or re.match(r"^[A-Za-z]:", relpath):
        return "absolute path or parent traversal"
    for prefix in DENY_PREFIXES:
        if low.startswith(prefix):
            return f"prefix {prefix!r}"
    for seg in parts[:-1]:
        if seg in DENY_SEGMENTS:
            return f"path segment {seg!r}"
    name = parts[-1]
    for pattern in DENY_GLOBS:
        if fnmatch.fnmatch(name, pattern.lower()):
            return f"glob {pattern!r}"
    return None


def sha256_of(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def human(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:,.1f} {unit}"
        x /= 1024
    return f"{n} B"


# ==========================================================================
# Collection
# ==========================================================================
def collect(root: Path, only: str | None) -> tuple[list[tuple[Path, str]], list[str]]:
    """Walk the allowlist. Returns (files, warnings). Symlinks are refused."""
    files: list[tuple[Path, str]] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for entry, _desc, globs in ALLOW:
        if only and not entry.startswith(only) and not entry in DENY_EXEMPT_EXACT:
            continue
        src = root / entry
        if not src.exists():
            warnings.append(f"allowlist entry missing, skipped: {entry}")
            continue
        if src.is_symlink():
            warnings.append(f"allowlist entry is a symlink, REFUSED: {entry}")
            continue

        if src.is_file():
            rp = rel_posix(src, root)
            if rp not in seen:
                seen.add(rp)
                files.append((src, rp))
            continue

        for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
            dpath = Path(dirpath)
            # Prune denied and symlinked directories before descending.
            keep = []
            for d in dirnames:
                drel = rel_posix(dpath / d, root).lower() + "/"
                if d.lower() in DENY_SEGMENTS:
                    warnings.append(f"pruned denied directory: {rel_posix(dpath / d, root)}")
                elif any(drel.startswith(p) for p in DENY_PREFIXES):
                    warnings.append(f"pruned denied directory: {rel_posix(dpath / d, root)}")
                elif (dpath / d).is_symlink():
                    warnings.append(f"pruned symlinked directory: {rel_posix(dpath / d, root)}")
                else:
                    keep.append(d)
            dirnames[:] = keep

            for fn in filenames:
                fpath = dpath / fn
                if fpath.is_symlink():
                    warnings.append(f"skipped symlink: {rel_posix(fpath, root)}")
                    continue
                if globs and not any(fnmatch.fnmatch(fn, g) for g in globs):
                    continue
                rp = rel_posix(fpath, root)
                # Apply the denylist here too, so that the screen in step 2 is a
                # genuine independent re-check that must always come back empty.
                if (rule := is_denied(rp)) is not None:
                    warnings.append(f"excluded by denylist ({rule}): {rp}")
                    continue
                if rp not in seen:
                    seen.add(rp)
                    files.append((fpath, rp))

    files.sort(key=lambda t: t[1])
    return files, warnings


def verify_dedup(root: Path) -> list[str]:
    """Re-verify that the deduplicated trees really are byte-identical to common/.

    The assumption may go stale; refusing to guess is the whole point.
    """
    notes: list[str] = []
    base = root / "data/processed/common"
    if not base.is_dir():
        return ["data/processed/common missing — dedup check skipped"]

    base_hashes = {p.name: sha256_of(p)[0] for p in sorted(base.glob("*.csv"))}
    for other in DEDUP_AGAINST_COMMON:
        odir = root / other
        if not odir.is_dir():
            notes.append(f"{other}: absent, nothing to deduplicate")
            continue
        other_hashes = {p.name: sha256_of(p)[0] for p in sorted(odir.glob("*.csv"))}
        if other_hashes != base_hashes:
            missing = set(base_hashes) ^ set(other_hashes)
            differing = [k for k in set(base_hashes) & set(other_hashes)
                         if base_hashes[k] != other_hashes[k]]
            fail(3, (
                f"{other} is NOT byte-identical to data/processed/common — "
                f"{len(differing)} differing file(s), {len(missing)} name mismatch(es). "
                "The deduplication assumption is stale: add this tree to ALLOW or "
                "resolve the divergence before publishing."
            ))
        saved = sum(p.stat().st_size for p in odir.glob("*.csv"))
        notes.append(f"{other}: verified identical to common/ ({len(other_hashes)} files, {human(saved)} saved)")
    return notes


def scan_pii(files: list[tuple[Path, str]]) -> list[tuple[str, str, str]]:
    """Return (relpath, pattern-name, redacted excerpt) for each hit."""
    hits: list[tuple[str, str, str]] = []
    compiled = [(name, re.compile(pat)) for name, pat in PII_PATTERNS]
    for src, rp in files:
        if Path(rp).suffix.lower() not in PII_TEXT_SUFFIXES:
            continue
        try:
            with open(src, "rb") as fh:
                blob = fh.read(PII_SCAN_BYTES).decode("utf-8", errors="replace")
        except OSError:
            continue
        for name, rx in compiled:
            m = rx.search(blob)
            if m:
                start = max(0, m.start() - 30)
                excerpt = blob[start:m.end() + 30].replace("\n", " ")
                hits.append((rp, name, excerpt[:120]))
                break
    return hits


# ==========================================================================
# Archive writing
# ==========================================================================
def build_manifest(files: list[tuple[Path, str]], verbose: bool) -> tuple[str, int, dict]:
    lines: list[str] = []
    total = 0
    per_tree: dict[str, dict] = {}
    for i, (src, rp) in enumerate(files, 1):
        digest, size = sha256_of(src)
        lines.append(f"{digest}  {rp}")
        total += size
        tree = "/".join(rp.split("/")[:3]) if "/" in rp else "(root)"
        slot = per_tree.setdefault(tree, {"files": 0, "bytes": 0})
        slot["files"] += 1
        slot["bytes"] += size
        if verbose and i % 500 == 0:
            print(f"    hashed {i:,}/{len(files):,} files ({human(total)})", flush=True)
    return "\n".join(lines) + "\n", total, per_tree


def open_archive(out_path: Path, fmt: str):
    """Return (writer, add_bytes_fn, close_fn) for the requested format."""
    if fmt == "zip":
        zf = zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True)
        return (
            zf,
            lambda arcname, src=None, data=None: (
                zf.write(src, arcname) if src is not None else zf.writestr(arcname, data)
            ),
            zf.close,
        )

    if fmt == "tar.gz":
        tf = tarfile.open(out_path, "w:gz", compresslevel=6)
    elif fmt == "tar.zst":
        try:
            import zstandard  # type: ignore
        except ImportError:
            zstd_cli = _which("zstd")
            if not zstd_cli:
                fail(1, (
                    "--format tar.zst needs either the 'zstandard' Python package "
                    "(pip install zstandard) or the zstd CLI on PATH. Python's stdlib "
                    "has no zstd before 3.14. Use --format zip or --format tar.gz instead. "
                    "(Not falling back silently: the output format must be what you asked for.)"
                ))
            proc = subprocess.Popen([zstd_cli, "-19", "-T0", "-f", "-o", str(out_path), "-"],
                                    stdin=subprocess.PIPE)
            tf = tarfile.open(fileobj=proc.stdin, mode="w|")

            def _close_cli():
                tf.close()
                proc.stdin.close()
                if proc.wait() != 0:
                    fail(1, "zstd CLI failed while compressing the bundle")

            return tf, _tar_adder(tf), _close_cli
        else:
            fh = open(out_path, "wb")
            writer = zstandard.ZstdCompressor(level=19).stream_writer(fh)
            tf = tarfile.open(fileobj=writer, mode="w|")

            def _close_mod():
                tf.close()
                writer.close()
                fh.close()

            return tf, _tar_adder(tf), _close_mod
    else:
        fail(1, f"unknown --format {fmt!r}")

    return tf, _tar_adder(tf), tf.close


def _tar_adder(tf: tarfile.TarFile):
    def add(arcname: str, src: Path | None = None, data: bytes | None = None) -> None:
        if src is not None:
            tf.add(str(src), arcname=arcname, recursive=False)
        else:
            info = tarfile.TarInfo(arcname)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return add


def _which(name: str) -> str | None:
    from shutil import which
    return which(name)


# ==========================================================================
# Main
# ==========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the Zenodo publication bundle from an explicit allowlist.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--repo", help="Repository root (default: inferred from this script's location)")
    ap.add_argument("--out", help="Output directory (default: <repo>/../zenodo_bundle). "
                                  "Must be OUTSIDE the repository.")
    ap.add_argument("--format", choices=["zip", "tar.gz", "tar.zst"], default="zip",
                    help="Archive format (default: zip — widest reviewer compatibility)")
    ap.add_argument("--version", default=BUNDLE_VERSION, help="Bundle version string")
    ap.add_argument("--only", help="Restrict to allowlist entries under this prefix (for testing)")
    ap.add_argument("--dry-run", action="store_true", help="Report the plan; write nothing")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="Proceed even though tracked files are modified (NOT recommended: "
                         "the bundle could not then be tied to a commit)")
    ap.add_argument("--allow-pii", action="store_true",
                    help="Proceed despite PII/local-path scan hits (review them first)")
    ap.add_argument("--include-fixedlabels-bak", action="store_true",
                    help="Also ship data/processed/common_fixedlabels_bak/ (see DATADICTIONARY §4.2)")
    args = ap.parse_args()

    root = find_repo_root(args.repo)
    print(f"Repository : {root}")
    print(f"Bundle ver : {args.version}   format: {args.format}")

    # -- git provenance -----------------------------------------------------
    commit = git(root, "rev-parse", "HEAD")
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    describe = git(root, "describe", "--always", "--tags")
    dirty_tracked = git(root, "status", "--porcelain", "--untracked-files=no")
    print(f"Commit     : {commit or '(unknown)'}  [{branch}] {describe}")

    if dirty_tracked:
        n = len(dirty_tracked.splitlines())
        msg = (f"{n} tracked file(s) are modified. A bundle built from a dirty tree "
               f"cannot be tied to the published code version.")
        if not args.allow_dirty:
            fail(5, msg + " Commit or stash first, or pass --allow-dirty.")
        print(f"[WARN] {msg} Proceeding because --allow-dirty was given.")

    if args.include_fixedlabels_bak:
        # Opt-in only, and it must relax BOTH denylist layers or the screen would
        # reject what the allowlist just collected.
        ALLOW.append(("data/processed/common_fixedlabels_bak",
                      "Alternative label set (canonicity undetermined)", ("*.csv",)))
        globals()["DENY_PREFIXES"] = tuple(
            p for p in DENY_PREFIXES if p != "data/processed/common_fixedlabels_bak/"
        )
        globals()["DENY_SEGMENTS"] = frozenset(DENY_SEGMENTS - {"common_fixedlabels_bak"})
        print("[WARN] common_fixedlabels_bak/ INCLUDED. Its canonicity is undetermined; "
              "document which label set is authoritative before publishing.")

    # -- LAYER 1: collect ---------------------------------------------------
    print("\n[1/6] Collecting from the allowlist ...")
    files, warnings = collect(root, args.only)
    for w in warnings[:12]:
        print(f"    note: {w}")
    if len(warnings) > 12:
        # Never truncate silently: a hidden exclusion reads as "we shipped everything".
        print(f"    note: ... and {len(warnings) - 12} further exclusion notes "
              f"(all are recorded as denylist/symlink exclusions)")
    if not files:
        fail(1, "allowlist produced zero files — check --repo / --only")
    print(f"    {len(files):,} files collected")

    # -- LAYER 2: screen ----------------------------------------------------
    print("[2/6] Screening every staged path against the denylist ...")
    denied = [(rp, rule) for _s, rp in files if (rule := is_denied(rp))]
    if denied:
        print(f"\n    {len(denied)} DENIED path(s):", file=sys.stderr)
        for rp, rule in denied[:20]:
            print(f"      {rp}   <- {rule}", file=sys.stderr)
        fail(2, "denylist violation — nothing was written. Fix ALLOW or investigate.")
    print(f"    clean: 0 denied paths out of {len(files):,}")

    # -- dedup --------------------------------------------------------------
    print("[3/6] Verifying the deduplication assumption ...")
    if args.only:
        print("    skipped (--only restricts the build)")
    else:
        for note in verify_dedup(root):
            print(f"    {note}")

    # -- PII ----------------------------------------------------------------
    print(f"[4/6] Scanning text files for PII / local paths (first {human(PII_SCAN_BYTES)} of each) ...")
    hits = scan_pii(files)
    if hits:
        print(f"    {len(hits)} file(s) matched:", file=sys.stderr)
        for rp, name, excerpt in hits[:10]:
            print(f"      {rp}  [{name}]  ...{excerpt}...", file=sys.stderr)
        if not args.allow_pii:
            fail(4, "PII/local-path scan hit — review the matches, then re-run with "
                    "--allow-pii if they are false positives.")
        print("    [WARN] proceeding despite hits because --allow-pii was given")
    else:
        print("    clean: no matches")

    # -- manifest -----------------------------------------------------------
    print("[5/6] Hashing files and building MANIFEST.sha256 ...")
    manifest, total_bytes, per_tree = build_manifest(files, verbose=True)
    print(f"    {len(files):,} files, {human(total_bytes)} raw")
    for tree, s in sorted(per_tree.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"      {tree:<34} {s['files']:>7,} files  {human(s['bytes']):>12}")

    info = {
        "bundle_version": args.version,
        "tool": "scripts/python/prepare_zenodo_bundle.py",
        "tool_version": TOOL_VERSION,
        "archive_format": args.format,
        "git": {
            "commit": commit,
            "branch": branch,
            "describe": describe,
            "tracked_tree_clean": not bool(dirty_tracked),
        },
        "contents": {
            "file_count": len(files),
            "total_bytes": total_bytes,
            "by_tree": per_tree,
        },
        "allowlist": [{"path": p, "description": d, "globs": list(g) if g else "all"}
                      for p, d, g in ALLOW],
        "denylist": {
            "prefixes": list(DENY_PREFIXES),
            "segments": sorted(DENY_SEGMENTS),
            "globs": list(DENY_GLOBS),
        },
        "excluded_by_policy": {
            "models/": "~115 GB of pickled/Keras model binaries; version-fragile and unsafe to unpickle",
            "results/outputs/hpc/": "HPC artefacts are reference-only, not reported results",
            "data/processed/SvmA|SvmW": "byte-identical duplicates of data/processed/common",
            "data/processed/_hpc_common_pull/": "HPC-derived",
            "data/processed/common_fixedlabels_bak/": "label-set canonicity undetermined",
            "logs/, docs/experiments/results/": "execution logs and internal write-ups",
            "results/analysis/exp1/": (
                "IV2025 manuscript source tree, including confidential peer-review "
                "correspondence, an iThenticate originality report, and IEEE LaTeX "
                "class files that may not be redistributed"
            ),
        },
        "notes": (
            "Paths in this file are repository-relative by construction; no absolute "
            "path, username or hostname from the build machine is recorded."
        ),
    }

    # -- write --------------------------------------------------------------
    stem = f"ddd-comparison-bundle-v{args.version}"
    if args.dry_run:
        print(f"\n[6/6] DRY RUN — nothing written. Would produce {stem}.{args.format}")
        print(f"       raw {human(total_bytes)}; expect roughly "
              f"{human(int(total_bytes * 0.35))}-{human(int(total_bytes * 0.45))} compressed "
              f"(JSON compresses ~6x, CSV ~2.3x).")
        return 0

    out_dir = Path(args.out).resolve() if args.out else (root.parent / "zenodo_bundle")
    try:
        out_dir.relative_to(root)
    except ValueError:
        pass
    else:
        fail(1, f"refusing to write inside the repository: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    final = out_dir / f"{stem}.{args.format}"
    partial = out_dir / f"{stem}.{args.format}.partial"
    print(f"\n[6/6] Writing {final} ...")

    prefix = stem
    try:
        _archive, add, close = open_archive(partial, args.format)
        try:
            for i, (src, rp) in enumerate(files, 1):
                add(f"{prefix}/{rp}", src=src)
                if i % 500 == 0:
                    print(f"    archived {i:,}/{len(files):,}", flush=True)
            add(f"{prefix}/MANIFEST.sha256", data=manifest.encode("utf-8"))
            add(f"{prefix}/BUNDLE_INFO.json",
                data=json.dumps(info, indent=2, ensure_ascii=False).encode("utf-8"))
        finally:
            close()
    except Exception:
        partial.unlink(missing_ok=True)     # never leave a partial that looks complete
        raise

    partial.replace(final)
    print(f"\nDone. {final}")
    print(f"  archive size : {human(final.stat().st_size)}")
    print(f"  raw contents : {human(total_bytes)} in {len(files):,} files")
    print(f"  ratio        : {total_bytes / max(final.stat().st_size, 1):.2f}x")
    print("\nUpload this single file to the Zenodo record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
#
# rewrite_bosch_authorship.sh — rewrite the 8 commits authored as
# <Yutaro.Nakagama@bosch.com> to an academic/personal identity.
#
# ############################################################################
# #  THIS SCRIPT IS NOT RUN AUTOMATICALLY. READ THIS HEADER BEFORE USING IT. #
# ############################################################################
#
# WHAT IT DOES
#   Uses `git filter-repo --mailmap` to rewrite author AND committer identity
#   across the whole history. Every commit from the old address gets the new
#   name/e-mail; all other commits are untouched in content but WILL receive new
#   SHAs from the rewrite point onward.
#
# WHAT IT COSTS — read all of it
#   1. EVERY COMMIT SHA FROM THE EARLIEST REWRITTEN COMMIT ONWARD CHANGES.
#      This history has 1,372 commits and the Bosch-authored ones are old, so
#      expect essentially the entire history to be renumbered.
#   2. The rewrite therefore requires a FORCE PUSH. Anyone else with a clone
#      must re-clone or hard-reset; ordinary `git pull` will conflict badly.
#   3. Existing references to specific commits break: the two paper submodules'
#      recorded pointers, any Zenodo/GitHub release archived from a commit, any
#      commit SHA quoted in a manuscript, issue, or BUNDLE_INFO.json.
#      >> If a Zenodo DOI has already been minted from this repository, DO NOT
#      >> RUN THIS. The archived snapshot would no longer correspond to any
#      >> commit that exists.
#   4. Signed commits lose their signatures.
#   5. git-filter-repo removes the `origin` remote by design after rewriting.
#
# ALTERNATIVE THAT COSTS NOTHING
#   A committed .mailmap file makes `git log`, `git shortlog` and GitHub's UI
#   display the corrected identity WITHOUT rewriting a single SHA. If the goal
#   is "the published history should not show a corporate address", the mailmap
#   is very often the right answer. This script writes one for you in step 3
#   and will stop there unless you pass --rewrite.
#
# PREREQUISITE
#   pip install git-filter-repo        # https://github.com/newren/git-filter-repo
#
# USAGE
#   bash scripts/shell/rewrite_bosch_authorship.sh                # audit + mailmap only (safe)
#   bash scripts/shell/rewrite_bosch_authorship.sh --rewrite      # actually rewrite history
#
set -euo pipefail

# ── Configuration — EDIT THESE ──────────────────────────────────────────────
OLD_EMAIL="Yutaro.Nakagama@bosch.com"
OLD_NAME="Nakagama Yutaro (VM-EPS/ESW1-JP)"

# Choose the identity the 8 commits should carry. Default: the academic address
# that already authors 1,081 of the 1,372 commits.
NEW_NAME="ynakagama"
NEW_EMAIL="ynakagama@jaist.ac.jp"
# ────────────────────────────────────────────────────────────────────────────

REWRITE=0
[[ "${1:-}" == "--rewrite" ]] && REWRITE=1

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "Repository : $REPO_ROOT"
echo "Rewriting  : ${OLD_NAME} <${OLD_EMAIL}>"
echo "         -> ${NEW_NAME} <${NEW_EMAIL}>"
echo

# ── 1. Audit: exactly which commits are affected? ───────────────────────────
echo "── 1. Affected commits ─────────────────────────────────────────────────"
mapfile -t AFFECTED < <(git log --all --format='%H' --author="$OLD_EMAIL")
echo "Author-side matches: ${#AFFECTED[@]}"
git log --all --format='  %h  %ad  %s' --date=short --author="$OLD_EMAIL" || true
echo
echo "Committer-side matches (may differ from author-side):"
git log --all --format='%H %cn <%ce>' | grep -F "$OLD_EMAIL" | wc -l || true
echo

if [[ ${#AFFECTED[@]} -eq 0 ]]; then
  echo "Nothing to rewrite. Exiting."
  exit 0
fi

EARLIEST="$(git log --all --format='%H' --author="$OLD_EMAIL" | tail -1)"
echo "Earliest affected commit: $EARLIEST"
echo "  -> every commit from here onward will get a new SHA if you rewrite."
echo "  -> commits that would be renumbered: $(git rev-list --count "${EARLIEST}"..HEAD 2>/dev/null || echo '?')"
echo

# ── 2. Safety checks ────────────────────────────────────────────────────────
echo "── 2. Safety checks ────────────────────────────────────────────────────"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "ABORT: working tree has modified tracked files. Commit or stash first."
  exit 1
fi
echo "  working tree      : clean"

if ! command -v git-filter-repo >/dev/null 2>&1 && ! git filter-repo --version >/dev/null 2>&1; then
  echo "  git-filter-repo   : NOT FOUND  (pip install git-filter-repo)"
  FILTER_REPO_OK=0
else
  echo "  git-filter-repo   : available"
  FILTER_REPO_OK=1
fi

if git submodule status --recursive 2>/dev/null | grep -q .; then
  echo "  submodules        : PRESENT — their recorded parent commits will change"
fi
echo "  current HEAD      : $(git rev-parse --short HEAD)"
echo

# ── 3. Write .mailmap (the no-cost alternative) ─────────────────────────────
echo "── 3. Writing .mailmap ─────────────────────────────────────────────────"
MAILMAP="$REPO_ROOT/.mailmap"
if grep -qF "$OLD_EMAIL" "$MAILMAP" 2>/dev/null; then
  echo "  .mailmap already maps $OLD_EMAIL — leaving it as is."
else
  printf '%s <%s> %s <%s>\n' \
    "$NEW_NAME" "$NEW_EMAIL" "$OLD_NAME" "$OLD_EMAIL" >> "$MAILMAP"
  echo "  appended mapping to .mailmap"
fi
echo
echo "  Verify the display identity now (no SHAs were harmed):"
echo "      git shortlog -sne --all | head"
echo "  Commit .mailmap and you are done — unless you specifically need the raw"
echo "  commit objects changed."
echo

# ── 4. The actual rewrite ───────────────────────────────────────────────────
if [[ $REWRITE -eq 0 ]]; then
  cat <<'EOF'
── 4. History rewrite: SKIPPED ─────────────────────────────────────────────
Re-run with --rewrite to perform it. Before you do:

  * Confirm no Zenodo DOI has been minted from a commit of this repository.
  * Make a full backup:
        git clone --mirror . ../vehicle_based_DDD_comparison-backup.git
  * Tell every collaborator they must re-clone afterwards.
EOF
  exit 0
fi

if [[ $FILTER_REPO_OK -eq 0 ]]; then
  echo "ABORT: --rewrite requested but git-filter-repo is unavailable."
  echo "       pip install git-filter-repo"
  exit 1
fi

echo "── 4. History rewrite ──────────────────────────────────────────────────"
BACKUP="../$(basename "$REPO_ROOT")-backup-$(date +%Y%m%d_%H%M%S).git"
echo "Creating mirror backup at: $BACKUP"
git clone --mirror . "$BACKUP"
echo "Backup complete."
echo
read -r -p "Rewrite ALL history now? Every SHA from $EARLIEST onward changes. [type REWRITE]: " CONFIRM
if [[ "$CONFIRM" != "REWRITE" ]]; then
  echo "Aborted by user. Backup kept at $BACKUP"
  exit 1
fi

# filter-repo's --mailmap rewrites both author and committer identities.
FILTER_MAILMAP="$(mktemp)"
printf '%s <%s> %s <%s>\n' \
  "$NEW_NAME" "$NEW_EMAIL" "$OLD_NAME" "$OLD_EMAIL" > "$FILTER_MAILMAP"
printf '%s <%s> <%s>\n' \
  "$NEW_NAME" "$NEW_EMAIL" "$OLD_EMAIL" >> "$FILTER_MAILMAP"

git filter-repo --mailmap "$FILTER_MAILMAP" --force
rm -f "$FILTER_MAILMAP"

echo
echo "── 5. Verification ─────────────────────────────────────────────────────"
REMAINING="$(git log --all --format='%an <%ae> %cn <%ce>' | grep -cF "$OLD_EMAIL" || true)"
echo "Remaining references to $OLD_EMAIL: $REMAINING  (expected 0)"
git shortlog -sne --all | head
echo
cat <<EOF
── 6. Manual steps that remain ─────────────────────────────────────────────
  1. git-filter-repo removed the 'origin' remote. Re-add it:
         git remote add origin git@github.com:YutaroNakagama/vehicle_based_DDD_comparison.git
  2. Inspect the result, then force-push every branch and tag:
         git push --force --all origin
         git push --force --tags origin
  3. Re-check out the submodules; their recorded parent commits changed:
         git submodule update --init --recursive
  4. Tell all collaborators to re-clone. A plain 'git pull' will not recover.
  5. If any commit SHA was quoted in a manuscript, release note, or
     BUNDLE_INFO.json, update it — the old SHAs no longer exist.

  Backup mirror (delete only once you are satisfied): $BACKUP
EOF

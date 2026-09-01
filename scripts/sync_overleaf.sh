#!/usr/bin/env bash
# PUSH writting_docs/paper TO OVERLEAF (2026-09-02).
#
# Overleaf gives every project a git remote, so the manual upload loop is avoidable.  The catch is
# that Overleaf's repository expects the project files at ITS root, while ours live in a subdirectory
# of the FAT repo, so this keeps a separate clone and copies into it.
#
# ONE-TIME SETUP
#   1. In the Overleaf project: Menu -> Git -> copy the URL, which looks like
#        https://git.overleaf.com/0123456789abcdef01234567
#   2. Create a git token: Account Settings -> Git integration -> "Generate token".
#      Username is "git", password is the token.
#   3.  git clone https://git.overleaf.com/<project-id> ~/overleaf-cfa
#
# THEN, after any edit here:
#   bash scripts/sync_overleaf.sh
#
# It copies the .tex, .bib, style/ and figure/ into the clone, commits and pushes.  Overleaf
# recompiles on its own.  Nothing is deleted there that is not in our tree, so anything added on the
# Overleaf side by hand survives.
set -eu
SRC="$(cd "$(dirname "$0")/.."; pwd)/writting_docs/paper"
DST="${OVERLEAF_DIR:-$HOME/overleaf-cfa}"

[ -d "$DST/.git" ] || { echo "no clone at $DST -- see the one-time setup in this file"; exit 1; }

rsync -a --delete-excluded \
  --include='*.tex' --include='*.bib' \
  --include='style/***' --include='figure/***' \
  --exclude='notes/***' --exclude='reference/***' --exclude='*.pdf' --exclude='*' \
  "$SRC/" "$DST/"

cd "$DST"
git add -A
if git diff --cached --quiet; then echo "nothing changed"; exit 0; fi
git commit -q -m "sync from FAT $(cd "$SRC" && git rev-parse --short HEAD) $(date '+%m-%d %H:%M')"
git push -q origin master 2>/dev/null || git push -q origin main
echo "pushed to Overleaf; it will recompile on its own"

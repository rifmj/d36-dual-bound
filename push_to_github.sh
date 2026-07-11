#!/usr/bin/env bash
# One-shot: initialize git here and publish to github.com/rifmj/d36-dual-bound
# Requires: gh (GitHub CLI) already authenticated as rifmj  ->  gh auth status
set -euo pipefail
cd "$(dirname "$0")"

# 1. init a fresh repo in-place (idempotent)
if [ ! -d .git ]; then
  git init -q
  git branch -m main
fi

# 2. stage & commit (skip if nothing changed)
git add -A
if git diff --cached --quiet; then
  echo "nothing to commit"
else
  git -c user.name="Rifat Jumagulov" -c user.email="jum.rifm@gmail.com" \
      commit -q -m "Initial commit: d=36 dual LP bound — paper, code, and exact certificate"
fi

# 3. create the remote repo (if absent) and push
if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
else
  gh repo create rifmj/d36-dual-bound --public --source=. --remote=origin --push \
    --description "A dual linear programming bound for sphere packing in dimension 36 — paper, code, and exact certificate"
fi

echo "Done -> https://github.com/rifmj/d36-dual-bound"

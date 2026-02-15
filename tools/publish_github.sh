#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <github_repo_url>"
  echo "Example: $0 git@github.com:<user>/fusion.git"
  exit 1
fi

REPO_URL="$1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git push -u origin main

echo "Published: $REPO_URL"
echo "Branch: main"

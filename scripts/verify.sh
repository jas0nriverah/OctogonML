#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${CAGEMETRICS_PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then echo "Missing .venv; see ./start.sh setup instructions" >&2; exit 1; fi
command -v node >/dev/null || { echo "Node.js is required for JavaScript syntax checks" >&2; exit 1; }
"$PYTHON" -m ruff check src tests migrations scripts
if [[ $# == 0 ]]; then
  "$PYTHON" -m pytest --cov=cagemetrics --cov-report=term-missing
else
  "$PYTHON" -m pytest "$@"
fi
node --check src/cagemetrics/web/app.js

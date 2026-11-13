#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${CAGEMETRICS_PYTHON:-.venv/bin/python}"
if [[ "${1:-}" == "--help" ]]; then
  echo "Usage: ./start.sh [--check | serve options]"
  echo "--check validates offline assets without SQL writes, downloads, training, or LLM access."
  exit 0
fi
setup_error() {
  echo "Missing or incomplete Python environment. Run: python3.12 -m venv .venv" >&2
  echo "Then: .venv/bin/python -m pip install -r requirements-lock.txt -e '.[dev]'" >&2
  exit 1
}
if [[ ! -x "$PYTHON" ]]; then setup_error; fi
if [[ "${1:-}" == "--check" && $# != 1 ]]; then
  echo "--check does not accept additional arguments" >&2
  exit 2
fi
if ! "$PYTHON" - <<'PY'
import importlib.util
import sys
required = ("cagemetrics", "fastapi", "uvicorn", "sqlalchemy", "alembic", "pandas",
            "numpy", "scipy", "sklearn", "joblib", "httpx", "threadpoolctl")
sys.exit(0 if all(importlib.util.find_spec(name) is not None for name in required) else 1)
PY
then
  setup_error
fi
if [[ "${1:-}" != "--check" ]]; then
  # Reject malformed startup arguments before bootstrap can write local SQL.
  "$PYTHON" - "$@" <<'PY'
import argparse
import os
parser = argparse.ArgumentParser(description="Start OctogonML after offline validation")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=os.environ.get("PORT", "8000"))
parser.add_argument("--llm-provider", choices=("ollama", "openai"))
args = parser.parse_args()
if not 1 <= args.port <= 65535:
    parser.error("--port must be between 1 and 65535")
PY
fi
"$PYTHON" -m cagemetrics.cli doctor
if [[ "${1:-}" == "--check" ]]; then exit 0; fi
"$PYTHON" -m cagemetrics.cli bootstrap
exec "$PYTHON" -m cagemetrics.cli serve "$@"

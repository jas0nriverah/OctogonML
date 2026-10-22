#!/bin/sh
# Workspace Ollama runtime; no global installer, API key, or hosted model call.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORKSPACE_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
LOCAL_LLM_ROOT=${CAGEMETRICS_LOCAL_LLM_DIR:-"$WORKSPACE_ROOT/work/local-llm"}
if [ ! -x "$LOCAL_LLM_ROOT/runtime/ollama" ]; then
    echo "Missing verified Ollama runtime: $LOCAL_LLM_ROOT/runtime/ollama" >&2
    exit 1
fi
exec env OLLAMA_MODELS="$LOCAL_LLM_ROOT/models" OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_NO_CLOUD=1 OLLAMA_NOHISTORY=1 "$LOCAL_LLM_ROOT/runtime/ollama" serve

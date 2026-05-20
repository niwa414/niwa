#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
MCP_LAUNCHER="$PROJECT_ROOT/bin/mietclaw-mcp"
SERVER_NAME="${1:-mietclaw}"

CODEX_BIN="${CODEX_BIN:-}"
if [ -z "$CODEX_BIN" ]; then
  if [ -x /Applications/Codex.app/Contents/Resources/codex ]; then
    CODEX_BIN=/Applications/Codex.app/Contents/Resources/codex
  else
    CODEX_BIN="$(command -v codex || true)"
  fi
fi

if [ -z "$CODEX_BIN" ] || [ ! -x "$CODEX_BIN" ]; then
  echo "Codex CLI not found. Set CODEX_BIN or install Codex first." >&2
  exit 1
fi

if "$CODEX_BIN" mcp get "$SERVER_NAME" >/dev/null 2>&1; then
  "$CODEX_BIN" mcp remove "$SERVER_NAME"
fi

"$CODEX_BIN" mcp add "$SERVER_NAME" -- "$MCP_LAUNCHER"
"$CODEX_BIN" mcp get "$SERVER_NAME"

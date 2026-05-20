#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_BIN_DIR="${HOME}/.local/bin"
TARGET_BIN="${TARGET_BIN_DIR}/mietclaw-mcp"

mkdir -p "$TARGET_BIN_DIR"
ln -sf "$PROJECT_ROOT/bin/mietclaw-mcp" "$TARGET_BIN"

ensure_path_line() {
  FILE_PATH="$1"
  LINE='export PATH="$HOME/.local/bin:$PATH"'
  if [ ! -f "$FILE_PATH" ]; then
    printf '%s\n' "$LINE" > "$FILE_PATH"
    return
  fi
  if ! grep -Fq "$LINE" "$FILE_PATH"; then
    printf '\n%s\n' "$LINE" >> "$FILE_PATH"
  fi
}

ensure_path_line "${HOME}/.zshrc"
ensure_path_line "${HOME}/.bashrc"

printf 'mietclaw-mcp launcher installed at %s\n' "$TARGET_BIN"
printf 'Open a new shell or run: export PATH=\"$HOME/.local/bin:$PATH\"\n'

#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-$HOME/.openclaw/openclaw.json}"
OPENCLAW_BIN="${OPENCLAW_BIN:-$HOME/.local/bin/openclaw}"
MODEL_ID="omlx/Huihui-Qwen3.5-27B-Claude-4.6-Opus-abliterated-4bit"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

if [ ! -x "$OPENCLAW_BIN" ]; then
  echo "OpenClaw CLI not found at $OPENCLAW_BIN" >&2
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "OpenClaw config not found at $CONFIG_PATH" >&2
  exit 1
fi

BACKUP_PATH="${CONFIG_PATH}.bak.$(date +%Y%m%d_%H%M%S)"
cp "$CONFIG_PATH" "$BACKUP_PATH"

python3 - "$CONFIG_PATH" "$MODEL_ID" "$PROJECT_ROOT" "$PYTHON_BIN" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
model_id = sys.argv[2]
project_root = Path(sys.argv[3]).resolve()
python_bin = sys.argv[4]

mcp_server = {
    "command": python_bin,
    "args": [
        "-m",
        "miet_claw.cli",
        "mcp-server",
        "--project-root",
        str(project_root),
        "--workspace-root",
        str(project_root / ".autonomy-mcp"),
        "--output-dir",
        str(project_root / "runs"),
        "--provider",
        "local",
    ],
    "env": {
        "PYTHONPATH": str(project_root / "src"),
    },
}

data = json.loads(config_path.read_text(encoding="utf-8"))

data.setdefault("agents", {}).setdefault("defaults", {}).setdefault("model", {})["primary"] = model_id
data.setdefault("acp", {})["backend"] = "acpx"
data["acp"]["enabled"] = True

plugins = data.setdefault("plugins", {})
allow = plugins.setdefault("allow", [])
for item in ["miet-claw-sim", "acpx"]:
    if item not in allow:
        allow.append(item)

entries = plugins.setdefault("entries", {})
entries.setdefault("miet-claw-sim", {})["enabled"] = True
acpx_entry = entries.setdefault("acpx", {})
acpx_entry["enabled"] = True
acpx_config = acpx_entry.setdefault("config", {})
acpx_config.setdefault("mcpServers", {})["mietclaw"] = mcp_server

if "mcp" in data:
    data["mcp"].pop("servers", None)
    if not data["mcp"]:
        data.pop("mcp", None)

tools = data.setdefault("tools", {})
tool_allow = tools.setdefault("allow", [])
if "miet-claw-sim" not in tool_allow:
    tool_allow.append("miet-claw-sim")

config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

"$OPENCLAW_BIN" config validate
echo "Updated $CONFIG_PATH"
echo "Backup saved to $BACKUP_PATH"
echo "Primary model: $MODEL_ID"
echo "MCP server command: $PYTHON_BIN -m miet_claw.cli mcp-server"

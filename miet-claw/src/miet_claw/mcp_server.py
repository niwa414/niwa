from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .autonomy import detect_project_root, materialize_autonomy_workspace, run_autonomy_job
from .bridge import BridgeError, run_kmc_lookup_bridge
from .chat import (
    format_artifact_report,
    format_bridge_report,
    format_draft_report,
    format_inspect_report,
    format_log_report,
    format_moire_compare_report,
    format_moire_diffusion_sweep_report,
    format_moire_kmc_report,
    format_moire_lammps_report,
    format_moire_workflow_report,
    format_run_list,
    format_run_report,
    get_local_model_status,
    get_log_excerpt,
    inspect_run,
    list_artifacts,
    list_runs,
)
from .executor import run_job
from .local_profile import get_runtime_settings
from .moire_runtime import (
    MoReWorkflowError,
    run_moire_diffusion_sweep,
    run_moire_event_compare,
    run_moire_lammps_case,
    run_moire_lammps_to_kmc,
    run_moire_repo_kmc,
)
from .planner import build_plan_payload
from .shell_runtime import collect_runtime_doctor, format_runtime_doctor
from .runtime.tool_dispatch import dispatch_mcp_tool
from .specs import load_job_spec
from .runtime.tool_registry import mcp_tool_definitions


DEFAULT_PROTOCOL_VERSION = "2025-03-26"


class MCPServerError(RuntimeError):
    pass


def _trace(message: str) -> None:
    trace_path = os.environ.get("MIETCLAW_MCP_TRACE")
    if not trace_path:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(trace_path).open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def _json_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _tool_result(text: str, structured: Optional[Dict[str, Any]] = None, is_error: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    }
    if structured is not None:
        result["structuredContent"] = structured
    return result


def _resolve_run_dir(output_dir: Path, run_dir: Optional[str] = None, run_name: Optional[str] = None) -> Path:
    if run_dir:
        candidate = Path(run_dir).expanduser()
        if not candidate.exists():
            raise MCPServerError(f"run_dir not found: {candidate}")
        return candidate.resolve()
    if run_name:
        candidate = (output_dir / run_name).resolve()
        if not candidate.exists():
            raise MCPServerError(f"run_name not found under output_dir: {run_name}")
        return candidate
    items = list_runs(output_dir, limit=1)
    if not items:
        raise MCPServerError("No runs found.")
    return Path(items[0]["path"]).resolve()


class MietClawMCPServer:
    def __init__(
        self,
        *,
        project_root: str,
        workspace_root: str,
        output_dir: str,
        provider: str = "local",
    ) -> None:
        self.project_root = str(detect_project_root(Path(project_root)))
        self.workspace_root = str(Path(workspace_root).resolve())
        self.output_dir = str(Path(output_dir).resolve())
        self.provider = provider
        self.initialized = False
        self.protocol_version = DEFAULT_PROTOCOL_VERSION

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return mcp_tool_definitions()

    def _dispatch_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return dispatch_mcp_tool(self, name, arguments, api=globals())

    def handle_request(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        method = message.get("method")
        params = message.get("params") or {}
        request_id = message.get("id")
        _trace(f"recv method={method} id={request_id}")

        if method == "notifications/initialized":
            self.initialized = True
            return None

        if method == "initialize":
            self.protocol_version = str(params.get("protocolVersion") or DEFAULT_PROTOCOL_VERSION)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "mietclaw-mcp", "version": "0.2.0"},
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": self.tool_definitions()}}

        if method == "tools/call":
            try:
                result = self._dispatch_tool(str(params.get("name")), params.get("arguments") or {})
                return {"jsonrpc": "2.0", "id": request_id, "result": result}
            except Exception as exc:  # noqa: BLE001
                text = f"{type(exc).__name__}: {exc}"
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": _tool_result(text, {"error": text, "traceback": traceback.format_exc()}, is_error=True),
                }

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    @staticmethod
    def _read_message() -> Optional[Tuple[Dict[str, Any], str]]:
        first_line = sys.stdin.buffer.readline()
        if not first_line:
            return None

        if first_line.lstrip().startswith((b"{", b"[")):
            message = json.loads(first_line.decode("utf-8"))
            _trace("read jsonl")
            return message, "jsonl"

        headers: Dict[str, str] = {}
        line = first_line
        while True:
            if line in {b"\r\n", b"\n"}:
                break
            try:
                name, value = line.decode("utf-8").split(":", 1)
            except ValueError as exc:  # noqa: BLE001
                raise MCPServerError(f"Malformed header line: {line!r}") from exc
            headers[name.strip().lower()] = value.strip()
            line = sys.stdin.buffer.readline()
            if not line:
                return None

        try:
            content_length = int(headers["content-length"])
        except KeyError as exc:
            raise MCPServerError("Missing Content-Length header") from exc
        body = sys.stdin.buffer.read(content_length)
        if not body:
            return None
        message = json.loads(body.decode("utf-8"))
        _trace(f"read content_length={content_length}")
        return message, "headers"

    @staticmethod
    def _write_message(message: Dict[str, Any], framing: str = "headers") -> None:
        _trace(f"send keys={sorted(message.keys())} id={message.get('id')} framing={framing}")
        payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
        if framing == "jsonl":
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.write(b"\n")
        else:
            header = f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
            sys.stdout.buffer.write(header)
            sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()

    def serve(self) -> int:
        _trace("server_start")
        while True:
            read_result = self._read_message()
            if read_result is None:
                _trace("server_eof")
                return 0
            message, framing = read_result
            response = self.handle_request(message)
            if response is not None:
                self._write_message(response, framing=framing)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mietclaw local MCP server")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--workspace-root", default=str(Path(".autonomy-mcp").resolve()))
    parser.add_argument("--output-dir", default=str(Path("runs").resolve()))
    parser.add_argument("--provider", default="local")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    server = MietClawMCPServer(
        project_root=args.project_root,
        workspace_root=args.workspace_root,
        output_dir=args.output_dir,
        provider=args.provider,
    )
    return server.serve()


if __name__ == "__main__":
    raise SystemExit(main())

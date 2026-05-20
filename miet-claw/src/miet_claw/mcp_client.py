from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_PROTOCOL_VERSION = "2025-03-26"


class MCPClientError(RuntimeError):
    pass


def _build_env(project_root: Path) -> Dict[str, str]:
    env = dict(os.environ)
    src_dir = project_root / "src"
    if src_dir.exists():
        parts = [str(src_dir)]
        if env.get("PYTHONPATH"):
            parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


class LocalMCPClient:
    def __init__(
        self,
        *,
        project_root: str,
        workspace_root: str,
        output_dir: str,
        provider: str = "local",
        python_executable: Optional[str] = None,
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.provider = provider
        self.python_executable = python_executable or sys.executable
        self.proc: Optional[subprocess.Popen[bytes]] = None
        self._next_id = 1

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def connect(self) -> "LocalMCPClient":
        if self.is_running():
            return self
        self.close()
        try:
            self.proc = subprocess.Popen(
                [
                    self.python_executable,
                    "-m",
                    "miet_claw.cli",
                    "mcp-server",
                    "--project-root",
                    str(self.project_root),
                    "--workspace-root",
                    str(self.workspace_root),
                    "--output-dir",
                    str(self.output_dir),
                    "--provider",
                    self.provider,
                ],
                cwd=str(self.project_root),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_build_env(self.project_root),
            )
            self._next_id = 1
            self._initialize()
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait(timeout=5)
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

    def __enter__(self) -> "LocalMCPClient":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _initialize(self) -> None:
        response = self._request(
            "initialize",
            {
                "protocolVersion": DEFAULT_PROTOCOL_VERSION,
                "clientInfo": {"name": "mietclaw-shell", "version": "0.1.0"},
            },
        )
        if response.get("result", {}).get("serverInfo", {}).get("name") != "mietclaw-mcp":
            raise MCPClientError("Unexpected MCP server handshake.")
        self._notify("notifications/initialized", {})

    def _notify(self, method: str, params: Dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        response = self._read()
        if response.get("id") != request_id:
            raise MCPClientError("MCP response id mismatch.")
        if "error" in response:
            raise MCPClientError(str(response["error"]))
        return response

    def _send(self, payload: Dict[str, Any]) -> None:
        if self.proc is None or self.proc.stdin is None:
            raise MCPClientError("MCP process is not running.")
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.proc.stdin.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("utf-8"))
        self.proc.stdin.write(raw)
        self.proc.stdin.flush()

    def _read(self) -> Dict[str, Any]:
        if self.proc is None or self.proc.stdout is None:
            raise MCPClientError("MCP process is not running.")
        headers: Dict[str, str] = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                stderr = ""
                if self.proc.stderr is not None:
                    stderr = self.proc.stderr.read().decode("utf-8", errors="replace")
                raise MCPClientError(f"MCP server closed unexpectedly. {stderr}".strip())
            if line in {b"\r\n", b"\n"}:
                break
            name, value = line.decode("utf-8").split(":", 1)
            headers[name.strip().lower()] = value.strip()
        size = int(headers["content-length"])
        body = self.proc.stdout.read(size)
        if not body:
            raise MCPClientError("MCP server returned an empty body.")
        return json.loads(body.decode("utf-8"))

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self.connect()
        try:
            response = self._request("tools/call", {"name": name, "arguments": arguments})
        except MCPClientError:
            if self.is_running():
                raise
            self.connect()
            response = self._request("tools/call", {"name": name, "arguments": arguments})
        result = response.get("result") or {}
        if result.get("isError"):
            message = ""
            contents = result.get("content") or []
            if contents and isinstance(contents[0], dict):
                message = str(contents[0].get("text") or "")
            if not message:
                message = str((result.get("structuredContent") or {}).get("error") or "MCP tool call failed")
            raise MCPClientError(message)
        return result


def call_local_mcp_tool(
    *,
    tool_name: str,
    arguments: Dict[str, Any],
    project_root: str,
    workspace_root: str,
    output_dir: str,
    provider: str = "local",
    python_executable: Optional[str] = None,
) -> Dict[str, Any]:
    with LocalMCPClient(
        project_root=project_root,
        workspace_root=workspace_root,
        output_dir=output_dir,
        provider=provider,
        python_executable=python_executable,
    ) as client:
        return client.call_tool(tool_name, arguments)
